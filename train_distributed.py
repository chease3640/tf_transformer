#!/usr/bin/env python
# coding: utf-8
"""This module for model distributed training."""
import datetime

import tensorflow as tf

from config import FLAGS
from dataset import Dataset, DataSchemaEnum
from model import AutoEncoder
from utils import get_optimizer_instance

params = FLAGS


def train():
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    is_chief = FLAGS.task_index == 0
    # construct the servers
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    worker_count = len(worker_hosts)
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    # start the training
    try:
        worker_device = "/job:worker/task:%d" % FLAGS.task_index
        print("worker_deivce = %s" % worker_device)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        # assign io related variables and ops to local worker device
        with tf.device(worker_device):
            iterator = Dataset(
                DataSchemaEnum.train, FLAGS.train_file, FLAGS,
                slice_count=worker_count, slice_id=FLAGS.task_index).get_iterator()
        # assign global variables to ps nodes
        with tf.device(tf.train.replica_device_setter(
                worker_device=worker_device, cluster=cluster)):
            model = AutoEncoder(iterator, FLAGS, tf.estimator.ModeKeys.TRAIN)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = params.lr
            if params.use_lr_decay:
                # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
                learning_rate = tf.train.exponential_decay(
                    learning_rate=params.lr,
                    global_step=global_step,
                    decay_steps=params.decay_steps,
                    decay_rate=params.decay_rate,
                    staircase=False)
            opt = get_optimizer_instance(params.optimizer, learning_rate)

            if params.is_sync:
                opt = tf.train.SyncReplicasOptimizer(
                    opt,
                    replicas_to_aggregate=int(0.9*worker_count),
                    total_num_replicas=worker_count,
                    use_locking=True)
                sync_replicas_hook = opt.make_session_run_hook(is_chief)

            train_var_list = tf.trainable_variables()

            gradients = tf.gradients(model.loss, train_var_list)
            if params.use_grad_clip:
                gradients, grad_norm = tf.clip_by_global_norm(gradients, params.grad_clip_norm)
            train_ops = opt.apply_gradients(zip(gradients, train_var_list), global_step=global_step)
        # The StopAtStepHook handles stopping after running given steps.
        if params.is_sync:
            hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps), sync_replicas_hook]
        else:
            hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps)]
        step = 0
        print("Start training")
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=is_chief,
                save_checkpoint_secs=FLAGS.save_checkpoint_secs,
                checkpoint_dir=FLAGS.checkpointDir,
                hooks=hooks,
                config=config
        ) as mon_sess:
            # Don't write inside the loop, or it will reinitialize and fill buffer size all the time.
            mon_sess.run(iterator.initializer)
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                step += 1
                _, g, loss, pred, label = mon_sess.run([train_ops, global_step, model.loss])
                # if step % (FLAGS.show_loss_per_steps // worker_count) == 0:
                if step % FLAGS.show_loss_per_steps == 0:

                    time_str = datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S')
                    print("[%s] step %d, global_step %d, loss %.4f" % (
                        time_str, step, g, loss))
        print("%d steps finished." % step)
    except Exception, e:
        print("catch a exception: %s" % e.message)

