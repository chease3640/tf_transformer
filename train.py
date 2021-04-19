#!/usr/bin/env python
"""This module for model training."""
import os
import time

import tensorflow as tf

from config import FLAGS
from dataset import Dataset, DataSchemaEnum
from model import AutoEncoder
from utils import print_args, load_model, speed, get_optimizer_instance


TRAIN_LOG_TEMPLATE = '{} step={:3d} speed={:5d} lr={:.4f}\ttrain loss={:.4f} accuracy={:.4f}'


def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 1. Model Preparation.
        train_iterator = Dataset(DataSchemaEnum.train, params.train_file, params).get_iterator()
        train_model = AutoEncoder(train_iterator, params, tf.estimator.ModeKeys.TRAIN)
        train_model.model_stats()

        saver = tf.train.Saver(max_to_keep=params.num_keep_ckpts)
        # Keep training.
        if params.resume:
            # load model
            load_model(sess, params.checkpointDir)
            params.lr = params.lr / 10
            print("Resume learning rate: {} devided 10 by initial learning rate".format(params.lr))

        # 2. Define train ops.
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

        train_var_list = tf.trainable_variables()
        gradients = tf.gradients(train_model.loss, train_var_list)
        if params.use_grad_clip:
            gradients, grad_norm = tf.clip_by_global_norm(gradients, params.grad_clip_norm)
        train_ops = opt.apply_gradients(zip(gradients, train_var_list), global_step=global_step)

        # 3. Run initial ops.
        init_ops = [
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            tf.tables_initializer(),
            train_iterator.initializer,
        ]
        sess.run(init_ops)
        # 4. Train.
        step = 0
        t0 = time.time()
        while True:
            try:
                sess.run(train_ops)
                step += 1
                # show train batch metrics
                if step % params.show_loss_per_steps == 0:
                    lr, loss, accuracy = sess.run([learning_rate, train_model.loss, train_model.accuracy])
                    now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                    # time_str = datetime.datetime.now().isoformat()
                    # print(m.tp, m.fp, m.tn, m.fn)
                    samples = step * FLAGS.batch_size
                    s = int(speed(samples, t0))
                    print(TRAIN_LOG_TEMPLATE.format(now_time, step, s, lr, loss, accuracy))
                # save model
                if params.save_model and step % FLAGS.save_per_steps == 0:
                    model_name = "model_{}".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
                    path = os.path.join(FLAGS.checkpointDir, model_name)
                    saver.save(sess, path, global_step=global_step.eval())
                    print("Export checkpoint with to {}".format(path))
                if params.max_steps and step >= params.max_steps:
                    break
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(FLAGS.tf_cpp_min_log_level)  # Defaults to 0, 1 to INFO, 2 WARNING, 3 ERROR
    tf.logging.set_verbosity(FLAGS.verbosity)
    tf.logging.info("Using TensorFlow Version %s" % tf.VERSION)  # or tf.__version__
    tf.set_random_seed(FLAGS.random_seed)
    print_args(FLAGS)
    params = FLAGS
    if params.train_distributed:
        # Train distributed
        from train_distributed import train
        train()
    else:
        train()

