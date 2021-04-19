#!/usr/bin/env python
# coding: utf-8
"""This module for model prediction."""
import time
import os
import tensorflow as tf

from config import FLAGS
from dataset import Dataset, DataSchemaEnum
from model import AutoEncoder
from utils import print_args, load_model, speed


def predict():
    if FLAGS.source == "xxx":
        # 该write方法会定义op，每次循环图会增加op，放到循环外面，否则越来越慢，直至报错
        # writer = tf.TableRecordWriter(FLAGS.outputs, slice_id=FLAGS.task_index)
        # batch_id = tf.placeholder(tf.string, [None, 1])
        # batch_embedding_str = tf.placeholder(tf.string, [None, 1])
        # write_to_table = writer.write(indices=[0, 1], values=[batch_id, batch_embedding_str])

        # 推荐使用，不涉及图，10x faster than tf.TableRecordWriter
        writer = tf.python_io.TableWriter(FLAGS.outputs, slice_id=FLAGS.task_index)
    else:
        writer = open(FLAGS.outputs, "w")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # load model
        load_model(sess, FLAGS.checkpointDir)
        sess.run(iterator.initializer)
        # tf.get_default_graph().finalize()
        print('Start Predicting...')
        step = 0
        t0 = time.time()
        while True:
            try:
                batch_id, batch_embedding = sess.run([iterator.id, model.encode])
                batch_embedding_str = [",".join(map(str, embeddings)) for embeddings in batch_embedding]
                if FLAGS.source == "odps":
                    # sess.run(write_to_table, feed_dict={...})
                    writer.write(values=zip(batch_id, batch_embedding_str), indices=[0, 1])
                else:
                    for id_, embedding_str in zip(batch_id, batch_embedding_str):
                        writer.write("\t".join([id_, embedding_str]) + "\n")
                step += 1
                if step % 10 == 0:
                    samples = step * FLAGS.predict_batch_size
                    s = int(speed(samples, t0))
                    now_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                    print('{} speed {} records/s\t predict {:2d} lines'.format(now_time, s, samples))

            except tf.errors.OutOfRangeError:
                break
        writer.close()
        print("Done. Write output into {}".format(FLAGS.outputs))


if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(FLAGS.tf_cpp_min_log_level)  # Defaults to 0, 1 to INFO, 2 WARNING, 3 ERROR
    tf.logging.set_verbosity(FLAGS.verbosity)
    tf.logging.info("Using TensorFlow Version %s" % tf.VERSION)
    FLAGS.batch_size = FLAGS.predict_batch_size
    print_args(FLAGS)

    # Model Preparation
    mode = tf.estimator.ModeKeys.PREDICT

    if FLAGS.predict_distributed:
        print("job name = %s" % FLAGS.job_name)
        print("task index = %d" % FLAGS.task_index)
        is_chief = FLAGS.task_index == 0
        # construct the servers
        ps_spec = FLAGS.ps_hosts.split(",")
        worker_spec = FLAGS.worker_hosts.split(",")
        cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
        worker_count = len(worker_spec)
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        iterator = Dataset(DataSchemaEnum.predict, FLAGS.predict_file, FLAGS, slice_count=worker_count,
                           slice_id=FLAGS.task_index).get_iterator()
        model = AutoEncoder(iterator, FLAGS, mode)

        # join the ps server
        if FLAGS.job_name == "ps":
            server.join()
        predict()

    else:
        iterator = Dataset(DataSchemaEnum.predict, FLAGS.predict_file, FLAGS).get_iterator()
        model = AutoEncoder(iterator, FLAGS, mode)
        predict()




