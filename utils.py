#!/usr/bin/env python
# coding: utf-8

"""This module contains some utility functions."""
from __future__ import division

import time
import tensorflow as tf
import re


def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    # tf_ver = [int(s) for s in tf.__version__.split('.')]
    # if tf_ver[1] <= 4:
    #     # only work for tensorflow<=1.4
    #     flags._parse_flags()
    #     for attr, value in sorted(flags.__flags.items()):
    #         print("{}={}".format(attr, value))
    # else:
    for attr in flags:
        value = flags[attr].value
        print("\t{}={}".format(attr, value))
    print("")


def speed(count, t):
    """Compute speed, units records/s."""
    elapsed_time = time.time() - t
    return count / elapsed_time


def get_optimizer_instance(opt, learning_rate=None):
    """Returns an optimizer instance."""
    _OPTIMIZER_CLS_NAMES = {
        'sgd': tf.train.GradientDescentOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'adam': tf.train.AdamOptimizer,
    }
    if opt in _OPTIMIZER_CLS_NAMES:
        if learning_rate is None:
            raise ValueError('learning_rate must be specified.')
        return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer %s" % opt)


def load_vocab(source, table):
    """Load vocab table.
    Args:
        table: vocab table
        source: `odps` or `local`
    Returns:
        vocab_table, vocab, vocab_size
    """
    if source == "odps":
        return _load_vocab_odps(table)
    elif source == "local":
        return _load_vocab_local(table)
    return


def _load_vocab_odps(table):
    """Load vocab from odps vocab table.
    Args:
        table: odps vocab table
    Returns:
        vocab, vocab_size
    """
    with tf.python_io.TableReader(
            table, selected_cols="", excluded_cols="", slice_id=0, slice_count=1) as reader:
        vocab_size = reader.get_row_count()
        vocab = [w[1] for w in reader.read(vocab_size)]
        print("Load {} vocab from {}".format(vocab_size, table))
        print("Vocab samples: %s" % vocab[:10])
    return vocab, vocab_size


def _load_vocab_local(table):
    """Load vocab from local vocab table.
    Args:
        table: local vocab table
    Returns:
        vocab, vocab_size
    """
    vocab_size = 0
    vocab = []
    with open(table, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))
            vocab_size += 1
    tf.logging.info("Load {} vocab from {}".format(vocab_size, table))
    return vocab, vocab_size


def load_model(sess, ckpt):
    with sess.as_default():
        with sess.graph.as_default():
            init_ops = [tf.global_variables_initializer(),
                        tf.local_variables_initializer(), tf.tables_initializer()]
            sess.run(init_ops)
            # load saved model
            ckpt_path = tf.train.latest_checkpoint(ckpt)
            if ckpt_path:
                tf.logging.info("Loading saved model: " + ckpt_path)
            else:
                raise ValueError("No checkpoint found in directory {}".format(ckpt))

            # reader = tf.train.NewCheckpointReader(ckpt+'model.ckpt_0.876-580500')
            # variables = reader.get_variable_to_shape_map()
            # for v in variables:
            #     print(v)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)


def is_en_character(char):
    if re.search(u'[a-zA-Z]', char):
        return True
    return False


def is_not_en_character(char):
    if re.search(u'[0-9\u4e00-\u9fa5]', char):
        return True
    return False


def segment_text_for_entity_alignment(text):
    if text is None:
        return ""

    text = text.lower().decode('utf-8')
    text = re.sub(u'[^a-zA-Z0-9\u4e00-\u9fa5]', ' ', text).strip()
    if len(text) <= 1:
        return text

    # add blank between cn and en
    i = 1
    while i < len(text):
        if (is_en_character(text[i-1]) and is_not_en_character(text[i])) or (is_not_en_character(text[i-1]) and is_en_character(text[i])):
            text = text[:i] + " " + text[i:]
            i += 1
        i += 1

    word_list = list()
    for word in text.split():
        if is_not_en_character(word):
            word_list += list(word)
        else:
            word_list.append(word)
    return ' '.join(word_list)

