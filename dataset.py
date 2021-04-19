#!/usr/bin/env python
# coding: utf-8
"""This module contains efficient data read and transform using tf.data API.

There are total 2 table schema for model training and predict as follows:
1. Input train data schema (1 fields):
    id   STRING
    x1   STRING
    x2   STRING

2. Input predict data schema (2 fields):
    id   STRING
    x    STRING

Notes: The order can not be changed, fields separator is \t.
"""

import random
from enum import Enum
import collections

import tensorflow as tf


class DataSchemaEnum(Enum):
    train = ("", "", "")
    predict = ("", "")


class BatchedInput(
    collections.namedtuple(
        "BatchedInput", ("initializer", "id", "src", "tgt", "label"))):  # 4 attributes
    pass


class Dataset(object):

    def __init__(self, schema, data, params, slice_id=0, slice_count=1):
        """
        Args:
            schema: Instance of DataSchemaEnum class.
            data: Data file string.
            params: Parameters defined in config.py.
            slice_count: Used in distributed settings.
            slice_id: Used in distributed settings.
        """
        assert isinstance(schema, DataSchemaEnum)
        self.schema = schema
        self.source = params.source.lower()
        self.data = data
        self.num_epochs = params.num_epochs
        self.batch_size = params.batch_size
        self.seq_max_len = params.seq_max_len
        self.shuffle_buffer_size = params.num_samples or params.batch_size * 1000
        # self.reshuffle_each_iteration = True  # Defaults to True
        self.random_seed = params.random_seed
        self.slice_count = slice_count
        self.slice_id = slice_id

        # Any out-of-vocabulary token will return a bucket ID based on its hash if num_oov_buckets is greater than zero.
        # Otherwise the default_value. The bucket ID range is [vocabulary size, vocabulary size + num_oov_buckets - 1].
        self.vocab_table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=params.vocab_file, num_oov_buckets=0, default_value=0)

        if self.schema == DataSchemaEnum.train:
            self._transform_func = self._transform
        else:
            self._transform_func = self._transform_predict

    def get_iterator(self):
        record_defaults = self.schema.value
        if self.source == "local":
            dataset = tf.data.TextLineDataset(self.data).map(
                lambda line: tf.decode_csv(
                    line, record_defaults=record_defaults, field_delim='\t', use_quote_delim=False))
        if self.source == "xxx":
            dataset = tf.data.TableRecordDataset(
                self.data, record_defaults=record_defaults, slice_count=self.slice_count, slice_id=self.slice_id)
        # Data transform
        dataset = dataset.map(self._transform_func, num_parallel_calls=4).prefetch(2*self.batch_size)
        if self.schema == DataSchemaEnum.train:
            dataset = dataset.shuffle(self.shuffle_buffer_size, self.random_seed).repeat(self.num_epochs)

        batched_dataset = self._batching_func(dataset)
        batched_iter = batched_dataset.make_initializable_iterator()  # sess.run(iterator.initializer)
        if self.schema == DataSchemaEnum.train:
            src, tgt, label = batched_iter.get_next()
            return BatchedInput(initializer=batched_iter.initializer, id=None, src=src, tgt=tgt, label=label)
        else:
            id, src = batched_iter.get_next()
            return BatchedInput(initializer=batched_iter.initializer, id=id, src=src, tgt=None, label=None)

    def _transform(self, x1, x2, label):
        src = tf.cast(self.vocab_table.lookup(tf.string_split([x1]).values[:self.seq_max_len]), tf.int32)
        tgt = tf.cast(self.vocab_table.lookup(tf.string_split([x2]).values[:self.seq_max_len]), tf.int32)
        label = tf.string_to_number(label, out_type=tf.float32)
        return src, tgt, label

    def _transform_predict(self, id, x):
        src = tf.cast(self.vocab_table.lookup(tf.string_split([x]).values[:self.seq_max_len]), tf.int32)
        return id, src

    def _batching_func(self, x):

        if self.schema == DataSchemaEnum.train:
            padded_shapes = (
                tf.TensorShape([self.seq_max_len]),
                tf.TensorShape([self.seq_max_len]),
                tf.TensorShape([]),  # label
            )
            padding_values = (0, 0, 0.0)
        else:
            padded_shapes = padded_shapes_all = (
                tf.TensorShape([]),  # id1
                tf.TensorShape([self.seq_max_len])
            )
            padding_values = ('', 0)

        return x.padded_batch(
            self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)  # tf >= 1.10 use drop_remainder=True
