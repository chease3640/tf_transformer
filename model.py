#!/usr/bin/env python
# coding: utf-8
"""This module implements the transformer auto-encoder model."""
import tensorflow as tf
from transformer.transformer import Transformer
from transformer.model_utils import contrastive_loss, compute_accuracy


class AutoEncoder(object):
    """AutoEncoder for text."""

    def __init__(self, iterator, params, mode):
        """Initialize model, build graph.
        Args:
          params: parameters.
          mode: train | eval | predict mode defined with tf.estimator.ModeKeys.
        """
        # Build graph.
        tf.logging.info("Initializing model, building graph...")
        # Predict single product embedding.
        if mode == tf.estimator.ModeKeys.PREDICT:
            self.encode = Transformer(params, False)(iterator.src)

        else:
            logits = Transformer(params, True)(iterator.src, iterator.tgt)
            with tf.name_scope("loss"):
                self.loss = contrastive_loss(iterator.label, logits)
            with tf.name_scope("accuracy"):
                self.accuracy = compute_accuracy(iterator.label, logits)

        self.model_stats()

    @staticmethod
    def model_stats():
        """Model size statistics info."""
        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))
