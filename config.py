#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import os

from utils import load_vocab

# Source params
tf.flags.DEFINE_enum("source", "odps", ["odps", "local"], "File source, `odps` or `local`.")

# File params
tf.flags.DEFINE_string("tables", None, "Tables file")  # 输入表
tf.flags.DEFINE_string("outputs", None, "Outputs file.")  # 输出表
tf.flags.DEFINE_string("buckets", None, "Buckets info")  #  
tf.flags.DEFINE_string("checkpointDir", None, "Checkpoint dir.")  # 

tf.flags.DEFINE_string("vocab_file", None, "Vocab file.")
tf.flags.DEFINE_string("embed_file", None, "Embed file.")

# Data params
tf.flags.DEFINE_integer("seq_max_len", 64, "Max sequence length [15].")
tf.flags.DEFINE_integer("num_samples", 100000, "Num of samples [60000000].")

# Model params
# tf.flags.DEFINE_bool("add_dense_layer", False, "Add dense layer after embedding or not.")
# tf.flags.DEFINE_integer("output_dim", 128, "Dense layer output units. [128].")
# tf.flags.DEFINE_enum("energy_func", "cosine", ['euclidean', 'cosine', 'exp_manhattan'], "Energy function.")
# tf.flags.DEFINE_float("margin", 1.0, "Contrastive loss margin.")
# common
# tf.flags.DEFINE_float("dropout", 0.8, "Dropout keep prob [0.8].")
# attention
tf.flags.DEFINE_integer("hidden_size", 128, "Input embedding dim (attention hidden size) [128].")
tf.flags.DEFINE_integer("num_hidden_layers", 3, "Number of layers in the encoder and decoder stacks [3].")
tf.flags.DEFINE_integer("num_heads", 8, "Number of heads to use in multi-headed attention [8].")
tf.flags.DEFINE_integer("filter_size", 1024, "Inner layer dimension in the feedforward network [2048].")
tf.flags.DEFINE_integer("attention_size", 256, "fully attention hidden size [256].")
tf.flags.DEFINE_float("layer_postprocess_dropout", 0.1, "Big model [0.3].")
tf.flags.DEFINE_float("attention_dropout", 0.1, "Attention dropout [0.1].")
tf.flags.DEFINE_float("relu_dropout", 0.1, "Relu dropout [0.1].")
tf.flags.DEFINE_float("label_smoothing", 0.1, "Label smoothing [0.1].")
tf.flags.DEFINE_boolean("allow_ffn_pad", True, "Allow ffn padding [True].")

# Training params
tf.flags.DEFINE_boolean("train_distributed", False, "Distributed training.")
tf.flags.DEFINE_integer("batch_size", 512, "Train batch size [1000].")
tf.flags.DEFINE_integer("num_epochs", 5, "Max epoch [5].")
tf.flags.DEFINE_integer("max_steps", None, "Max steps for training.")
tf.flags.DEFINE_float("lr", 0.002, "Initial learning rate [0.002].")
tf.flags.DEFINE_boolean("use_lr_decay", True, "Use learning rate decay or not [True].")
tf.flags.DEFINE_integer("decay_steps", 100000, "Use learning decay or not [True].")
tf.flags.DEFINE_float("decay_rate", 0.96, "The decay rate.")
tf.flags.DEFINE_enum("optimizer", "adam", ["adam", "rmsprop", "sgd", "momentum", "adagrad"], "Optimizer name.")
tf.flags.DEFINE_integer("show_loss_per_steps", 1000, "Show train info steps [1000].")
tf.flags.DEFINE_boolean("resume", False, "Keep training or not [False].")
tf.flags.DEFINE_boolean("save_model", True, "Save model (SavedModel) or not [True].")
tf.flags.DEFINE_integer("eval_per_steps", 50000, "Every steps to eval model [50000].")
tf.flags.DEFINE_integer("save_per_steps", 10000, "Every steps to save model [10000].")

tf.flags.DEFINE_boolean("use_grad_clip", False, "Clip grads or not [False].")
tf.flags.DEFINE_integer("grad_clip_norm", 5, "Max grad norm if use grad clip [5].")
tf.flags.DEFINE_integer("num_keep_ckpts", 5, "Max num of ckpts [5].")
tf.flags.DEFINE_integer("random_seed", 12345, "Random seed for model [123].")

# Predict params
tf.flags.DEFINE_boolean("predict_distributed", False, "Distributed predicting.")
tf.flags.DEFINE_integer("predict_batch_size", 1000, "Predict batch size [1000].")

# distributed params
tf.flags.DEFINE_boolean("is_sync", False, "Sync distributed if set True.")
tf.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs.")
tf.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs.")
tf.flags.DEFINE_string("job_name", "", "One of `ps`, `worker`.")
tf.flags.DEFINE_integer("task_index", 0, "Index of task within the job.")
tf.flags.DEFINE_integer("save_checkpoint_secs", 1800, "Save checkpoint per secs [1800].")

# Misc params
# 1 for INFO, 2 for WARNING, 3 for ERROR. Defaults to 0
tf.flags.DEFINE_integer("tf_cpp_min_log_level", 3, "Control core cpp log message output [0].")
tf.flags.DEFINE_enum("verbosity", "INFO", ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"], "Control python log message.")

# Auto params, do not need to set
tf.flags.DEFINE_string("train_file", None, "Train file.")
tf.flags.DEFINE_string("predict_file", None, "Inference file.")
tf.flags.DEFINE_string("vocab_size", None, "Vocab size.")

FLAGS = tf.flags.FLAGS
FLAGS.train_file = FLAGS.tables.split(",")[0]
FLAGS.predict_file = FLAGS.tables.split(",")[0]

# read vocab from ODPS:
FLAGS.vocab_file = FLAGS.tables.split(",")[1]
vocab, vocab_size = load_vocab(FLAGS.source, FLAGS.vocab_file)

# read vocab from file:
FLAGS.vocab_file = os.path.join(FLAGS.buckets, 'vocab')
# vocab, vocab_size = load_vocab("local", FLAGS.vocab_file)

FLAGS.vocab_size = vocab_size
