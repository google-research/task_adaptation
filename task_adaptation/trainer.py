# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimization utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def apply_warmup_lr(global_step, lr, base_lr, warmup_steps):
  """Modifies learning rate, such that it linearly grows from 0 to base_lr."""

  if warmup_steps > 0:
    warmup_lr = (tf.cast(global_step, tf.float32) * (base_lr / warmup_steps))

    lr = tf.cond(tf.less(tf.cast(global_step, tf.float32), warmup_steps),
                 lambda: warmup_lr,
                 lambda: lr)

  return lr


def get_train_op(loss,
                 initial_learning_rate,
                 momentum,
                 lr_decay_factor,
                 decay_steps,
                 warmup_steps,
                 use_tpu=False):
  """Builds an SGD update operation."""

  global_step = tf.train.get_or_create_global_step()

  lr = tf.train.piecewise_constant(
      global_step,
      decay_steps,
      [initial_learning_rate * (lr_decay_factor ** i)
       for i in range(len(decay_steps) + 1)])
  lr = apply_warmup_lr(global_step, lr, initial_learning_rate, warmup_steps)

  optimizer = tf.train.MomentumOptimizer(learning_rate=lr,
                                         momentum=momentum)
  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  train_op = optimizer.minimize(loss)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  global_step_inc_op = tf.assign_add(global_step, 1)

  return tf.group([train_op, update_ops, global_step_inc_op])
