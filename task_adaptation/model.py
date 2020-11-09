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

"""Model that runs a given hub-module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import task_adaptation.trainer as trainer

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


def model_fn(features, mode, params):
  """A function for applying hub module that follows Estimator API."""

  hub_module = params["hub_module"]
  finetune_layer = params["finetune_layer"]
  num_classes = params["num_classes"]
  initial_learning_rate = params["initial_learning_rate"]
  momentum = params["momentum"]
  lr_decay_factor = params["lr_decay_factor"]
  decay_steps = params["decay_steps"]
  warmup_steps = params["warmup_steps"]

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  module_path = hub.resolve(hub_module)
  is_legacy_hub_module = tf.io.gfile.exists(
      os.path.join(module_path, "tfhub_module.pb"))
  if is_legacy_hub_module:
    module = hub.Module(hub_module,
                        trainable=is_training,
                        tags={"train"} if is_training else None)
    pre_logits = module(features["image"],
                        signature=params["hub_module_signature"],
                        as_dict=True)[finetune_layer]
  else:
    module = hub.load(hub_module)
    tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES).extend(
        module.trainable_variables)
    pre_logits = module(features["image"], training=is_training)

  num_dim_pre_logits = len(pre_logits.get_shape().as_list())
  if num_dim_pre_logits == 4:
    pre_logits = tf.squeeze(pre_logits, [1, 2])
  elif num_dim_pre_logits != 2:
    raise ValueError("Invalid number of dimensions in the representation "
                     "layer: {}, but only 2 or 4 are allowed".format(
                         num_dim_pre_logits))

  logits = tf.layers.dense(pre_logits,
                           units=num_classes,
                           kernel_initializer=tf.zeros_initializer(),
                           name="linear_head")

  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=features["label"])
  loss = tf.reduce_mean(loss)

  def accuracy_metric(logits, labels):
    return {"accuracy": tf.metrics.accuracy(
        labels=labels,
        predictions=tf.argmax(logits, axis=-1))}
  eval_metrics = (accuracy_metric, [logits, features["label"]])

  if mode == tf.estimator.ModeKeys.EVAL:
    if params["tpu_name"] is not None:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, eval_metrics=eval_metrics)
    else:
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss,
          eval_metric_ops=eval_metrics[0](*eval_metrics[1]))
  elif mode == tf.estimator.ModeKeys.TRAIN:
    train_op = trainer.get_train_op(loss,
                                    initial_learning_rate,
                                    momentum,
                                    lr_decay_factor,
                                    decay_steps,
                                    warmup_steps,
                                    use_tpu=params["tpu_name"] is not None)
    spec_type = (tf.contrib.tpu.TPUEstimatorSpec
                 if params["tpu_name"] is not None
                 else tf.estimator.EstimatorSpec)
    return spec_type(mode=mode, loss=loss, train_op=train_op)
  else:
    raise ValueError("Unsupported mode %s" % mode)
