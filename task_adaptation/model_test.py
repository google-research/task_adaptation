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

"""Tests for the model module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest

import numpy as np
import task_adaptation.model as model
import task_adaptation.test_utils as test_utils

import tensorflow.compat.v1 as tf

NUM_CLASSES = 1000


class ModelTest(tf.test.TestCase):

  def call_model_fn(self, hub_module):
    params = {
        k: v for d in [
            test_utils.get_optimization_params(),
            test_utils.get_data_params(),
            {
                "hub_module": hub_module,
                "hub_module_signature": None,
                "num_classes": NUM_CLASSES
            }
        ] for k, v in d.items()
    }

    for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
      tf.reset_default_graph()
      images = tf.constant(
          np.random.random([32, 224, 224, 3]), dtype=tf.float32)
      labels = tf.constant(np.random.randint(0, 1000, [32]), dtype=tf.int64)
      model.model_fn({"image": images, "label": labels}, mode, params)

  def test_hub_module_fn(self):
    path = os.path.join(self.get_temp_dir(), "module")
    test_utils.create_dummy_hub_model(path, NUM_CLASSES)

    self.call_model_fn(path)

  def test_saved_model_fn(self):
    path = os.path.join(self.get_temp_dir(), "model")
    test_utils.create_dummy_saved_model(path)

    self.call_model_fn(path)


if __name__ == "__main__":
  absltest.main()
