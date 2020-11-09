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

"""Tests for data_loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from task_adaptation import data_loader
from task_adaptation import test_utils

import tensorflow.compat.v1 as tf


class DataLoaderTest(absltest.TestCase):

  def test_build_data_pipeline(self):
    input_fn = data_loader.build_data_pipeline(test_utils.get_data_params(),
                                               mode="eval")
    data = input_fn({"batch_size": 32}).make_one_shot_iterator().get_next()
    self.assertIsInstance(data["image"], tf.Tensor)
    self.assertIsInstance(data["label"], tf.Tensor)


if __name__ == "__main__":
  absltest.main()
