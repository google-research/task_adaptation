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

"""Tests for food101.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import food101
import tensorflow.compat.v1 as tf


class Food101Test(data_testing_lib.BaseTfdsDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(Food101Test, self).setUp(
        data_wrapper=food101.Food101Data(),
        num_classes=101,
        expected_num_samples=dict(
            train=68175,
            val=7575,
            trainval=75750,
            test=25250,
        ),
        required_tensors_shapes={
            'image': (None, None, 3),
            'label': (),
        })


if __name__ == '__main__':
  tf.test.main()
