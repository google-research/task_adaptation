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

"""Tests for sun397.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import sun397
import tensorflow.compat.v1 as tf


class Sun397Test(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(Sun397Test, self).setUp(
        data_wrapper=sun397.Sun397Data(config="tfds"),
        num_classes=397,
        expected_num_samples=dict(
            train=76128,
            val=10875,
            trainval=10875 + 76128,
            test=21750,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        })

if __name__ == "__main__":
  tf.test.main()
