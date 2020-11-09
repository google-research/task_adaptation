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

"""Tests for clevr.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import clevr
from task_adaptation.data import data_testing_lib
import tensorflow.compat.v1 as tf


class CLEVRDataCountAllTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(CLEVRDataCountAllTest, self).setUp(
        data_wrapper=clevr.CLEVRData(task="count_all"),
        num_classes=8,
        expected_num_samples=dict(
            train=63000,
            val=7000,
            trainval=70000,
            test=15000,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class CLEVRDataCountCylindersTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(CLEVRDataCountCylindersTest, self).setUp(
        data_wrapper=clevr.CLEVRData(task="count_cylinders"),
        num_classes=11,
        expected_num_samples=dict(
            train=63000,
            val=7000,
            trainval=70000,
            test=15000,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class CLEVRDataClosestTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(CLEVRDataClosestTest, self).setUp(
        data_wrapper=clevr.CLEVRData(task="closest_object_distance"),
        num_classes=6,
        expected_num_samples=dict(
            train=63000,
            val=7000,
            trainval=70000,
            test=15000,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


if __name__ == "__main__":
  tf.test.main()
