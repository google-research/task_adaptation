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

"""Tests for smallnorb.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import smallnorb
import tensorflow.compat.v1 as tf


# For each of the 4 different data set configuration we need to run all the
# standardized tests in data_testing_lib.BaseDataTest.
class SmallNORBTestDefault(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(SmallNORBTestDefault, self).setUp(
        data_wrapper=smallnorb.SmallNORBData("label_category"),
        num_classes=5,
        expected_num_samples=dict(
            train=24300,
            val=12150,
            trainval=36450,
            test=12150,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (96, 96, 3),
            "label": (),
        },
        tfds_label_key_map="label_category")


class SmallNORBTestElevation(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(SmallNORBTestElevation, self).setUp(
        data_wrapper=smallnorb.SmallNORBData("label_elevation"),
        num_classes=9,
        expected_num_samples=dict(
            train=24300,
            val=12150,
            trainval=36450,
            test=12150,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (96, 96, 3),
            "label": (),
        },
        tfds_label_key_map="label_elevation")


class SmallNORBTestAzimuth(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(SmallNORBTestAzimuth, self).setUp(
        data_wrapper=smallnorb.SmallNORBData("label_azimuth"),
        num_classes=18,
        expected_num_samples=dict(
            train=24300,
            val=12150,
            trainval=36450,
            test=12150,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (96, 96, 3),
            "label": (),
        },
        tfds_label_key_map="label_azimuth")


class SmallNORBTestLighting(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(SmallNORBTestLighting, self).setUp(
        data_wrapper=smallnorb.SmallNORBData("label_lighting"),
        num_classes=6,
        expected_num_samples=dict(
            train=24300,
            val=12150,
            trainval=36450,
            test=12150,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (96, 96, 3),
            "label": (),
        },
        tfds_label_key_map="label_lighting")


class SmallNORBIncorrectTest(tf.test.TestCase):
  """Tests SmallNORBData raises a ValueError for incorrect attributes."""

  def test_incorrect_classes(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "invalid_attribute is not a valid attribute to predict."):
      smallnorb.SmallNORBData("invalid_attribute")


if __name__ == "__main__":
  tf.test.main()
