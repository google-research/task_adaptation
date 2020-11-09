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

"""Tests for dsprites.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import dsprites
import tensorflow.compat.v1 as tf


# For each of the 5 different data set configuration we need to run all the
# standardized tests in data_testing_lib.BaseDataTest.
class DSpritesTestDefault(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(DSpritesTestDefault, self).setUp(
        data_wrapper=dsprites.DSpritesData("label_shape"),
        num_classes=3,
        expected_num_samples=dict(
            train=589824,
            val=73728,
            trainval=663552,
            test=73728,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (64, 64, 3),
            "label": (),
        },
        tfds_label_key_map="label_shape")


class DSpritesTestScale(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(DSpritesTestScale, self).setUp(
        data_wrapper=dsprites.DSpritesData("label_scale"),
        num_classes=6,
        expected_num_samples=dict(
            train=589824,
            val=73728,
            trainval=663552,
            test=73728,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (64, 64, 3),
            "label": (),
        },
        tfds_label_key_map="label_scale")


class DSpritesTestOrientation(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(DSpritesTestOrientation, self).setUp(
        data_wrapper=dsprites.DSpritesData("label_orientation"),
        num_classes=40,
        expected_num_samples=dict(
            train=589824,
            val=73728,
            trainval=663552,
            test=73728,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (64, 64, 3),
            "label": (),
        },
        tfds_label_key_map="label_orientation")


class DSpritesTestXPosition(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(DSpritesTestXPosition, self).setUp(
        data_wrapper=dsprites.DSpritesData("label_x_position"),
        num_classes=32,
        expected_num_samples=dict(
            train=589824,
            val=73728,
            trainval=663552,
            test=73728,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (64, 64, 3),
            "label": (),
        },
        tfds_label_key_map="label_x_position")


class DSpritesTestXPositionGrouped(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(DSpritesTestXPositionGrouped, self).setUp(
        data_wrapper=dsprites.DSpritesData("label_x_position", 15),
        num_classes=15,
        expected_num_samples=dict(
            train=589824,
            val=73728,
            trainval=663552,
            test=73728,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (64, 64, 3),
            "label": (),
        },
        tfds_label_key_map={})


class DSpritesTestYPosition(data_testing_lib.BaseVTABDataTest):
  """See base classes for usage and test descriptions."""

  def setUp(self):
    super(DSpritesTestYPosition, self).setUp(
        data_wrapper=dsprites.DSpritesData("label_y_position"),
        num_classes=32,
        expected_num_samples=dict(
            train=589824,
            val=73728,
            trainval=663552,
            test=73728,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (64, 64, 3),
            "label": (),
        },
        tfds_label_key_map="label_y_position")


class DSpritesIncorrectTest(tf.test.TestCase):
  """Tests DSpritesData raises a ValueError for incorrect attributes."""

  def test_incorrect_classes(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "invalid_attribute is not a valid attribute to predict."):
      dsprites.DSpritesData("invalid_attribute")

if __name__ == "__main__":
  tf.test.main()
