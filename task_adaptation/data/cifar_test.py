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

"""Tests for cifar.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import cifar
from task_adaptation.data import data_testing_lib
import tensorflow.compat.v1 as tf


class Cifar10Test(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(Cifar10Test, self).setUp(
        data_wrapper=cifar.CifarData(num_classes=10),
        num_classes=10,
        expected_num_samples=dict(
            train=45000,
            val=5000,
            trainval=50000,
            test=10000,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (32, 32, 3),
            "label": (),
        })


class Cifar100Test(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(Cifar100Test, self).setUp(
        data_wrapper=cifar.CifarData(num_classes=100),
        num_classes=100,
        expected_num_samples=dict(
            train=45000,
            val=5000,
            trainval=50000,
            test=10000,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (32, 32, 3),
            "label": (),
        })


class CifarIncorrectTest(tf.test.TestCase):
  """Tests CifarData raises a ValueError for incorrect number of classes."""

  def test_incorrect_classes(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Number of classes must be 10 or 100, got 99"):
      cifar.CifarData(num_classes=99)


if __name__ == "__main__":
  tf.test.main()
