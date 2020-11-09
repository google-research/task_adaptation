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

"""Tests for dtd.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import dtd
import tensorflow.compat.v1 as tf


class DTDDataTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(DTDDataTest, self).setUp(
        data_wrapper=dtd.DTDData(),
        num_classes=47,
        expected_num_samples=dict(
            train=1880,
            val=1880,
            trainval=1880 + 1880,
            test=1880,
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
