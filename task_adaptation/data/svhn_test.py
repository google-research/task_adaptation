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

"""Tests for svhn.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import svhn
import tensorflow as tf


class SvhnTest(data_testing_lib.BaseTfdsDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    # The test scenarios have been defined in the base class
    # data_testing_lib.BaseDataTest already, which tests the information
    # provided in the setup function:
    # classses, num, dataset_output, tfds_splits keys
    super(SvhnTest, self).setUp(
        data_wrapper=svhn.SvhnData(),
        num_classes=10,
        expected_num_samples=dict(
            train=65931,
            val=7326,
            trainval=73257,
            test=26032,
        ),
        required_tensors_shapes={
            "image": (32, 32, 3),
            "label": (),
        })


if __name__ == "__main__":
  tf.test.main()
