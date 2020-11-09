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

"""Tests for patch_camelyon.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import patch_camelyon
import tensorflow.compat.v1 as tf


class PatchCamelyonTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(PatchCamelyonTest, self).setUp(
        data_wrapper=patch_camelyon.PatchCamelyonData(),
        num_classes=2,
        expected_num_samples=dict(
            train=262144,
            val=32768,
            trainval=262144 + 32768,
            test=32768,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (96, 96, 3),
            "label": (),
        })


if __name__ == "__main__":
  tf.test.main()
