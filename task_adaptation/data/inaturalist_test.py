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

# Lint as: python3
"""Tests for inaturalist.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from task_adaptation.data import data_testing_lib
from task_adaptation.data import inaturalist
import tensorflow.compat.v1 as tf


class INaturalistTest(data_testing_lib.BaseTfdsDataTest):

  def setUp(self):
    super(INaturalistTest, self).setUp(
        data_wrapper=inaturalist.INaturalistData(),
        num_classes=5089,
        expected_num_samples=dict(
            train=521266,
            val=57918,
            trainval=579184,
            test=95986,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        })


if __name__ == "__main__":
  tf.test.main()
