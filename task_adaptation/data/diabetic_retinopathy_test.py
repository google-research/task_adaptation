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

import mock
from task_adaptation.data import data_testing_lib
from task_adaptation.data import diabetic_retinopathy
import tensorflow.compat.v1 as tf


class RetinopathyTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(RetinopathyTest, self).setUp(
        data_wrapper=diabetic_retinopathy.RetinopathyData(),
        num_classes=5,
        expected_num_samples=dict(
            train=35126,
            val=10906,
            trainval=35126 + 10906,
            test=42670,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        })

  def test_heavy_data_augmentation(self):
    augmentation_parameters = (
        0.0,  # Relative up/down scale: 0.0 -> No scaling.
        3.141592 / 4.0,  # Rotation angle.
        3.141592 / 4.0,  # Rotation angle.
        1.0,  # Horizontal flip? 1.0 -> No flipping.
        1.0,  # Vertical flip?
        0.0,  # Relative x-translation.
        0.0)  # Relative y-translation.
    default_config = self.data_wrapper._config
    try:
      image = tf.random.uniform(
          shape=(32, 32, 3), minval=0, maxval=256, dtype=tf.int32)
      image = tf.cast(image, dtype=tf.uint8)

      # Test that rotation preserves the original background color from the
      # images. For that, we rotate 45 degrees the images and check that the
      # corners have the expected color.

      # Pictures with grey background.
      self.data_wrapper._config = "btgraham-300"
      with self.cached_session() as sess:
        with mock.patch.object(
            self.data_wrapper,
            "_sample_heavy_data_augmentation_parameters",
            return_value=augmentation_parameters):
          output = self.data_wrapper._heavy_data_augmentation_fn(
              {"image": image})
          self.assertEqual(output["image"].dtype, image.dtype)
          output = sess.run(output)
      self.assertAllEqual(output["image"][0, 0], (127, 127, 127))
      self.assertAllEqual(output["image"][0, -1], (127, 127, 127))
      self.assertAllEqual(output["image"][-1, 0], (127, 127, 127))
      self.assertAllEqual(output["image"][-1, -1], (127, 127, 127))

      # Pictures with black background.
      self.data_wrapper._config = "250K"
      with self.cached_session() as sess:
        with mock.patch.object(
            self.data_wrapper,
            "_sample_heavy_data_augmentation_parameters",
            return_value=augmentation_parameters):
          output = self.data_wrapper._heavy_data_augmentation_fn(
              {"image": image})
          self.assertEqual(output["image"].dtype, image.dtype)
          output = sess.run(output)
      self.assertAllClose(output["image"][0, 0], (0, 0, 0))
      self.assertAllClose(output["image"][0, -1], (0, 0, 0))
      self.assertAllClose(output["image"][-1, 0], (0, 0, 0))
      self.assertAllClose(output["image"][-1, -1], (0, 0, 0))
    finally:
      self.data_wrapper._config = default_config


if __name__ == "__main__":
  tf.test.main()
