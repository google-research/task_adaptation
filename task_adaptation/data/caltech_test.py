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

"""Tests for caltech.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from task_adaptation.data import caltech
from task_adaptation.data import data_testing_lib
import tensorflow.compat.v1 as tf


class Caltech101Test(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    self.dataset = caltech.Caltech101()
    super(Caltech101Test, self).setUp(
        data_wrapper=self.dataset,
        num_classes=102,  # N.b. Caltech101 has 102 classes (1 for background).
        expected_num_samples=dict(
            train=2754,
            val=306,
            trainval=2754 + 306,  # 3060 (30 images / class).
            test=6084,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        })

  def test_all_classes_in_train(self):
    """Tests that the train set has at least one element in every class."""
    # Runs over the small validation set, rather than the full train set.
    # For each class, there should be fewer than 30 items for there to be at
    # least one in the training set.
    ds = self.dataset.get_tf_data("val", batch_size=1, epochs=1)
    ds.repeat(1)
    next_element = tf.data.make_one_shot_iterator(ds).get_next()
    class_count = collections.defaultdict(int)
    with tf.Session() as sess:
      while True:
        try:
          value = sess.run(next_element)
          class_count[value["label"][0]] += 1
        except tf.errors.OutOfRangeError:
          break

    self.assertGreater(30, max(class_count.values()))


if __name__ == "__main__":
  tf.test.main()
