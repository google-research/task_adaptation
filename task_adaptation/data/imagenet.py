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

"""Implements ImageNet data class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import task_adaptation.data.base as base
from task_adaptation.registry import Registry
import tensorflow as tf
import tensorflow_datasets as tfds

# This constant specifies the percentage of data that is used to create custom
# train/val splits. Specifically, TRAIN_SPLIT_PERCENT% of the official training
# split is used as a new training split and the rest is used for validation.
TRAIN_SPLIT_PERCENT = 96


@Registry.register("data.imagenet", "object")
class ImageNetData(base.ImageTfdsData):
  """Provides ImageNet data."""

  def __init__(self, features=("image", "label")):

    dataset_builder = tfds.builder("imagenet2012:5.*.*")

    # Defines dataset specific train/val/trainval/test splits.
    # Note, that the test split for "imagenet2012" dataset is not available.
    # Thus, we use the val split as test. Moreover, we split the train split
    # into two parts: new train split and new val split.
    tfds_splits = {}
    tfds_splits["train"] = "train[:{}%]".format(TRAIN_SPLIT_PERCENT)
    tfds_splits["val"] = "train[{}%:]".format(TRAIN_SPLIT_PERCENT)
    tfds_splits["trainval"] = "train"
    tfds_splits["test"] = "validation"

    # Creates a dict with example counts.
    num_samples_splits = {}
    trainval_count = dataset_builder.info.splits["train"].num_examples
    test_count = dataset_builder.info.splits["validation"].num_examples
    num_samples_splits["train"] = (TRAIN_SPLIT_PERCENT * trainval_count) // 100
    num_samples_splits["val"] = trainval_count - num_samples_splits["train"]
    num_samples_splits["trainval"] = trainval_count
    num_samples_splits["test"] = test_count

    super(ImageNetData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn(features),
        filter_fn=self._get_filter_fn(),
        num_classes=dataset_builder.info.features["label"].num_classes)

  def _get_filter_fn(self):
    return None


@Registry.register("data.imagenet_with_fnames", "object")
class ImageNetWithFNamesData(ImageNetData):
  """Provides ImageNet data, supports file name based filtering."""

  def __init__(self, filter_filename=None):
    self._filter_filename = filter_filename
    super(ImageNetWithFNamesData, self).__init__(
        features=("image", "label", "file_name"))

  def _get_filter_fn(self):

    def _filter_fn(example):
      """Filtering based on external file name list."""
      if self._filter_filename is not None:
        with tf.gfile.Open(self._filter_filename, "r") as f:
          filename_list = json.load(f)
          filename_list = filename_list["values"]
          filename_list = tf.constant(filename_list)
          filename_list = tf.contrib.lookup.index_table_from_tensor(
              mapping=filename_list,
              num_oov_buckets=0,
              default_value=-1,
              name="filter_filename_lookup_table")

        return tf.math.greater_equal(
            filename_list.lookup(example["file_name"]), 0)
      else:
        return True

    return _filter_fn if self._filter_filename else None
