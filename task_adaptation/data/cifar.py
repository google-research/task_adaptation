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

"""Implements Cifar data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import task_adaptation.data.base as base
from task_adaptation.registry import Registry
import tensorflow_datasets as tfds

# This constant specifies the percentage of data that is used to create custom
# train/val splits. Specifically, TRAIN_SPLIT_PERCENT% of the official training
# split is used as a new training split and the rest is used for validation.
TRAIN_SPLIT_PERCENT = 90


@Registry.register("data.cifar", "object")
class CifarData(base.ImageTfdsData):
  """Provides Cifar10 or Cifar100 data.

  Cifar comes only with a training and test set. Therefore, the validation set
  is split out of the original training set, and the remaining examples are used
  as the "train" split. The "trainval" split corresponds to the original
  training set.

  For additional details and usage, see the base class.
  """

  def __init__(self, num_classes=10, data_dir=None):

    if num_classes == 10:
      dataset_builder = tfds.builder("cifar10:3.*.*", data_dir=data_dir)
    elif num_classes == 100:
      dataset_builder = tfds.builder("cifar100:3.*.*", data_dir=data_dir)
    else:
      raise ValueError(
          "Number of classes must be 10 or 100, got {}".format(num_classes))

    dataset_builder.download_and_prepare()

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {}
    tfds_splits["train"] = "train[:{}%]".format(TRAIN_SPLIT_PERCENT)
    tfds_splits["val"] = "train[{}%:]".format(TRAIN_SPLIT_PERCENT)
    tfds_splits["trainval"] = "train"
    tfds_splits["test"] = "test"

    # Creates a dict with example counts for each split.
    num_samples_splits = {}
    trainval_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
    test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples
    num_samples_splits["train"] = (TRAIN_SPLIT_PERCENT * trainval_count) // 100
    num_samples_splits["val"] = trainval_count - num_samples_splits["train"]
    num_samples_splits["trainval"] = trainval_count
    num_samples_splits["test"] = test_count

    super(CifarData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn(["image", "label"]),
        num_classes=dataset_builder.info.features["label"].num_classes)
