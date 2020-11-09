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
"""Food101 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import task_adaptation.data.base as base
from task_adaptation.registry import Registry
import tensorflow_datasets as tfds

TRAIN_SPLIT_PERCENT = 90


@Registry.register("data.food101", "class")
class Food101Data(base.ImageTfdsData):
  """Food101 dataset from TFDS."""

  def __init__(self, data_dir=None, train_split_percent=None):
    train_split_percent = train_split_percent or TRAIN_SPLIT_PERCENT
    dataset_builder = tfds.builder("food101:2.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()
    tfds_splits = {
        "train": "train[:{}%]".format(train_split_percent),
        "val": "train[{}:]".format(train_split_percent),
        "trainval": "train",
        "test": "validation",
    }
    # Creates a dict with example counts for each split.
    num_train_examples_full = dataset_builder.info.splits["train"].num_examples
    num_train_examples = (
        (num_train_examples_full * train_split_percent) // 100)
    num_valid_examples = num_train_examples_full - num_train_examples
    num_samples_splits = {
        "train": num_train_examples,
        "val": num_valid_examples,
        "trainval": dataset_builder.info.splits["train"].num_examples,
        "test": dataset_builder.info.splits["validation"].num_examples,
    }

    super(Food101Data, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=100,
        shuffle_buffer_size=10000,
        image_key="image",
        num_classes=dataset_builder.info.features["label"].num_classes)
