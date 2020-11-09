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

"""Implements CUB data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import base
from task_adaptation.registry import Registry
import tensorflow_datasets as tfds

TRAIN_SPLIT_PERCENT = 90


@Registry.register("data.cub2011", "class")
class CUB2011Data(base.ImageTfdsData):
  """Caltech Birds (CUB) 2011 dataset."""

  def __init__(self, data_dir=None):
    dataset_builder = tfds.builder("caltech_birds2011:0.1.1", data_dir=data_dir)

    tfds_splits = {
        "train": "train[:{}%]".format(TRAIN_SPLIT_PERCENT),
        "val": "train[{}%:]".format(TRAIN_SPLIT_PERCENT),
        "trainval": "train",
        "test": "test"
    }

    # Example counts are retrieved from the tensorflow dataset info.
    trainval_count = dataset_builder.info.splits["train"].num_examples
    train_count = int(round(trainval_count * TRAIN_SPLIT_PERCENT / 100.0))
    val_count = trainval_count - train_count
    test_count = dataset_builder.info.splits["test"].num_examples

    # Creates a dict with example counts for each split.
    num_samples_splits = {
        "train": train_count,
        "val": val_count,
        "trainval": trainval_count,
        "test": test_count
    }

    super(CUB2011Data, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        base_preprocess_fn=base.make_get_and_cast_tensors_fn({
            "image": ("image", None),
            "label": ("label", None),
        }),
        num_classes=dataset_builder.info.features["label"].num_classes,
        image_key="image")
