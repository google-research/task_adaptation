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

"""Implements the Describable Textures Dataset (DTD) data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import task_adaptation.data.base as base
from task_adaptation.registry import Registry
import tensorflow_datasets as tfds


@Registry.register("data.dtd", "object")
class DTDData(base.ImageTfdsData):
  """Provides Describable Textures Dataset (DTD) data.

  As of version 1.0.0, the train/val/test splits correspond to those of the
  1st fold of the official cross-validation partition.

  For additional details and usage, see the base class.
  """

  def __init__(self, data_dir=None):

    dataset_builder = tfds.builder("dtd:3.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    # Defines dataset specific train/val/trainval/test splits.
    tfds_splits = {}
    tfds_splits["train"] = "train"
    tfds_splits["val"] = "validation"
    tfds_splits["trainval"] = "train+validation"
    tfds_splits["test"] = "test"

    # Creates a dict with example counts for each split.
    num_samples_splits = {}
    train_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
    val_count = dataset_builder.info.splits[tfds.Split.VALIDATION].num_examples
    test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples
    num_samples_splits["train"] = train_count
    num_samples_splits["val"] = val_count
    num_samples_splits["trainval"] = train_count + val_count
    num_samples_splits["test"] = test_count

    super(DTDData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Export only image and label tensors with their original types.
        base_preprocess_fn=base.make_get_tensors_fn(["image", "label"]),
        num_classes=dataset_builder.info.features["label"].num_classes)
