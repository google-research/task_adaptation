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

"""Implements oxford flowers 102 data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import base
from task_adaptation.registry import Registry
import tensorflow_datasets as tfds


@Registry.register("data.oxford_flowers102", "object")
class OxfordFlowers102Data(base.ImageTfdsData):
  """Provides Oxford 102 categories flowers dataset.

  See corresponding tfds dataset for details.

  URL: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
  """

  def __init__(self, data_dir=None):
    dataset_builder = tfds.builder("oxford_flowers102:2.*.*", data_dir=data_dir)
    dataset_builder.download_and_prepare()

    tfds_splits = {
        "train": "train",
        "val": "validation",
        "trainval": "train+validation",
        "test": "test",
    }

    # Example counts are retrieved from the tensorflow dataset info.
    train_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
    val_count = dataset_builder.info.splits[tfds.Split.VALIDATION].num_examples
    test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples

    # Creates a dict with example counts for each split.
    num_samples_splits = {
        "train": train_count,
        "val": val_count,
        "trainval": train_count + val_count,
        "test": test_count
    }

    super(OxfordFlowers102Data, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=10000,
        # Note: Rename tensors but keep their original types.
        base_preprocess_fn=base.make_get_and_cast_tensors_fn({
            "image": ("image", None),
            "label": ("label", None),
        }),
        num_classes=dataset_builder.info.features["label"]
        .num_classes)
