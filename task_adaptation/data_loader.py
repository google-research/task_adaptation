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

"""Helper function for loading input data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools

from task_adaptation.data import base
# pylint: disable=unused-import
from task_adaptation.data import caltech
from task_adaptation.data import cifar
from task_adaptation.data import clevr
from task_adaptation.data import diabetic_retinopathy
from task_adaptation.data import dmlab
from task_adaptation.data import dsprites
from task_adaptation.data import dtd
from task_adaptation.data import eurosat
from task_adaptation.data import kitti
from task_adaptation.data import oxford_flowers102
from task_adaptation.data import oxford_iiit_pet
from task_adaptation.data import patch_camelyon
from task_adaptation.data import resisc45
from task_adaptation.data import smallnorb
from task_adaptation.data import sun397
from task_adaptation.data import svhn
# pylint: enable=unused-import

from task_adaptation.registry import Registry

import tensorflow.compat.v1 as tf


def get_dataset_instance(data_params):
  if isinstance(data_params["dataset"], str):
    data_cls = Registry.lookup(data_params["dataset"])
    return data_cls(data_dir=data_params["data_dir"])
  elif isinstance(data_params["dataset"], base.ImageData):
    return data_params["dataset"]
  else:
    raise ValueError("Unknown type for \"dataset\" field: {}".format(
        type(data_params["dataset"])))


def preprocess_fn(data, size=224, input_range=(0.0, 1.0)):
  image = data["image"]
  image = tf.image.resize(image, [size, size])

  image = tf.cast(image, tf.float32) / 255.0
  image = image * (input_range[1] - input_range[0]) + input_range[0]

  data["image"] = image
  return data


def build_data_pipeline(data_params, mode):
  """Builds data input pipeline."""

  if mode not in ["train", "eval"]:
    raise ValueError("The input pipeline supports two modes: `train` or `eval`."
                     "Provided mode is {}".format(mode))

  data = get_dataset_instance(data_params)
  data_fn = functools.partial(
      data.get_tf_data,
      split_name=(data_params["dataset_train_split_name"] if mode == "train"
                  else data_params["dataset_eval_split_name"]),
      preprocess_fn=functools.partial(
          preprocess_fn,
          input_range=data_params["input_range"],
          ),
      for_eval=mode == "eval",
      shuffle_buffer_size=data_params["shuffle_buffer_size"],
      prefetch=data_params["prefetch"],
      train_examples=data_params["train_examples"],
      )

  # Estimator's API requires a named parameter "params".
  def input_fn(params):
    return data_fn(batch_size=params["batch_size"])

  return input_fn
