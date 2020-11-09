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

"""Main adaptation and evaluation loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import task_adaptation.data_loader as data_loader
import task_adaptation.model as model

import tensorflow.compat.v1 as tf


# TPU-specific constant, see
# https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUConfig for more
# details.
TPU_ITERATION_PER_LOOP = 300

FLAGS = flags.FLAGS


def setup_estimator(
    hub_module,
    hub_module_signature,
    work_dir,
    tpu_name,
    save_checkpoints_steps,
    optimization_params,
    data_params):
  """Produces TPUEstimator object for a given configuration."""

  # Merge all parameters into single dictionary (for tf.estimator API).
  num_classes = data_params["dataset"].get_num_classes()
  params = {k: v for d in [optimization_params, data_params,
                           {"hub_module": hub_module,
                            "hub_module_signature": hub_module_signature,
                            "num_classes": num_classes}]
            for k, v in d.items()}

  # Defines the configutation of an adaptation/evaluation loop.

  if tpu_name is not None:
    cluster = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    config = tf.contrib.tpu.RunConfig(
        model_dir=work_dir,
        cluster=cluster,
        keep_checkpoint_max=None,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=TPU_ITERATION_PER_LOOP))
  else:
    config = tf.estimator.RunConfig(
        model_dir=work_dir,
        keep_checkpoint_max=None,
        save_checkpoints_steps=save_checkpoints_steps)

  if tpu_name is not None:
    batch_size = params.pop("batch_size")
    batch_size_eval = params.pop("batch_size_eval")
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model.model_fn,
        model_dir=work_dir,
        params=params,
        config=config,
        use_tpu=True,
        train_batch_size=batch_size,
        eval_batch_size=batch_size_eval)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=work_dir,
        params=params,
        config=config)

  return estimator


def run_training_loop(hub_module,
                      hub_module_signature,
                      work_dir,
                      tpu_name,
                      save_checkpoints_steps,
                      optimization_params,
                      data_params):
  """Runs training loop."""
  data_params["dataset"] = data_loader.get_dataset_instance(data_params)
  estimator = setup_estimator(hub_module,
                              hub_module_signature,
                              work_dir,
                              tpu_name,
                              save_checkpoints_steps,
                              optimization_params,
                              data_params)
  input_fn = data_loader.build_data_pipeline(data_params, mode="train")

  # TPUs require the max number of steps to be specified explicitly.
  estimator.train(input_fn, max_steps=optimization_params["max_steps"])


def run_evaluation_loop(hub_module,
                        hub_module_signature,
                        work_dir,
                        tpu_name,
                        save_checkpoints_steps,
                        optimization_params,
                        data_params):
  """Runs evaluation loop."""
  data_params["dataset"] = data_loader.get_dataset_instance(data_params)
  estimator = setup_estimator(hub_module,
                              hub_module_signature,
                              work_dir,
                              tpu_name,
                              save_checkpoints_steps,
                              optimization_params,
                              data_params)
  input_fn = data_loader.build_data_pipeline(data_params, mode="eval")

  with tf.gfile.Open(os.path.join(work_dir, "result_file.txt"), "w") as f:
    all_checkpoints = set([".".join(f.split(".")[:-1])
                           for f in tf.gfile.ListDirectory(work_dir)
                           if f.startswith("model.ckpt")])
    # Sort checkpoints by the global step.
    all_checkpoints = sorted(all_checkpoints,
                             key=lambda x: int(x.split("-")[-1]))
    # For efficiency reasons we evluate only the last checkpoint
    for ckpt in all_checkpoints[-1:]:
      ckpt = os.path.join(work_dir, ckpt)
      res = estimator.evaluate(input_fn,
                               steps=(data_params["dataset"].get_num_samples(
                                   data_params["dataset_eval_split_name"]) //
                                      data_params["batch_size_eval"]),
                               checkpoint_path=ckpt)
      f.write("Accuracy at step {}: {}\n".format(res["global_step"],
                                                 res["accuracy"]))
