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

"""Tests for loop.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl.testing import absltest
from task_adaptation import loop
from task_adaptation import test_utils
import tensorflow.compat.v1 as tf


class LoopTest(tf.test.TestCase):

  def test_run_training_loop(self):
    module_path = os.path.join(self.get_temp_dir(), "module")
    test_utils.create_dummy_hub_model(module_path, num_outputs=10)
    tmp_dir = tempfile.mkdtemp()
    loop.run_training_loop(
        hub_module=module_path,
        hub_module_signature=None,
        work_dir=tmp_dir,
        tpu_name=None,
        save_checkpoints_steps=10,
        optimization_params=test_utils.get_optimization_params(),
        data_params=test_utils.get_data_params())

    self.assertNotEmpty([f for f in os.listdir(tmp_dir)
                         if f.startswith("model.ckpt")])


if __name__ == "__main__":
  absltest.main()
