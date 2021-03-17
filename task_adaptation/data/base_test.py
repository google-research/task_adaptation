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

"""Tests for task_adaptation.data.base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from task_adaptation.data import base
from task_adaptation.data import data_testing_lib
import tensorflow.compat.v1 as tf


class FakeImageData(base.ImageData):
  EXAMPLE_NUM = 5

  def __init__(self):
    super(FakeImageData, self).__init__(
        num_samples_splits={'fake': FakeImageData.EXAMPLE_NUM},
        shuffle_buffer_size=1,  # no shuffle
        num_preprocessing_threads=1,
        num_classes=1)

  def _get_dataset_split(self, split_name, shuffle_files=False):

    def fake_data(index):
      return {'image': index, 'label': index}

    return tf.data.Dataset.range(FakeImageData.EXAMPLE_NUM).map(fake_data)


class BaseTest(tf.test.TestCase):

  def test_make_get_tensors_fn(self):
    input_dict = {'tens1': 1, 'tens2': 2, 'tens3': 3}
    # Normal case.
    fn = base.make_get_tensors_fn(output_tensors=['tens1', 'tens2'])
    self.assertTrue(callable(fn))
    self.assertEqual(fn(input_dict), {'tens1': 1, 'tens2': 2})

    # One output tensor is not specified in the input dict.
    fn = base.make_get_tensors_fn(output_tensors=['tens1', 'tens2', 'tens4'])
    self.assertTrue(callable(fn))
    with self.assertRaises(KeyError):
      fn(input_dict)

    # Empty output.
    fn = base.make_get_tensors_fn(output_tensors=())
    self.assertTrue(callable(fn))
    self.assertEqual(fn(input_dict), {})

  def test_make_get_and_cast_tensors_fn(self):
    input_dict = {
        't1': tf.constant(value=0, dtype=tf.int32),
        't2': tf.constant(value=-1.0, dtype=tf.float32),
        't3': tf.constant(value=1.0, dtype=tf.float64),
    }
    # Equivalent to get_tensors_fn.
    fn = base.make_get_and_cast_tensors_fn(output_tensors={
        't1': None,
        't2': None,
    })
    self.assertTrue(callable(fn))
    self.assertEqual(
        fn(input_dict), {
            't1': input_dict['t1'],
            't2': input_dict['t2']
        })

    # Cast to different type.
    fn = base.make_get_and_cast_tensors_fn(output_tensors={
        't1': tf.float64,
        't2': tf.float64,
    })
    self.assertTrue(callable(fn))
    output_dict = fn(input_dict)
    self.assertSetEqual(set(output_dict.keys()), {'t1', 't2'})
    self.assertEqual(output_dict['t1'].dtype, tf.float64)
    self.assertEqual(output_dict['t2'].dtype, tf.float64)

    # General case.
    fn = base.make_get_and_cast_tensors_fn(output_tensors={
        't1': ('t1_new_name', tf.float64),
        't2': tf.float64,
        't3': None,
    })
    self.assertTrue(callable(fn))
    output_dict = fn(input_dict)
    self.assertSetEqual(set(output_dict.keys()), {'t1_new_name', 't2', 't3'})
    self.assertEqual(output_dict['t1_new_name'].dtype, tf.float64)
    self.assertEqual(output_dict['t2'].dtype, tf.float64)
    self.assertEqual(output_dict['t3'].dtype, input_dict['t3'].dtype)

    # Output key does not exist.
    fn = base.make_get_and_cast_tensors_fn(output_tensors={
        't1': None,
        't25': ('t2', tf.float32),
    })
    self.assertTrue(callable(fn))
    with self.assertRaises(KeyError):
      fn(input_dict)

  def test_compose_preprocess_fn(self):
    # If no functions are given to compose, returns identity function.
    fn = base.compose_preprocess_fn()
    self.assertTrue(callable(fn))
    self.assertEqual(fn(25), 25)

    # If a single function is given, compose simply returns it.
    fn = base.compose_preprocess_fn(lambda x: 2 * x)
    self.assertTrue(callable(fn))
    self.assertEqual(fn(25), 50)

    # None should be ignored, so this is like the identity function.
    fn = base.compose_preprocess_fn(None)
    self.assertTrue(callable(fn))
    self.assertEqual(fn(25), 25)

    # None should be ignored, result_fn(x) = 2 * x + 1
    fn = base.compose_preprocess_fn(lambda x: 2 * x, None, lambda x: x + 1)
    self.assertTrue(callable(fn))
    self.assertEqual(fn(25), 51)

  def test_pairwise_mix(self):

    def fake_pairwise_mix(data1, data2):
      return {
          'image': data1['image'] + data2['image'],
          'label': tf.math.maximum(data1['label'], data2['label']),
      }

    data = FakeImageData()
    dataset = data.get_tf_data(
        split_name='fake',
        batch_size=1,
        pairwise_mix_fn=fake_pairwise_mix,
        epochs=1)

    with self.session() as sess:
      actual = list(
          data_testing_lib.BaseDataTest.iterate_dataset(dataset, sess))
      self.assertAllEqual([
          {
              'image': np.asarray([1]),
              'label': np.asarray([1])
          },
          {
              'image': np.asarray([3]),
              'label': np.asarray([2])
          },
          {
              'image': np.asarray([5]),
              'label': np.asarray([3])
          },
          {
              'image': np.asarray([7]),
              'label': np.asarray([4])
          },
      ], actual)


if __name__ == '__main__':
  tf.test.main()
