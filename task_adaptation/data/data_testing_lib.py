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

"""Library for testing the dataset wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import os
import six
from task_adaptation.data import base
import tensorflow.compat.v1 as tf

# pylint: disable=g-bad-import-order


@six.add_metaclass(abc.ABCMeta)
class BaseDataTest(tf.test.TestCase):
  """Base class for testing subclasses of base.ImageData.

  To use this testing library, subclass BaseDataTest and override setUp().
  Pass into BaseDataTest's setUp method the expected statistics for the
  specific dataset being tested. These statistics are stored as instance
  attributes to be used in the tests.

  Attributes:
    data_wrapper: Subclass of base.ImageData for testing.
    default_label_key: str, key of the default output label tensor.
    expected_num_classes: Dict with the expected number of classes for each
      output label tensor.
    expected_num_samples: Dict containing expected number of examples in the
      "train", "val", "trainval", and "test" splits of the dataset.
    required_tensors_shapes: Dictionary with the names of the tensors that
      the dataset should output and their shapes. The dataset could output more
      tensors, but only these are required.
  """

  @abc.abstractmethod
  def setUp(self, data_wrapper, num_classes, expected_num_samples,
            required_tensors_shapes, default_label_key="label"):
    super(BaseDataTest, self).setUp()
    self.data_wrapper = data_wrapper
    # Expected dataset statistics.
    self.expected_num_samples = expected_num_samples
    self.required_tensors_shapes = required_tensors_shapes
    self.default_label_key = default_label_key
    if isinstance(num_classes, int):
      self.expected_num_classes = {default_label_key: num_classes}
    elif isinstance(num_classes, dict):
      self.expected_num_classes = num_classes
    else:
      raise ValueError("`num_classes` must be either int or dict")

  @property
  def expected_splits(self):
    return ("train", "val", "trainval", "test")

  def test_base_class(self):
    """Tests that the dataset wrapper inherits from base.ImageData."""
    self.assertIsInstance(self.data_wrapper, base.ImageData,
                          "Dataset class must inherit from `base.ImageData`.")

  def test_split_dict_keys(self):
    """Tests that the "num_samples" splits contain the correct keys."""
    expected_keys = set(self.expected_num_samples.keys())
    actual_keys = set(self.data_wrapper._num_samples_splits.keys())  # pylint: disable=protected-access
    self.assertSetEqual(expected_keys, actual_keys)

  def test_num_samples(self):
    """Tests that the number of samples for each split is correct."""
    for split, expected in self.expected_num_samples.items():
      self.assertEqual(
          expected, self.data_wrapper.get_num_samples(split),
          msg="Number of examples does not match for split \"%s\"" % split)

  def test_dataset_output(self):
    """Tests that the final tf.Dataset object has expected output shapes."""
    batch_size = 2
    for split in self.expected_splits:
      tf_data = self.data_wrapper.get_tf_data(split, batch_size)
      tf_data_output_shapes = tf.data.get_output_shapes(tf_data)
      self.assertIsInstance(tf_data_output_shapes, dict)
      for tensor_name, expected_shape in self.required_tensors_shapes.items():
        self.assertIn(tensor_name, tf_data_output_shapes.keys())
        expected_shape = [batch_size] + list(expected_shape)
        actual_shape = tf_data_output_shapes[tensor_name].as_list()
        self.assertEqual(
            actual_shape,
            expected_shape,
            msg=("Tensor {!r} for split {!r} does not match the expected "
                 "value".format(tensor_name, split)))

  def test_label_keys(self):
    self.assertEqual(
        self.default_label_key, self.data_wrapper.default_label_key)
    self.assertIn(self.default_label_key, self.data_wrapper.label_keys)
    # pylint: disable=protected-access
    self.assertDictEqual(
        self.expected_num_classes, self.data_wrapper._num_classes)
    # pylint: enable=protected-access

  def test_get_num_classes(self):
    """Tests the expected number of classes."""
    # Check get_num_classes default output.
    self.assertEqual(
        self.expected_num_classes[self.data_wrapper.default_label_key],
        self.data_wrapper.get_num_classes())

    # Check get_num_classes output with particular label keys.
    for label_key, num_classes in self.expected_num_classes.items():
      self.assertEqual(
          num_classes, self.data_wrapper.get_num_classes(label_key),
          msg="Number of classes does not match for label \"%s\"" % label_key)

  @classmethod
  def iterate_dataset(cls, dataset, session):
    dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
    get_next = dataset_iter.get_next()
    try:
      session.run(dataset_iter.initializer)
      while True:
        yield session.run(get_next)
    except tf.errors.OutOfRangeError:
      pass


class BaseTfdsDataTest(BaseDataTest):
  """Base class for testing subclasses of base.ImageTfdsData.

  See base.ImageTfdsData for additional attributes.

  Attributes:
    tfds_label_key_map: Mapping specifying how to compare the number of labels
    with TFDS.
  """

  @abc.abstractmethod
  def setUp(self, data_wrapper, num_classes, default_label_key="label",
            tfds_label_key_map=None, **kwargs):
    super(BaseTfdsDataTest, self).setUp(
        data_wrapper=data_wrapper,
        num_classes=num_classes,
        default_label_key=default_label_key,
        **kwargs)
    # Set the tfds_label_key_map attribute.
    if isinstance(num_classes, int):
      if tfds_label_key_map is None:
        self.tfds_label_key_map = {default_label_key: default_label_key}
      elif tfds_label_key_map:
        if not isinstance(tfds_label_key_map, (str, list, tuple)):
          raise ValueError(
              "If `num_classes` is an int, `tfds_label_key_map` must be None, "
              "a string, or a tuple of strings.")
        self.tfds_label_key_map = {default_label_key: tfds_label_key_map}
      else:
        self.tfds_label_key_map = {}
    elif isinstance(num_classes, dict):
      if not (tfds_label_key_map is None or
              isinstance(tfds_label_key_map, dict)):
        raise ValueError(
            "If `num_classes` is a dict, `tfds_label_key_map` must be None or "
            "a dict.")
      self.tfds_label_key_map = tfds_label_key_map or {}
    else:
      raise ValueError("`num_classes` must be either int or dict")

  def test_base_class(self):
    """Tests that the dataset wrapper inherits from base.ImageData."""
    self.assertIsInstance(self.data_wrapper, base.ImageTfdsData,
                          "Dataset class must inherit from `base.ImageData`.")

  def test_split_dict_keys(self):
    """Tests the "tfds" and "num_samples" splits contain the correct keys."""
    super(BaseTfdsDataTest, self).test_split_dict_keys()
    expected_keys = set(self.expected_num_samples.keys())
    actual_keys = set(self.data_wrapper._tfds_splits.keys())  # pylint: disable=protected-access
    self.assertSetEqual(expected_keys, actual_keys)

  def test_get_num_classes(self):
    """Tests the expected number of classes."""
    super(BaseTfdsDataTest, self).test_get_num_classes()

    # Check get_num_classes output against TFDS.
    def _get_from_dict_recursive(key, features):
      """Returns an entry from a dict, recursively."""
      # Examples:
      # _get_from_dict_recursive("key", features) -> features["key"]
      # _get_from_dict_recursive(["key1", "key2"], features) ->
      #   features["key1"]["key2"]
      if isinstance(key, (list, tuple)) and len(key) > 1:
        return _get_from_dict_recursive(key[1:], features[key[0]])
      elif isinstance(key, (list, tuple)):
        return features[key[0]]
      else:
        return features[key]

    for label_key, tfds_label_key in self.tfds_label_key_map.items():
      # pylint: disable=protected-access
      tfds_num_classes = _get_from_dict_recursive(
          tfds_label_key,
          self.data_wrapper._dataset_builder.info.features).num_classes
      # pylint: enable=protected-access
      self.assertEqual(
          self.data_wrapper.get_num_classes(label_key), tfds_num_classes)


class BaseVTABDataTest(BaseTfdsDataTest):

  @property
  def expected_splits(self):
    return ("train", "val", "trainval", "test", "train800", "val200",
            "train800val200")
