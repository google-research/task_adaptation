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

"""Abstract class for reading the data using tfds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf
import tensorflow_datasets as tfds


def make_get_tensors_fn(output_tensors):
  """Create a function that outputs a collection of tensors from the dataset."""

  def _get_fn(data):
    """Get tensors by name."""
    return {tensor_name: data[tensor_name] for tensor_name in output_tensors}

  return _get_fn


def make_get_and_cast_tensors_fn(output_tensors):
  """Create a function that gets and casts a set of tensors from the dataset.

  Optionally, you can also rename the tensors.

  Examples:
    # This simply gets "image" and "label" tensors without any casting.
    # Note that this is equivalent to make_get_tensors_fn(["image", "label"]).
    make_get_and_cast_tensors_fn({
      "image": None,
      "label": None,
    })

    # This gets the "image" tensor without any type conversion, casts the
    # "heatmap" tensor to tf.float32, and renames the tensor "class/label" to
    # "label" and casts it to tf.int64.
    make_get_and_cast_tensors_fn({
      "image": None,
      "heatmap": tf.float32,
      "class/label": ("label", tf.int64),
    })

  Args:
    output_tensors: dictionary specifying the set of tensors to get and cast
      from the dataset.

  Returns:
    The function performing the operation.
  """

  def _iter_dictionary():
    for tensor_name, tensor_dtype in output_tensors.items():
      if isinstance(tensor_dtype, tuple) and len(tensor_dtype) == 2:
        yield tensor_name, tensor_dtype[0], tensor_dtype[1]
      elif tensor_dtype is None or isinstance(tensor_dtype, tf.dtypes.DType):
        yield tensor_name, tensor_name, tensor_dtype
      else:
        raise ValueError('Values of the output_tensors dictionary must be '
                         'None, tf.dtypes.DType or 2-tuples.')

  def _get_and_cast_fn(data):
    """Get and cast tensors by name, optionally changing the name too."""

    return {
        new_name:
        data[name] if new_dtype is None else tf.cast(data[name], new_dtype)
        for name, new_name, new_dtype in _iter_dictionary()
    }

  return _get_and_cast_fn


def compose_preprocess_fn(*functions):
  """Compose two or more preprocessing functions.

  Args:
    *functions: Sequence of preprocess functions to compose.

  Returns:
    The composed function.
  """

  def _composed_fn(x):
    for fn in functions:
      if fn is not None:  # Note: If one function is None, equiv. to identity.
        x = fn(x)
    return x

  return _composed_fn


@six.add_metaclass(abc.ABCMeta)
class ImageData(object):
  """Abstract data provider class.

  IMPORTANT: You should use ImageTfdsData below whenever is posible. We want
  to use as many datasets in TFDS as possible to ensure reproducibility of our
  experiments. Your data class should only inherit directly from this if you
  are doing experiments while creating a TFDS dataset.
  """

  @abc.abstractmethod
  def __init__(self,
               num_samples_splits,
               shuffle_buffer_size,
               num_preprocessing_threads,
               num_classes,
               default_label_key='label',
               base_preprocess_fn=None,
               filter_fn=None,
               image_decoder=None,
               num_channels=3):
    """Initializer for the base ImageData class.

    Args:
      num_samples_splits: a dictionary, that maps splits ("train", "trainval",
          "val", and "test") to the corresponding number of samples.
      shuffle_buffer_size: size of a buffer used for shuffling.
      num_preprocessing_threads: the number of parallel threads for data
          preprocessing.
      num_classes: int/dict, number of classes in this dataset for the
        `default_label_key` tensor, or dictionary with the number of classes in
        each label tensor.
      default_label_key: optional, string with the name of the tensor to use
        as label. Default is "label".
      base_preprocess_fn: optional, base preprocess function to apply in all
        cases for this dataset.
      filter_fn: optional, function to filter the examples to use in the
        dataset.
      image_decoder: a function to decode image.
      num_channels: number of channels in the dataset image.
    """
    self._log_warning_if_direct_inheritance()
    self._num_samples_splits = num_samples_splits
    self._shuffle_buffer_size = shuffle_buffer_size
    self._num_preprocessing_threads = num_preprocessing_threads
    self._base_preprocess_fn = base_preprocess_fn
    self._default_label_key = default_label_key
    self._filter_fn = filter_fn
    self._image_decoder = image_decoder
    self._num_channels = num_channels

    if isinstance(num_classes, dict):
      self._num_classes = num_classes
      if default_label_key not in num_classes:
        raise ValueError(
            'No num_classes was specified for the default_label_key %r' %
            default_label_key)
    elif isinstance(num_classes, int):
      self._num_classes = {default_label_key: num_classes}
    else:
      raise ValueError(
          '"num_classes" must be a int or a dict, but type %r was given' %
          type(num_classes))

  @property
  def default_label_key(self):
    return self._default_label_key

  @property
  def label_keys(self):
    return self._num_classes.keys()

  @property
  def num_channels(self):
    return self._num_channels

  def get_num_samples(self, split_name):
    return self._num_samples_splits[split_name]

  def get_num_classes(self, label_key=None):
    if label_key is None:
      label_key = self._default_label_key
    return self._num_classes[label_key]

  def get_tf_data(self,
                  split_name,
                  batch_size,
                  preprocess_fn=None,
                  epochs=None,
                  drop_remainder=True,
                  for_eval=False,
                  shuffle_buffer_size=None,
                  prefetch=1,
                  train_examples=None,
                  filtered_num_samples=None):
    """Provides preprocessed and batched data.

    Args:
      split_name: name of a data split to provide. Can be "train", "val",
          "trainval" or "test".
      batch_size: batch size.
      preprocess_fn: a function for preprocessing input data. It expects a
          dictionary with a key "image" associated with a 3D image tensor.
      epochs: number of full passes through the data. If None, the data is
          provided indefinitely.
      drop_remainder: if True, the last incomplete batch of data is dropped.
          Normally, this parameter should be True, otherwise it leads to
          the unknown batch dimension, which is not compatible with training
          or evaluation on TPUs.
      for_eval: get data for evaluation. Disables shuffling.
      shuffle_buffer_size: overrides default shuffle buffer size.
      prefetch: number of batches to prefetch.
      train_examples: optional number of examples to take for training.
        If greater than available number of examples, equivalent to None (all).
        Ignored with for_eval is True.
      filtered_num_samples: required when filter_fn is set, number of
        samples after applying filter_fn.

    Returns:
      A tf.data.Dataset object as a dictionary containing the output tensors.
    """

    # Obtains tf.data object.
    # We shuffle later when not for eval, it's important to not shuffle before
    # a subset of data is retrieved.
    data = self._get_dataset_split(
        split_name=split_name,
        shuffle_files=not (for_eval or train_examples))

    if not for_eval:
      # Dataset filtering priority: (1) filter_fn; (2) train_examples.
      if self._filter_fn and train_examples:
        raise ValueError('You must not set both filter_fn and train_examples.')
      if self._filter_fn:
        tf.logging.warning(
            'You are filtering the dataset. Notice that this may hurt your '
            'throughput, since examples still need to be decoded, and may '
            'make the result of get_num_samples() inacurate. '
            'train_examples is ignored for filtering, but only used for '
            'calculating training steps.')
        data = data.filter(self._filter_fn)
        num_samples = filtered_num_samples
        assert num_samples is not None, (
            'You must set filtered_num_samples once filter_fn is set.')
        # Get actual train examples...
      elif train_examples:
        # Deterministic for same dataset version.
        data = data.take(train_examples)
        num_samples = train_examples
      else:
        num_samples = self.get_num_samples(split_name)

      # Cache the whole dataset if it's smaller than 150K examples
      if num_samples <= 150000:
        data = data.cache()

    # Repeats data `epochs` time or indefinitely if `epochs` is None.
    if epochs is None or epochs > 1:
      data = data.repeat(epochs)

    shuffle_buffer_size = shuffle_buffer_size or self._shuffle_buffer_size
    if not for_eval and shuffle_buffer_size > 1:
      data = data.shuffle(shuffle_buffer_size)

    # Compose the base_preprocess_fn and the given preprocess_fn.
    preprocess_fn = compose_preprocess_fn(self._image_decoder,
                                          self._base_preprocess_fn,
                                          preprocess_fn)

    # Currently, map_and_batch provides noticable (almost 2-fold) speedup as
    # compared to using non-fused operations.
    data = data.apply(tf.data.experimental.map_and_batch(
        map_func=preprocess_fn,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        num_parallel_calls=self._num_preprocessing_threads))

    return data.prefetch(prefetch)

  @abc.abstractmethod
  def _get_dataset_split(self, split_name, shuffle_files=False):
    """Return the Dataset object for the given split name.

    Args:
      split_name: Name of the dataset split to get.
      shuffle_files: Whether or not to shuffle files in the dataset.

    Returns:
      A tf.data.Dataset object containing the data for the given split.
    """

  def _log_warning_if_direct_inheritance(self):
    tf.logging.warning(
        'You are directly inheriting from ImageData. Please, consider porting '
        'your dataset to TFDS (go/tfds) and inheriting from ImageTfdsData '
        'instead.')


class ImageTfdsData(ImageData):
  """Abstract data provider class for datasets available in Tensorflow Datasets.

  To add new datasets inherit from this class. See imagenet.py for an example.
  This class implements a simple API that is used throughout the project and
  provides standardized way of data preprocessing and batching.
  """

  @abc.abstractmethod
  def __init__(self, dataset_builder, tfds_splits, image_key='image', **kwargs):
    """Initializer for the base ImageData class.

    Args:
      dataset_builder: tfds dataset builder object.
      tfds_splits: a dictionary, that maps splits ("train", "trainval", "val",
          and "test") to the corresponding tfds `Split` objects.
      image_key: image key.
      **kwargs: Additional keyword arguments for the ImageData class.
    """
    self._dataset_builder = dataset_builder
    self._tfds_splits = tfds_splits
    self._image_key = image_key

    # Overwrite image decoder
    def _image_decoder(data):
      decoder = dataset_builder.info.features[image_key].decode_example
      data[image_key] = decoder(data[image_key])
      return data
    self._image_decoder = _image_decoder

    kwargs.update({'image_decoder': _image_decoder})

    super(ImageTfdsData, self).__init__(**kwargs)

  def _get_dataset_split(self, split_name, shuffle_files):
    dummy_decoder = tfds.decode.SkipDecoding()
    return self._dataset_builder.as_dataset(
        split=self._tfds_splits[split_name], shuffle_files=shuffle_files,
        decoders={self._image_key: dummy_decoder})

  def _log_warning_if_direct_inheritance(self):
    pass
