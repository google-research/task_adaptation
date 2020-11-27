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

"""Tests for kitti.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_adaptation.data import data_testing_lib
from task_adaptation.data import kitti
import tensorflow.compat.v1 as tf


class KittiDataCountTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataCountTest, self).setUp(
        data_wrapper=kitti.KittiData(task="count_all"),
        num_classes=16,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class KittiDataCountLeftTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataCountLeftTest, self).setUp(
        data_wrapper=kitti.KittiData(task="count_left"),
        num_classes=16,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class KittiDataCountFarTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataCountFarTest, self).setUp(
        data_wrapper=kitti.KittiData(task="count_far"),
        num_classes=16,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class KittiDataCountNearTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataCountNearTest, self).setUp(
        data_wrapper=kitti.KittiData(task="count_near"),
        num_classes=16,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class KittiDataClosestDistanceTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataClosestDistanceTest, self).setUp(
        data_wrapper=kitti.KittiData(task="closest_object_distance"),
        num_classes=5,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class KittiDataClosestXLocTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataClosestXLocTest, self).setUp(
        data_wrapper=kitti.KittiData(task="closest_object_x_location"),
        num_classes=5,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class KittiDataCountVehiclesTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataCountVehiclesTest, self).setUp(
        data_wrapper=kitti.KittiData(task="count_vehicles"),
        num_classes=4,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class KittiDataClosestVehicleTest(data_testing_lib.BaseVTABDataTest):
  """See base class for usage and test descriptions."""

  def setUp(self):
    super(KittiDataClosestVehicleTest, self).setUp(
        data_wrapper=kitti.KittiData(task="closest_vehicle_distance"),
        num_classes=4,
        expected_num_samples=dict(
            train=6347,
            val=423,
            trainval=6770,
            test=711,
            train800val200=1000,
            train800=800,
            val200=200,
        ),
        required_tensors_shapes={
            "image": (None, None, 3),
            "label": (),
        },
        tfds_label_key_map={})


class TestPreprocessing(tf.test.TestCase):

  def test_count_vehicles(self):
    sess = tf.Session()
    x = {"image": tf.constant([0])}
    x["objects"] = {"type": tf.constant([0])}
    self.assertEqual(1, sess.run(kitti._count_vehicles_pp(x)["label"]))
    x["objects"] = {"type": tf.constant([3])}
    self.assertEqual(0, sess.run(kitti._count_vehicles_pp(x)["label"]))
    x["objects"] = {"type": tf.constant([0, 1])}
    self.assertEqual(2, sess.run(kitti._count_vehicles_pp(x)["label"]))
    x["objects"] = {"type": tf.constant([0, 1, 2])}
    self.assertEqual(3, sess.run(kitti._count_vehicles_pp(x)["label"]))
    x["objects"] = {"type": tf.constant([0, 1, 2, 2, 2, 2, 2])}
    self.assertEqual(3, sess.run(kitti._count_vehicles_pp(x)["label"]))

  def test_closest_vehicle(self):
    sess = tf.Session()
    x = {"image": tf.constant([0])}
    x["objects"] = {
        "type": tf.constant([0]),
        "location": tf.constant([[0.0, 0.0, 1.0]]),
    }
    self.assertEqual(0,
                     sess.run(kitti._closest_vehicle_distance_pp(x)["label"]))
    x["objects"] = {
        "type": tf.constant([0]),
        "location": tf.constant([[0.0, 0.0, 10.0]]),
    }
    self.assertEqual(1,
                     sess.run(kitti._closest_vehicle_distance_pp(x)["label"]))
    x["objects"] = {
        "type": tf.constant([0]),
        "location": tf.constant([[0.0, 0.0, 30.0]]),
    }
    self.assertEqual(2,
                     sess.run(kitti._closest_vehicle_distance_pp(x)["label"]))
    x["objects"] = {
        "type": tf.constant([4]),
        "location": tf.constant([[0.0, 0.0, 30.0]]),
    }
    self.assertEqual(3,
                     sess.run(kitti._closest_vehicle_distance_pp(x)["label"]))
    x["objects"] = {
        "type": tf.constant([0, 1]),
        "location": tf.constant([[0.0, 0.0, 30.0], [0.0, 0.0, 1.0]]),
    }
    self.assertEqual(0,
                     sess.run(kitti._closest_vehicle_distance_pp(x)["label"]))


if __name__ == "__main__":
  tf.test.main()
