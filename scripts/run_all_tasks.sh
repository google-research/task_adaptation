#!/bin/bash
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

# This script fine tunes a given Hub model on all tasks from the VTAB benchmark.
# It expects two parameters:
# 1. Hub model path or link.
# 2. Work directory path.

datasets=('caltech101' \
          'cifar(num_classes=100)' \
          'dtd' \
          'oxford_flowers102' \
          'oxford_iiit_pet' \
          'patch_camelyon' \
          'sun397' \
          'svhn' \
          'resisc45' \
          'eurosat' \
          'dmlab' \
          'kitti(task="closest_vehicle_distance")' \
          'smallnorb(predicted_attribute="label_azimuth")' \
          'smallnorb(predicted_attribute="label_elevation")' \
          'dsprites(predicted_attribute="label_x_position",num_classes=16)' \
          'dsprites(predicted_attribute="label_orientation",num_classes=16)' \
          'clevr(task="closest_object_distance")' \
          'clevr(task="count_all")' \
          'diabetic_retinopathy(config="btgraham-300")')

adapt_and_eval(){
    dataset=$1
    hub=$2
    workdir=$3
    adapt_and_eval.py \
      --hub_module $hub \
      --hub_module_signature default \
      --finetune_layer pre_logits \
      --work_dir $workdir/$dataset \
      --dataset $dataset \
      --train_examples 1000 \
      --batch_size 64 \
      --batch_size_eval 10 \
      --initial_learning_rate 0.01 \
      --decay_steps 300,600,900 \
      --input_range 0.0,1.0 \
      --max_steps 1000 \
      --warmup_steps 80 \
      --save_checkpoint_steps 1000
}

for dataset in "${datasets[@]}"
do
    adapt_and_eval $dataset $1 $2
done
