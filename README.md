## Visual Task Adaptation Benchmark (VTAB)

This repository contains code for evaluating visual models on a challenging
set of downstream vision tasks, coming from diverse domains: natural images,
artificial environments (structured) and images captured with non-standard
cameras (specialized). These tasks, together with our evaluation protocol,
constitute *VTAB*, short for Visual Task Adaptation Benchmark.

The *VTAB* benchmark contains the following 19 tasks that are derived from the
public datasets:

- [*Caltech101*](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
- [*CIFAR-100*](https://www.cs.toronto.edu/~kriz/cifar.html)
- [*CLEVR distance prediction*](https://cs.stanford.edu/people/jcjohns/clevr/)
- [*CLEVR counting*](https://cs.stanford.edu/people/jcjohns/clevr/)
- [*Diabetic Rethinopathy*](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
- [*Dmlab Frames*](https://arxiv.org/abs/1910.04867)
- [*dSprites orientation prediction*](https://github.com/deepmind/dsprites-dataset)
- [*dSprites location prediction*](https://github.com/deepmind/dsprites-dataset)
- [*Describable Textures Dataset (DTD)*](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- [*EuroSAT*](https://github.com/phelber/eurosat)
- [*KITTI distance prediction*](http://www.cvlibs.net/datasets/kitti/)
- [*102 Category Flower Dataset*](http://www.robots.ox.ac.uk/~vgg/data/flowers/)
- [*Oxford IIIT Pet dataset*](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [*PatchCamelyon*](https://github.com/basveeling/pcam)
- [*Resisc45*](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)
- [*Small NORB azimuth prediction*](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
- [*Small NORB elevation prediction*](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
- [*SUN397*](https://groups.csail.mit.edu/vision/SUN/)
- [*SVHN*](http://ufldl.stanford.edu/housenumbers/)

Our benchmark expects a pretrained model as an input. The model should be
provided as a [Hub module](https://www.tensorflow.org/hub). The given model is
independently fine tuned for solving each of the above 20 tasks. Average
accuracy across all tasks is used to measure the model's performance.

Detailed description of all tasks, evaluation protocol and other details
can be found in the [VTAB paper](https://arxiv.org/abs/1910.04867).

## Installation

You can install the library and all necessary dependencies by running:
`pip install -e .` from the `task_adaptation/` folder.

## Dataset preparation

VTAB uses the [tensorflow datasets library (TFDS)](https://www.tensorflow.org/datasets)
that automatically downloads and preprocesses VTAB datasets. *TFDS* will
download and preprocess a dataset when it is used for the first time.
Subsequently, it will reuse already downloaded and preprocessed dataset.

We recommend triggering dataset downloading and preprocessing before running
*VTAB* benchmark. Otherwise, if downloading of one of the datasets fails
(because of an unstable internet connection or missing dependency) you will get
an incomplete result.

Dataset downloading can be triggered in various ways. We recommend running
dataset tests located in `task_adaptation/data` folder. Execution of each test
will result in the corresponding dataset being downloaded and preprocessed.
In order to download and preprocess datasets faster we recommend running all
data tests in parallel:

```
# run inside task_adaptation/data
$ find . -name '*_test.py' | xargs -P14 -IX python X
```

Data preparation may take substantial amount of time: up to a day depending on
your internet connection and CPU model.

Note, that *Diabetic Retinopathy* and *Resisc45* datasets can not be downloaded
automatically, so a user should download these dataset themself, see [TFDS
documentation](https://www.tensorflow.org/datasets) for more details.

#### Install dataset specific dependencies

Data preprocessing step requires additional python dependencies to be installed:

```
$ pip install opencv-python scipy
```

Depending on your Linux Distribution, `opencv-python` library may be missing
some binary dependencies. For instance, if you are using `Debian 9` you may need
to install extra packages using the following command:

```
$ apt-get install libsm6 libxext6 libxrender-dev
```

## Running locally

After installing `task_adaptation` package you can adapt and evaluate
Hub modules locally by running `adapt_and_eval.py` script. For instance, the
following command adapts publicly available ResNet50 v2 model trained on
ImageNet for the CIFAR-100 dataset:

```
adapt_and_eval.py \
    --hub_module https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3  \
    --hub_module_signature image_feature_vector \
    --finetune_layer resnet_v2_50/global_pool \
    --work_dir /tmp/cifar100 \
    --dataset 'cifar(num_classes=100)' \
    --batch_size 64 \
    --batch_size_eval 10 \
    --initial_learning_rate 0.01 \
    --decay_steps 300,600,900 \
    --max_steps 1000
```

Results are written into `result_file.txt` which is created in the directory
specified by the `--work_dir` flag.

In the `scripts/` directory we provide a `run_all_tasks.sh` script that runs
our benchmark on all tasks from the VTAB benchmark.

## Running on Google Cloud TPU

In order to run VTAB on Cloud TPU follow these steps:

#### Create Cloud TPU Instance

For the comprehensive documentation on how to set up Google Cloud TPU visit
[this link](https://cloud.google.com/tpu/docs/).

If you are familiar with Google Cloud infrastructure and already have an account
there, you can skip the above step, go to the Google Cloud Console and run the
following command:

```
$ ctpu up \
  --name <NAME OF A CLOUD TPU INSTANCE> \
  --tpu-size v3-8 \
  --zone <ZONE> \
  --project <COMPUTE PROJECT NAME>
```

After running the above command you will be able to find a new VM instance in
the Google Cloud UI with the instructions of how to log in the newly created
machine.

Note, in this example we use a TPU machine with 8 TPU cores, which is a
recommended setup.

#### Create a bucket in Google Cloud Storage

After logging to the VM instance, create a Cloud Storage bucket for storing
data and checkpoints:

```
$ export BUCKET=<CLOUD STORAGE BUCKET NAME>
$ gsutil mb gs://$BUCKET
```

Note, that Cloud TPU does not support local file system, so the above step is
absolutely necessary.

#### Setup tensorflow hub cache directory

Now we need to setup tensorflow hub to cache Hub modules in our bucket:

```
$ export TFHUB_CACHE_DIR=gs://$BUCKET/tfhub-cache/
```

#### Install VTAB

Install VTAB as described in the installation section above.

#### Install additional python dependencies

Install python packages required by Cloud TPU:

```
$ pip install --upgrade google-api-python-client oauth2client
```

#### Run adapt_and_eval.py command line utility

Now everything is ready for running adaptation and evaluation of your Hub model.

The following command provides an example that adapts publicly available
ResNet-50 v2 model trained on the ImageNet dataset to the CIFAR-100 dataset.
The resulting accuracy on CIFAR-100 test set should be around 83%.

```
$ adapt_and_eval.py \
    --hub_module https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3  \
    --hub_module_signature image_feature_vector \
    --finetune_layer resnet_v2_50/global_pool \
    --work_dir gs://$BUCKET/cifar100 \
    --dataset 'cifar(num_classes=100)' \
    --batch_size 512 \
    --batch_size_eval 512 \
    --initial_learning_rate 0.01 \
    --decay_steps 750,1500,2250 \
    --max_steps 2500 \
    --data_dir gs://$BUCKET/tfds-data \
    --tpu_name <NAME OF A CLOUD TPU INSTANCE>  # The same name as used in the `ctpu` call
```

Results are written into `result_file.txt` which is created in the directory
specified by the `--work_dir` flag.
