---
title: FLAIR - Federated Learning Annotated Image Repository
url: https://github.com/apple/ml-flair
labels: [image classification, multiclass, cross-device, large]
dataset: [FLAIR]
---

# FLAIR - Federated Learning Annotated Image Repository

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html

**Authors:** *_Congzheng Song, Filip Granqvist, Kunal Talwar_*

**Abstract:** Cross-device federated learning is an emerging machine learning (ML) paradigm
where a large population of devices collectively train an ML model while the data
remains on the devices. This research field has a unique set of practical challenges,
and to systematically make advances, new datasets curated to be compatible with
this paradigm are needed. Existing federated learning benchmarks in the image
domain do not accurately capture the scale and heterogeneity of many real-world
use cases. We introduce FLAIR, a challenging large-scale annotated image dataset
for multi-label classification suitable for federated learning. FLAIR has 429,078
images from 51,414 Flickr users and captures many of the intricacies typically
encountered in federated learning, such as heterogeneous user data and a long-tailed
label distribution. We implement multiple baselines in different learning setups
for different tasks on this dataset. We believe FLAIR can serve as a challenging
benchmark for advancing the state-of-the art in federated learning. Dataset access
and the code for the benchmark are available at https://github.com/apple/
ml-flair

## About this baseline

The code in this directory can be used to reproduce the results reported in the paper *FLAIR - Federated Learning Annotated Image Repository* on the [FLAIR](https://github.com/apple/ml-flair) dataset.
The paper reports results on centralized (C), federated learning (FL) and private federated learning (PFL), but this directory only focus on the FL case.
The original paper used 4 Nvidia A100 GPUs (50% of a `p4d.24xlarge` on AWS to be precise) to train.
There is no specific minimum requirements since the user datasets are small and the model is ResNet18, but recommended is at least 4 GPUs.

**Contributors:** [Filip Granqvist](https://github.com/grananqvist).

## Experimental Setup

**Task:** Multi-class image classification task with cross-device FL characteristics.

**Model:** ResNet18. Trained from scractch or pre-trained with ImageNet.

**Dataset:** See [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html) and [original repo](https://github.com/apple/ml-flair) for more information on the FLAIR dataset. This setup tries to replicate the original benchmark setup.

**Training Hyperparameters:** The hyperparameters are available in `flair/conf/base.yaml`.

## Download and preprocess data

The dataset repo has utilities to download and preprocess the dataset:

```
git clone git@github.com:apple/pfl-research.git
cd pfl-research/benchmarks

pip install Pillow h5py numpy datasets
mkdir -p ../../data
python -m dataset.flair.download_preprocess --output_file ../../data/flair_federated.hdf5
```

There is also a small preprocessed dataset available for testing here:
```
wget https://pfl-data.s3.us-east-2.amazonaws.com/flair_federated_small.h5
```

## Environment Setup

```
pyenv local 3.10.6
poetry env use 3.10.6
poetry install
poetry shell
```

## Running the Experiments

Make sure `data_path` in the config points to your preprocessed HDF5 file, and then run using default settings:
```
python -m flair.main
```

## Expected Results

The commands below will reproduce the most important "FL" baselines from Table 1 of the [FLAIR paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html):

<div style="border: 1px solid #f0ad4e; background-color: #fcf8e3; padding: 10px; border-radius: 4px;">
<strong>Warning:</strong> The benchmarks are not fully reproduced yet. Marco-averaged AP for `FL-F-17` is 31.5, but expected to be 62.0.
</div>

**`FL-R-17` -- Federated Learning from scratch with coarse-grained labels**
Expected macro AP: 50.1±0.5
```bash
poetry run python -m flair.main pretrained=false
```

**`FL-F-17` -- Federated Learning with pre-trained model and coarse-grained labels**
Expected macro AP: 62.0±0.3
```bash
poetry run python -m flair.main
```

**`FL-R-1628` -- Federated Learning from scratch with fine-grained labels**
Expected micro AP: 22.5±0.3
```bash
poetry run python -m flair.main pretrained=false dataset.use_fine_grained_labels=true
```

**`FL-F-1628` -- Federated Learning with pre-trained model and fine-grained labels**
expected micro AP: 27.0±0.4
```bash
poetry run python -m flair.main dataset.use_fine_grained_labels=true
```
