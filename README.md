## Team-QuantumVisions
 Blending AI and math to create captivating art that bridges science and creativity.

## Revolutionary Technical Features

 AI-Powered Mathematical Artistry

Math-Driven Visualization

Creative AI Collaborator

Seamless AI Data Integration

Dynamic Mathematics 
Generator

## Installation:
<a name="installation"></a>

#### 1. Clone the repo

```shell
git clone git@github.com:Stability-AI/generative-models.git
cd generative-models
```

#### 2. Setting up the virtualenv

This is assuming you have navigated to the `generative-models` root after cloning it.

**NOTE:** This is tested under `python3.8` and `python3.10`. For other python versions, you might encounter version conflicts.


**PyTorch 1.13**

```shell
# install required packages from pypi
python3 -m venv .pt13
source .pt13/bin/activate
pip3 install -r requirements/pt13.txt
```

**PyTorch 2.0**


```shell
# install required packages from pypi
python3 -m venv .pt2
source .pt2/bin/activate
pip3 install -r requirements/pt2.txt
```


#### 3. Install `sgm`

```shell
pip3 install .
```

#### 4. Install `sdata` for training

```shell
pip3 install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
```

## Packaging

This repository uses PEP 517 compliant packaging using [Hatch](https://hatch.pypa.io/latest/).

To build a distributable wheel, install `hatch` and run `hatch build`
(specifying `-t wheel` will skip building a sdist, which is not necessary).

```
pip install hatch
hatch build -t wheel
```

You will find the built package in `dist/`. You can install the wheel with `pip install dist/*.whl`.

Note that the package does **not** currently specify dependencies; you will need to install the required packages,
depending on your use case and PyTorch version, manually.

## Inference

We provide a [streamlit](https://streamlit.io/) demo for text-to-image and image-to-image sampling in `scripts/demo/sampling.py`.
We provide file hashes for the complete file as well as for only the saved tensors in the file (see [Model Spec](https://github.com/Stability-AI/ModelSpec) for a script to evaluate that).
The following models are currently supported:

- [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  ```
  File Hash (sha256): 31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b
  Tensordata Hash (sha256): 0xd7a9105a900fd52748f20725fe52fe52b507fd36bee4fc107b1550a26e6ee1d7
  ```
- [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
  ```
  File Hash (sha256): 7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f
  Tensordata Hash (sha256): 0x1a77d21bebc4b4de78c474a90cb74dc0d2217caf4061971dbfa75ad406b75d81
  ```
- [SDXL-base-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
- [SDXL-refiner-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9)
- [SD-2.1-512](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.safetensors)
- [SD-2.1-768](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors)

**Weights for SDXL**:

**SDXL-1.0:**
The weights of SDXL-1.0 are available (subject to a [`CreativeML Open RAIL++-M` license](model_licenses/LICENSE-SDXL1.0)) here:
- base model: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/
- refiner model: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/


**SDXL-0.9:**
The weights of SDXL-0.9 are available and subject to a [research license](model_licenses/LICENSE-SDXL0.9).
If you would like to access these models for your research, please apply using one of the following links:
[SDXL-base-0.9 model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9), and [SDXL-refiner-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9).
This means that you can apply for any of the two links - and if you are granted - you can access both.
Please log in to your Hugging Face Account with your organization email to request access.


After obtaining the weights, place them into `checkpoints/`.
Next, start the demo using

```
streamlit run scripts/demo/sampling.py --server.port <your_port>
```

### Invisible Watermark Detection

Images generated with our code use the
[invisible-watermark](https://github.com/ShieldMnt/invisible-watermark/)
library to embed an invisible watermark into the model output. We also provide
a script to easily detect that watermark. Please note that this watermark is
not the same as in previous Stable Diffusion 1.x/2.x versions.

To run the script you need to either have a working installation as above or
try an _experimental_ import using only a minimal amount of packages:
```bash
python -m venv .detect
source .detect/bin/activate

pip install "numpy>=1.17" "PyWavelets>=1.1.1" "opencv-python>=4.1.0.25"
pip install --no-deps invisible-watermark
```

To run the script you need to have a working installation as above. The script
is then useable in the following ways (don't forget to activate your
virtual environment beforehand, e.g. `source .pt1/bin/activate`):
```bash
# test a single file
python scripts/demo/detect.py <your filename here>
# test multiple files at once
python scripts/demo/detect.py <filename 1> <filename 2> ... <filename n>
# test all files in a specific folder
python scripts/demo/detect.py <your folder name here>/*
```

## Training:

We are providing example training configs in `configs/example_training`. To launch a training, run

```
python main.py --base configs/<config1.yaml> configs/<config2.yaml>
```

where configs are merged from left to right (later configs overwrite the same values).
This can be used to combine model, training and data configs. However, all of them can also be
defined in a single config. For example, to run a class-conditional pixel-based diffusion model training on MNIST,
run

```bash
python main.py --base configs/example_training/toy/mnist_cond.yaml
```

**NOTE 1:** Using the non-toy-dataset configs `configs/example_training/imagenet-f8_cond.yaml`, `configs/example_training/txt2img-clipl.yaml` and `configs/example_training/txt2img-clipl-legacy-ucg-training.yaml` for training will require edits depending on the used dataset (which is expected to stored in tar-file in the [webdataset-format](https://github.com/webdataset/webdataset)). To find the parts which have to be adapted, search for comments containing `USER:` in the respective config.

**NOTE 2:** This repository supports both `pytorch1.13` and `pytorch2`for training generative models. However for autoencoder training as e.g. in `configs/example_training/autoencoder/kl-f4/imagenet-attnfree-logvar.yaml`, only `pytorch1.13` is supported.

**NOTE 3:** Training latent generative models (as e.g. in `configs/example_training/imagenet-f8_cond.yaml`) requires retrieving the checkpoint from [Hugging Face](https://huggingface.co/stabilityai/sdxl-vae/tree/main) and replacing the `CKPT_PATH` placeholder in [this line](configs/example_training/imagenet-f8_cond.yaml#81). The same is to be done for the provided text-to-image configs.

### Building New Diffusion Models

#### Conditioner

The `GeneralConditioner` is configured through the `conditioner_config`. Its only attribute is `emb_models`, a list of
different embedders (all inherited from `AbstractEmbModel`) that are used to condition the generative model.
All embedders should define whether or not they are trainable (`is_trainable`, default `False`), a classifier-free
guidance dropout rate is used (`ucg_rate`, default `0`), and an input key (`input_key`), for example, `txt` for text-conditioning or `cls` for class-conditioning.
When computing conditionings, the embedder will get `batch[input_key]` as input.
We currently support two to four dimensional conditionings and conditionings of different embedders are concatenated
appropriately.
Note that the order of the embedders in the `conditioner_config` is important.

#### Network

The neural network is set through the `network_config`. This used to be called `unet_config`, which is not general
enough as we plan to experiment with transformer-based diffusion backbones.

#### Loss

The loss is configured through `loss_config`. For standard diffusion model training, you will have to set `sigma_sampler_config`.

#### Sampler config

As discussed above, the sampler is independent of the model. In the `sampler_config`, we set the type of numerical
solver, number of steps, type of discretization, as well as, for example, guidance wrappers for classifier-free
guidance.

### Dataset Handling


For large scale training we recommend using the data pipelines from our [data pipelines](https://github.com/Stability-AI/datapipelines) project. The project is contained in the requirement and automatically included when following the steps from the [Installation section](#installation).
Small map-style datasets should be defined here in the repository (e.g., MNIST, CIFAR-10, ...), and return a dict of
data keys/values,
e.g.,

```python
example = {"jpg": x,  # this is a tensor -1...1 chw
           "txt": "a beautiful image"}
```

where we expect images in -1...1, channel-first format.
#   T e a m - Q u a n t u m V i s i o n s  
 #   T e a m - Q u a n t u m V i s i o n s  
 #   T e a m - Q u a n t u m V i s i o n s  
 