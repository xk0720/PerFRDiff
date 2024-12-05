# PerFRDiff: Personalised Weight Editing for Multiple Appropriate Facial Reaction Generation
This repository contains a pytorch implementation of "PerFRDiff: Personalised Weight Editing for Multiple Appropriate Facial Reaction Generation".

## 👨‍🏫 Main Sections:
[//]: # (- [Overview]&#40;#overview&#41;)
[//]: # (- [Installation]&#40;#installation&#41;)
[//]: # (- [Dataset]&#40;#dataset&#41;)
[//]: # (- [Usage]&#40;#usage&#41;)

<details>
<summary><b>🛠️ Dependency Installation</b></summary>
<p>

[//]: # (## 🛠️ Dependency Installation)

We provide detailed instructions for setting up the environment using conda. First, create and activate a new environment:
``` shell
conda create -n react python=3.10
conda activate react
```

### 1. Install PyTorch
First, check your CUDA version:
``` shell
nvidia-smi
```
Visit [Pytorch official website](https://pytorch.org/) to get the appropriate installation command. For example:
``` shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Install PyTorch3D Dependencies
Install the following dependencies:
``` shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
For CUDA versions older than 11.7, you will need to install the CUB library. There are two installation options:

Option A: Using conda (Recommended)
``` shell
conda install -c bottler nvidiacub
```
Option B: Manual installation
1. Download the CUB library from NVIDIA CUB Releases.
2. Unpack it to a folder of your choice. For example, on Linux/Mac:
``` shell
cd ~
mkdir CUB
curl -LO https://github.com/NVIDIA/cub/archive/2.1.0.tar.gz
tar xzf 2.1.0.tar.gz
```
[//]: # (Define the environment variable CUB_HOME in `~/.bashrc` before building and point it to the directory that contains `CMakeLists.txt` for CUB.)
3. Define the environment variable CUB_HOME in `~/.bashrc`. This variable should point to the directory that contains `CMakeLists.txt` for CUB. Add this line to your `~/.bashrc`:
``` shell
export CUB_HOME=~/CUB/cub-2.1.0
```

[//]: # (Install jupyter-notebook and make the environment `react` available to jupyter-notebook by running)
To enable Jupyter notebook support, install Jupyter and register the environment:
``` shell
conda install jupyter
python -m ipykernel install --user --name=react
```

### 3. Install PyTorch3D
First, verify your CUDA version in Python:
``` shell
import torch
torch.version.cuda
```
[//]: # (Download `pytorch3d` file based on the version of python, cuda and pytorch from https://anaconda.org/pytorch3d/pytorch3d/files. For example, to install for Python 3.8, PyTorch 1.12.1 and CUDA 11.6, select the below file to download)
Download the appropriate `PyTorch3D` package from [Anaconda](https://anaconda.org/pytorch3d/pytorch3d/files) based on your Python, CUDA, and PyTorch versions. For example, for Python 3.10, CUDA 11.6, and PyTorch 1.12.0:

[//]: # (Finally install `pytorch3d` via the downloaded `.tar.bz2` file via conda)
``` shell
# linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
conda install linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
```

### 4. Install Additional Dependencies
[//]: # (pip install omegaconf scikit-video pandas soundfile av decord tensorboard numpy tslearn scikit-image matplotlib imageio plotly opencv-python librosa einops)
Install all remaining dependencies specified in requirements.txt:
``` shell
pip install -r requirements.txt
```

</p>
</details>

<details>
<summary><b>📊 Dataset</b></summary>
<p>

[//]: # (### Dataset)

</p>
</details>

<details>
<summary><b>📖 Usage</b></summary>
<p>

### Pre-trained Models
This project provides several pre-trained models, such as:
* Generic Appropriate Facial Reaction Generator (GAFRG)
* Personalized Weight Shifts Generation (PWSG) Block
* Personalized Style Space Learning (PSSL) Block

You can access and download all the available pre-trained models from the following [Google Drive link](https://drive.google.com/file/d/1Drdq3WnQjuOM9GxptC3UsTn_JSsn8_M-/view?usp=sharing). After downloading, please unzip the file and place the `checkpoints` folder into the root directory of this project.

### Training
``` python
# Training GAFRG for multiple appropriate facial reaction generation
python train_diffusion.py --mode train --config ./configs/diffusion_model.yaml

# Training Personalized GAFRG (with Weight Editing) for multiple appropriate facial reaction generation
python train_rewrite_weight.py --mode train --config ./configs/rewrite_weight.yaml
```

### Inference
``` python
# Inference using GAFRG for multiple appropriate facial reaction generation
python evaluate_diffusion.py --mode test --config ./configs/diffusion_model.yaml

# Inference using Personalised GAFRG (with Weight Editing) for multiple appropriate facial reaction generation
python evaluate_rewrite_weight.py --mode test --config ./configs/rewrite_weight.yaml
```

</p>
</details>

## 📽 Visualization of Facial Reactions

### Qualitative Results

[//]: # (Qualitative comparison between our method and existing baselines.)
[//]: # (![Comparison]&#40;docs/figures/comparison.png&#41;)

<details>
<summary>Qualitative comparison between our method and existing baselines (Click to expand) </summary>
<p>

![Comparison](docs/figures/comparison.png)

</p>
</details>

### Dynamic Visualization

https://github.com/user-attachments/assets/5799f032-3fbf-4f41-bd29-6293c6f4e151

## 🤝 Acknowledgement
We extend our sincere gratitude to the following open-source projects:
- [FaceVerse](https://github.com/LizhenWangT/FaceVerse)
- [PIRender](https://github.com/RenYurui/PIRender)

## 🖊️ Citation
```
@inproceedings{zhu2024perfrdiff,
  title={Perfrdiff: Personalised weight editing for multiple appropriate facial reaction generation},
  author={Zhu, Hengde and Kong, Xiangyu and Xie, Weicheng and Huang, Xin and Shen, Linlin and Liu, Lu and Gunes, Hatice and Song, Siyang},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9495--9504},
  year={2024}
}
```
