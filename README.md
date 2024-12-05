# PerFRDiff: Personalised Weight Editing for Multiple Appropriate Facial Reaction Generation
This repository contains a pytorch implementation of "PerFRDiff: Personalised Weight Editing for Multiple Appropriate Facial Reaction Generation"

## 📖 Main Sections:
[//]: # (- [Overview]&#40;#overview&#41;)
[//]: # (- [Installation]&#40;#installation&#41;)
[//]: # (- [Dataset]&#40;#dataset&#41;)
[//]: # (- [Usage]&#40;#usage&#41;)

<details>
<summary><b>Installation</b></summary>
<p>

[//]: # (### Installation)

</p>
</details>

<details>
<summary><b>Dataset</b></summary>
<p>

[//]: # (### Dataset)

</p>
</details>

<details>
<summary><b>Usage</b></summary>
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

## 👀 Visualization of Facial Reactions

### Qualitative Results

[//]: # (Qualitative comparison between our method and existing baselines.)
[//]: # (![Comparison]&#40;docs/figures/comparison.png&#41;)

<details>
<summary>Qualitative comparison between our method and existing baselines</summary>
<p>

![Comparison](docs/figures/comparison.png)

</p>
</details>

### Dynamic Visualization

https://github.com/user-attachments/assets/5799f032-3fbf-4f41-bd29-6293c6f4e151

## TODO
- [ ] Installation
- [ ] Dataset

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

## 🤝 Acknowledgement
We extend our sincere gratitude to the following open-source projects:
- [FaceVerse](https://github.com/LizhenWangT/FaceVerse)
- [PIRender](https://github.com/RenYurui/PIRender)
