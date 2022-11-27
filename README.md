# EEG-KD

This repository is the official implementation of "Enhancing Low-Density EEG-Based Brain-Computer Interface with Knowledge Distillation". 


## Requirements
#### Step 1:
To install requirements:
```setup
git clone https://github.com/CECNL/EEG-KD.git
cd EEG-KD
conda env create -f KD.yaml
conda activate KD
```
#### Step 2:
Download [dataset](https://www.bbci.de/competition/iv/) and put them to the folder "BCICIV_2a".

## Training and testing

To train the teacher model or baseline student model (w/o KD), run this command:
```train_teacher
python train_teacher.py --model SCCNet22 --save_folder ./savedata/SCCNet22
```

To train the student model, run this command:
```train_student
python train_student.py --alpha 0.9 --beta 450 --save_folder ./savedata/SK --teacher_model SCCNet22 --teacher_folder ./savedata/SCCNet22
```

## Reference

If you use this our codes in your research, please cite our paper and the related references in your publication as:
```bash
@article{,
  title={},
  author={},
  journal={arXiv preprint},
  year={2022}
}
```
If you use the SCCNet model, please cite the following:
```bash
@inproceedings{wei2019spatial,
  title={Spatial component-wise convolutional network (SCCNet) for motor-imagery EEG classification},
  author={Wei, Chun-Shu and Koike-Akino, Toshiaki and Wang, Ye},
  booktitle={2019 9th International IEEE/EMBS Conference on Neural Engineering (NER)},
  pages={328--331},
  year={2019},
  organization={IEEE}
}
```
If you use the EEGNet model, please cite the following:
```bash
@article{Lawhern2018,
  author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
  title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
  journal={Journal of Neural Engineering},
  volume={15},
  number={5},
  pages={056013},
  url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
  year={2018}
}
```
If you use the ShalowConvNet model, please cite the following:
```bash
@article{hbm23730,
author = {Schirrmeister Robin Tibor and 
          Springenberg Jost Tobias and 
          Fiederer Lukas Dominique Josef and 
          Glasstetter Martin and 
          Eggensperger Katharina and 
          Tangermann Michael and 
          Hutter Frank and 
          Burgard Wolfram and 
          Ball Tonio},
title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
journal = {Human Brain Mapping},
volume = {38},
number = {11},
pages = {5391-5420},
keywords = {electroencephalography, EEG analysis, machine learning, end‐to‐end learning, brain–machine interface, brain–computer interface, model interpretability, brain mapping},
doi = {10.1002/hbm.23730},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730}
}
```

