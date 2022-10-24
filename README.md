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

```
@inproceedings{wei2019spatial,
  title={Spatial component-wise convolutional network (SCCNet) for motor-imagery EEG classification},
  author={Wei, Chun-Shu and Koike-Akino, Toshiaki and Wang, Ye},
  booktitle={2019 9th International IEEE/EMBS Conference on Neural Engineering (NER)},
  pages={328--331},
  year={2019},
  organization={IEEE}
}
```

```
@article{lawhern2018eegnet,
  title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of neural engineering},
  volume={15},
  number={5},
  pages={056013},
  year={2018},
  publisher={IOP Publishing}
}
```

```
@article{schirrmeister2017deep,
  title={Deep learning with convolutional neural networks for EEG decoding and visualization},
  author={Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
  journal={Human brain mapping},
  volume={38},
  number={11},
  pages={5391--5420},
  year={2017},
  publisher={Wiley Online Library}
}
```
