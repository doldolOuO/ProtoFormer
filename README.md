# Point Cloud Semantic Scene Completion with Prototype-Guided Transformer (AAAI 2026)

## 🌱 Datasets
### SSC-PC & NYUCAD-PC
The SSC-PC and NYUCAD-PC dataset used in this work is sourced from the [CasFusionNet](https://github.com/JinfengX/CasFusionNet).

## 🚀 Getting Started
### Requirements
- Ubuntu: 18.04 and above
- CUDA: 11.3 and above
- PyTorch: 1.10.1 and above

### Using CUDA extension
activate your environment and then
```
cd cuda/ChamferDistance
python setup.py install
```
and
```
cd cuda/pointnet2_ops_lib
python setup.py install
```

### Training
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
### Evaluation
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```
### Pre-trained weights
- [SSC-PC](https://drive.google.com/file/d/1StgClDcE9VaymA9B6zkdNabtHj_NDzQu/view?usp=drive_link)

- [NYUCAD-PC](https://drive.google.com/file/d/1wJuXDbD81THfG2Cig9Ljd16SS_HXqlb8/view?usp=drive_link)

## ❤️ Acknowledgements
Some of the code of this repo is borrowed from:

- [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet)

- [ChamferDistance](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

- [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)

## 📄 Cite this work

```bibtex
@inproceedings{fang2026protoformer,
  title={Point Cloud Semantic Scene Completion with Prototype-Guided Transformer},
  author={Chenghao Fang and Jianqing Liang and Jiye Liang and Zijin Du and Feilong Cao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```

## 📌 License

This project is open sourced under MIT license.
