# KnowSAM
Official code for "[Learnable Prompting SAM-induced Knowledge Distillation for Semi-supervised Medical Image Segmentation](https://arxiv.org/pdf/2412.13742)"

## Installation

To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```

## Extract Sample Data

We provide a reference sample dataset (SampleData.rar) that allows users to quickly test and run the model. Extract the dataset using the following command:
```bash
unrar x SampleData.rar
```
For processed ACDC dataset, you can download it from the [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC), and place it directly in the `SampleData` folder.


## Training
To train the model on a dataset, execute:
```bash
python train_semi_SAM.py
```

For ACDC dataset training:
```bash
python train_semi_SAM_ACDC.py
```

## Prediction
After training, you can make predictions using:
```bash
python prediction.py
```

For ACDC dataset inference:
```bash
python prediction_ACDC.py
```

## Acknowledgements
Our code is based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

## Questions
If you have any questions, welcome contact me at 'taozhou.dreams@gmail.com'
