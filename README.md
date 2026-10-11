# COSC-5361-deepNeuralNetworks-midterm

## Dataset Setup

1. Download the Food-101 21-class subset from Kaggle:
   https://www.kaggle.com/datasets/dansbecker/food-101
2. Extract so the structure is:
   food-101/images/<class>/*.jpg
   food-101/meta/train.txt, test.txt
3. Set `dataPath` at the top of resnet_transfer.py
