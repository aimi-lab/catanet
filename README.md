# CataNet: Predicting remaining cataract surgery duration


This repository contains the code accompanying the paper "CataNet: Predicting remaining cataract surgery duration", accepted for publication at MICCAI 2021. 

### Abstract

Cataract surgery is a sight saving surgery that is performed over 10 million times each year around the world. With such a large demand, the ability to organize surgical wards and operating rooms efficiently is critical to delivery this therapy in routine clinical care. In this context, estimating the remaining surgical duration (RSD) during procedures is one way to help streamline patient throughput and workflows. To this end, we propose CataNet, a method for cataract surgeries that predicts in real time the RSD jointly with two influential elements: the surgeon's experience, and the current phase of the surgery. We compare CataNet to state-of-the-art RSD estimation methods, showing that it outperforms them even when phase and experience are not considered. We investigate this improvement\reviewed{ and show that a significant contributor is the way we integrate the elapsed time into CataNet's feature extractor.

# Instructions

To reproduce our experiments, please download the [cataract 101 dataset](http://ftp.itec.aau.at/datasets/ovid/cat-101/) and our modified [label files](https://doi.org/10.5281/zenodo.4984167). 
Put all video and label files into a single folder and run
```
cd tools
python process_videos.py --input path/to/cataract101 --output ../data/cataract101
```
The videos are sampled at 2.5 fps and resized to 256x256 pixels. Additionally, the sequence before the start and after 
the end of the surgery are discarded.
Once extracted, split the dataset into train/val/test. To reproduce the 6-fold-crossvalidation in the paper run the following, where you choose which fold to extract (1 to 6).
```
cd tools
python split_data.py --input ../data/cataract101 --out ../data/cataract101 --fold 1
```

## Train CataNet

Train first the 2D-CNN on surgical phase and experience prediction with 
```
python train_cnn.py --out path/to/cnn/folder --log True
```
Then train the RNN on phase, experience and RSD prediction with
```
python train_rnn.py --out path/to/rnn/folder --log True --pretrained path/to/cnn/folder/catRSDNet_CNN.pth
```

### Evaluation
Test the trained model on a test set by running:
```
cd tools
python inference_catrsdnet.py --out path/to/output --checkpoint path/to/model.pth --input path/to/dataset/test
```
In the output folder there is a csv-file with the model inference and a visualization for each video. Additionally, 
the script prints the summary performance statistics for the whole test set. Performance metrics are macro averaged over
videos.

## Resources

You are going to need ffmpeg.

### Acknowledgments

This code was developed jointly by Michel Hayoz and Andrés Marafioti. The development was partially supported by the Haag-Streit Foundation and the University of Bern.
