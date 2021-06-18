# CataNet: Predicting remaining cataract surgery duration


This repository contains the code accompanying the paper "CataNet: Predicting remaining cataract surgery duration", accepted for publication at MICCAI 2021. 

### Abstract

Cataract surgery is a sight saving surgery that is performed over 10 million times each year around the world. With such a large demand, the ability to organize surgical wards and operating rooms efficiently is critical to delivery this therapy in routine clinical care. In this context, estimating the remaining surgical duration (RSD) during procedures is one way to help streamline patient throughput and workflows. To this end, we propose CataNet, a method for cataract surgeries that predicts in real time the RSD jointly with two influential elements: the surgeon's experience, and the current phase of the surgery. We compare CataNet to state-of-the-art RSD estimation methods, showing that it outperforms them even when phase and experience are not considered. We investigate this improvement\reviewed{ and show that a significant contributor is the way we integrate the elapsed time into CataNet's feature extractor.

# Instructions

To reproduce our experiments, please download the [cataract 101 dataset](http://ftp.itec.aau.at/datasets/ovid/cat-101/). 

With the dataset, we extracted the frames by running: 

``python utils/process_videos.py --label cataract101 --input path/to/cataract101 --output data``



## Train CataNet

To train the model, first run the train_cnn 



## Resources


### Acknowledgments

This code was developed jointly by Michel Hayoz and Andrés Marafioti. The development was partially supported by the Haag-Streit Foundation and the University of Bern.
