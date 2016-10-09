#DCSM:Distinct class-specific saliency maps
By Watal Shimoda, Keiji Yanai.
![Flow](https://github.com/shimoda-uec/dcrm/blob/master/process.png "flow")
##Description
This repository contains the codes for the "DCSM" weakly supervised semantic segmentation method.  
It has been published at ECCV2016.  
Our codes are based on the MatConvNet deep learning library.  
Caffe implementation is also here [Caffe implementaion](https://github.com/shimoda-uec/dcsm).  
Caluculation detail and computational cost are different.
##Requirements
Requirements for MatConvNet (see: [MatConvNet installation instructions](http://www.vlfeat.org/matconvnet/))  
##Install
First, you should clone the repository as below.  
```
git clone https://github.com/shimoda-uec/mat_dcsm.git
```
##Usage 
A main code is dcsm.m file.
You need some modified files for MatConvNet and downloading network parameters for MatConvNet format to start up.
The modified file locations are following:  
*matlab/vl_simplenn.m
*matlab/vl_gbp.m
*matlab/vl_nnsigmoid.m
*matlab/vl_nnrelu2.m

Trained network model is here [MatConvNet model](http://mm.cs.uec.ac.jp/shimoda-k/models/mp512_iter_20000.caffemodel).
##License and Citation
Please cite our paper if it helps your research:
```
@inproceedings{shimodaECCV16  
  Author = {Shimoda, Wataru and Yanai, Keiji},  
  Title = {Distinct class-specific saliency maps},  
  Booktitle = {International Conference on Computer Vision ({ECCV})},  
  Year = {2016}  
}  
```
