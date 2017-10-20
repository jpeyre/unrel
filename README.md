# Weakly-supervised learning of visual relations

Created by Julia Peyre at INRIA, Paris.

### Introduction

This is the code for the paper :

Julia Peyre, Ivan Laptev, Cordelia Schmid, Josef Sivic, Weakly-supervised learning of visual relations, ICCV17.

The webpage for this project is available [here](http://www.di.ens.fr/willow/research/unrel/), with a link to the [paper](http://www.di.ens.fr/willow/research/unrel/paper.pdf) and [UnRel dataset](http://www.di.ens.fr/willow/research/unrel/data/unrel-dataset.tar.gz). 

### License

The code and dataset are available for research purpose. Check the LICENSE file for details. 

### Cite

If you find this code useful in your research, please, consider citing our paper:

> @InProceedings{Peyre17,
>   author      = "Peyre, Julia and Laptev, Ivan and Schmid, Cordelia and Sivic, Josef",
>   title       = "Weakly-supervised learning of visual relations",
>   booktitle   = "ICCV",
>   year        = "2017"
>}


### Related work

If you are interested in learning more about the optimization with discriminative clustering, check :
 
Miech, Antoine and Alayrac, Jean-Baptiste and Bojanowski, Piotr and Laptev, Ivan and Sivic, Josef, Learning from video and text via large-scale discriminative clustering ([paper](https://arxiv.org/abs/1707.09074),[code](https://github.com/antoine77340/iccv17learning))

### Contents

  1. [Dependencies](#dependencies)
  2. [Data](#data)
  3. [Demo](#demo)
  4. [Running on new images](#running-on-new-images)

### Dependencies

To run this code, you need to install : 
1. [MOSEK](https://www.mosek.com/downloads/) : version 7 
2. [CVX](http://cvxr.com/cvx/download/) : version 2.1 
3. [VLFEAT](http://www.vlfeat.org/download.html) : version 0.9.20

Once installed, setup the paths in startup file :
```Matlab
startup.m
```


### Data

To use this code with Visual Relationship Detection dataset and UnRel, follow the following steps to get the data.

1. **Download the pre-processed data** 
```Shell
wget http://www.di.ens.fr/willow/research/unrel/release/preproc_data.zip
unzip preproc_data.zip
```

This repository contains the folders for the datasets. The structure of the folders is as follows :

```Shell
./data/
------ models/
------------- gmm-{weak,full}.mat   # GMM model for quantized spatial configurations trained with weak/full supervision
------------- pca-{weak,full}.mat   # PCA model for dimension reduction on appearance features trained with weak/full supervision
------------- vgg16_fast_rcnn_iter_80000.caffemodel   # VGG16 object detector caffemodel
------------- test.prototxt                           # VGG16 object detector prototxt
------ classifiers/   # trained models
------ vrd-dataset/   # dataset name
------------------ test/   # split
------------------------ candidates/   # candidates name
----------------------------------- pairs.mat     # candidates pairs of boxes stored in a structure :
						  # pairs.im_id : image id for the pair of boxes
						  # pairs.sub_id : id of the subject box
						  # pairs.obj_id : id of the object box
						  # pairs.rel_id : id for the pair of boxes
						  # pairs.sub_cat : subject category
						  # pairs.obj_cat : object category
						  # pairs.rel_cat : predicate category
						  # pairs.sub_box : coordinates of subject box [xmin,ymin,xmax,ymax]
						  # pairs.obj_box : coordinates of object box [xmin,ymin,xmax,ymax]
----------------------------------- objects.mat   # candidates objects
----------------------------------- objectscores.mat   # object scores for the candidates objects computed with object detector
----------------------------------- features/
-------------------------------------------- appearance-{weak,full}/   # appearance features for the candidates objects for each image (first column indicates obj_id)  
```

The folder candidates/ can contain different types of candidates :
- **annotated**: candidates pairs corresponding to groundtruth positive annotations
- **candidates**: candidates pairs of boxes obtained with selective search
- **gt-candidates**: candidates pairs of boxes built from all groundtruth object boxes (but not necessarily annotated)
- **Lu-candidates**: candidates pairs of boxes provided by [Lu16]
- **densecap-candidates**: candidates regions obtained with DenseCap method [Johnson16]

The folder features/ only contains the pre-processed appearance features for the objects (given by an object detector, and after applying PCA). Now, we need to compute the spatial features for the pairs of boxes. 


2. **Extract the spatial features**  
To compute the spatial features, run :
```Matlab
populate_spatialfeats.m
```
This step will pre-compute the spatial features for all candidates pairs of boxes in each image.  


### Demo

Now you can run our demo script to train and/or evaluate : 
```Matlab
   demo.m
```

You can change the training/evaluation options by modifying the object opts. Refer to config.m to see the different options available for training and evaluation. We also provide scripts in folder experiments/ to reproduce the results in our paper. 


You will need approximately 10G memory to re-train our weakly-supervised model on Visual Relationship Dataset. 


### Running on new images

You might want to test our model on new images. For this, follow these steps :

1. Get candidate pairs of boxes for the new images (you can use different proposal methods, e.g. [selective search](https://github.com/sergeyk/selective_search_ijcv_with_python))
2. Run our pre-trained [Fast-RCNN](https://github.com/rbgirshick/py-faster-rcnn) object detector on Caffe to get object scores and extract the appearance features
3. Apply PCA for dimension reduction on appearance features
4. Run compute_spatial_features.m to compute the quantized spatial features

You can contact the first author for more information regarding the pre-processing steps on new images.  



