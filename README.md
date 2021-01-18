# Transfer Learning for Humanoid Robot Appearance-based Localization in a Visual Map

This is the code implementation accompanying IEEE Access paper of the same title (https://ieeexplore.ieee.org/abstract/document/9312592). 

Update (Jan 18, 2021): Feel free to contact me by email: e.ovallemagallanes@ugto.mx for any further details/contributions.

Citation:

```
@ARTICLE{9312592,
  author={E. {Ovalle-Magallanes} and N. G. {Aldana-Murillo} and J. G. {Avina-Cervantes} and J. {Ruiz-Pinales} and J. {Cepeda-Negrete} and S. {Ledesma}},
  journal={IEEE Access}, 
  title={Transfer Learning for Humanoid Robot Appearance-Based Localization in a Visual Map}, 
  year={2021},
  volume={9},
  number={},
  pages={6868-6877},
  doi={10.1109/ACCESS.2020.3048936}}
```


# Abstract

Autonomous robot visual navigation is a fundamental locomotion task based on extracting relevant features from images taken from the surrounded environment to control an independent displacement. In the navigation, the use of a known visual map helps obtain an accurate localization, but in the absence of this map, a guided or free exploration pathway must be executed to obtain the images sequence representing the visual map. This paper presents an appearance-based localization method based on a visual map and an end-to-end Convolutional Neural Network (CNN). The CNN is initialized via transfer learning (trained using the ImageNet dataset), evaluating four state-of-the-art CNN architectures: VGG16, ResNet50, InceptionV3, and Xception. A typical pipeline for transfer learning includes changing the last layer to adapt the number of neurons according to the number of custom classes. In this work, the dense layers after the convolutional and pooling layers were substituted by a Global Average Pooling (GAP) layer, which is parameter-free. Additionally, an L2 -norm constraint was added to the GAP layer feature descriptors, restricting the features from lying on a fixed radius hypersphere. These different pre-trained configurations were analyzed and compared using two visual maps found in the CIMAT-NAO datasets consisting of 187 and 94 images, respectively. For evaluating the localization tasks, a set of 278 and 94 images were available for each visual map, respectively. The numerical results proved that by integrating the L2 -norm constraint in the training pipeline, the appearance-based localization performance is boosted. Specifically, the pre-trained VGG16 and Xception networks achieved the best localization results, reaching a top-3 accuracy of 90.70% and 93.62% for each dataset, respectively, overcoming the referenced approaches based on hand-crafted feature extractors.

# Use the datasets and the proposed model

You may download the repository and set your current working directory in the main.ipynb file. In this file, you can configure the parameters: i.e., dataset, network, learning-rate, epochs, number of augmented images. 

## Contact

e.ovallemagallanes@ugto.mx

Google Scholar Profile: https://scholar.google.com/citations?user=zql1lk8AAAAJ&hl=es