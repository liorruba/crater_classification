# Crater Classification
## Final project for UCLA C247: neural networks

## Summary
In planetary remote sensing, training on imagery data is often demanding due to the need to collect hunders of samples, prepare and label them. This project adopts a semi-supervised training approach, where a GAN (generative adversarial network) is used to prepare training data and trains the binary classifier. This model is based on a [paper][https://papers.nips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf] by Salimans et al. from 2016.


## Introduction
Impact craters are ubiquitous throughout the solar system. These bowl-shaped depressions are formed when a meteorite, which is a rocky or an icy fragment orbiting the Sun, collides with the surface of a larger object. Upon impact, the kinetic energy of the meteorite, given as the product of its mass by its velocity squared, is invested in excavating the crater by forming a shock wave that compresses and ejects material from the surface. As a result, the final size of an impact crater is a proxy to the energy of the object that formed it and the physical properties of the target material.

In the absence of a protective atmosphere, even the smallest meteorite impacts on the surface of the Moon form craters. Assuming the impact flux (rate per unit area) is known, the number of craters on a given surface may be used to determine its age. The power-law size distribution of the impactor population in the solar system dictates that small crater would be significantly more abundant than large craters whose formation occurs on much longer time scales. Consequently, estimating the size distribution of small craters is crucial to our understanding of how recently resurfaced topographic features, such as volcanoes, form and evolve.

## Scope
As explained above, the power law size distribution of impactors and the craters they form makes small crater significantly more abundance than large craters. For example, a recent survey conducted manually found more than 1.5 million impact craters > 1 km on the Moon. It should now be clear that manually counting craters smaller than this size range is virtually impossible.
The goal of this project is to develop a deep learning binary classifier that would be able to classify craters from other topographic features. The main problem with previous efforts was that they employed a convolutional neural network, and thus required a substantial training dataset to be collected manually. Even though millions of images of the lunar surface have been collected thusfar, only a handful were manually classified. Here I will attempt to implement a model that would use the smallest possible training dataset that would still be reasonably accurate, by using a semi-supervised GAN (SGAN). 
Even though binary classification is considered an “easier” problem compared to multi-class classifications, it is important to remember topographic features captured by spacecrafts often look similar and usually consist of the similar colors and shades. As a result, a CNN is often prone to underfitting or overfitting.

## Methods
In a semi-supervised Generative adversarial network (SGAN), the synthetic (fake) images produced by the generator (Salimans+ 2016) are used to boost the performance of a classifier. For example, in an image classification problem with N classes, the images synthetically produced by the generator constitute the the (N+1)th class. The discriminator model learns whether an image is real or synthetic (not from the dataset), and as a result extracts features from the unlabeled data, greatly improving classification accuracy.

In order to train the SGAN model, the discriminator is trained both on the supervised and the unsupervised data simultaneously.  In the unsupervised training part, the model learns features from the unlabeled data. In its supervised mode, the model classifies and labels the data.

Here I implement the Salimans+ 2016 model which employs a Softmax activation function for the supervised discriminator and an exponent-sum activation for the unsupervised discriminator, $$D_unsupervised(x) = \frac{\Sum \exp{l_k(x)}}{1+\Sum{\exp{l_k(x)}}$$, which is implemented using a Lambda layer.
