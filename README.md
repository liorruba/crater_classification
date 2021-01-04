# Crater Classification
#### Also a final project for UCLA C247: neural networks

## Summary
In planetary remote sensing, training on imagery data is often demanding due to the need to collect hunders of samples, prepare and label them. This project adopts a semi-supervised training approach, where a GAN (generative adversarial network) is used to prepare training data and trains the binary classifier. This model is based on a [paper][https://papers.nips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf] by Salimans et al. from 2016. This implementation also draws some insperation from [this article](https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/).


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

Here I implement the Salimans+ 2016 model which employs a Softmax activation function for the supervised discriminator and an exponent-sum activation for the unsupervised discriminator, <img src="https://render.githubusercontent.com/render/math?math=D_{u} = \frac{\sum \exp{l_k(x)}}{1+\sum{\exp{l_k(x)}}}">, which is implemented using a Lambda layer.

<img src="https://github.com/liorruba/crater_classification/blob/main/craters.png" alt="Training samples. Top: craters. Bottom: non-craters" width="400"/>

## Results and Discussion
### CNN MODEL
To evaluate the SGAN model, I compare it with a simple CNN model trained on 11,200 samples of craters and “not craters”, which were randomly selected images obtained by the camera of the Lunar Reconnaissance Orbiter spacecraft (Robinson+ 2010). The data was validated on 2800 test samples. Albeit simplified, the model achieved very good classification results – probably because of the large training dataset. 

#### Layers and parameters
The model (see right table) employs an average pooling layer after the first convolution layer in order to smooth out smaller craters from being identified. This helped the model correctly classify only features on the same scale as the feature in question. The model was trained for 5 epochs using batches of 32 images. Using a higher number of epochs resulted in overfitting that could be slightly mitigated by adding a dropout layer. A batch normalization layer did not work, potentially due to the similarities in complexion between the images. I investigated and ruled out a possibility that a gradient trap (in a bad local minimum) is causing the decrease in by testing a few different learning rates. However, this problem repeats even in both high and low learning rates.

The model achieves accuracy of ~95% with 5 epochs, which was sufficient for this demonstration. The learning rate (0.1%) was also chosen on basis of trial and error as a compromise between model performance and running time.

<img src="https://github.com/liorruba/crater_classification/blob/main/fake_samples_at_ts_1600.png" alt="Fake crater samples proposed by the GAN." width="400"/>

### SGAN MODEL
The SGAN model performed significantly better than the CNN model for the crater dataset. I first tuned the model hyperparameters to achieve a high training accuracy while maintaining a reasonable validation accuracy to avoid overfitting. As an exploratory test, I first set the number of epochs equal to 10, and set the number of labeled samples to equal the batch size. The results of this test are shown in the figure below. It is interesting to see that both the number of labeled samples and the training batch size affect the model accuracy. When the model has not enough labeled samples, it is not properly trained. When the model has too large batch sizes, it affects the model’s ability to generalize. According to Keskar+ 2016, this is related to the gradient descent’s ability to converge and high uncertainty involved when using larger batches.

Finally, I also determined through trial and error that 5 epochs are sufficient to achieve good accuracy.

From the figure below, it seems it is best to use smaller batch sizes. As an example, I choose batch size = 10 and attempt to vary the number of labeled samples over 5 epochs.  This greatly improved the model accuracy compared to the CNN model, even when fully trained (see table below). In fact, due to the similar complexion of the craters and the other topographic features, the CNN based model did not converge at all (underfit) in all cases for which the number of samples was smaller than 1000.

<img src="https://github.com/liorruba/crater_classification/blob/main/accuracy.png" alt="Testing and training accuracy." width="400"/>

| Number of samples  |  CNN accuracy (%) after 5 epochs | SGN accuracy (%) after five epochs  |
|---|---|---|
| 10  |  N/A | 0.799  |
| 20  |  N/A | 0.871  |
| 50  |  N/A  | 0.905  |
| 100  |  N/A  | 0.917  |
| 1000 |  0.899 |  0.945 |

## Summary
Above I employed a SGAN model to classify topographic features on the Moon. Unlike images taken in color in Earth’s atmosphere, images obtained by spacecrafts often suffer from similarities which may lead to over- and underfitting when using traditional CNNs. Additionally, the labeling process of images obtained by these spacecrafts is demanding and, in some cases, impossible. Employing SGAN assisted classification solved both problems. The large unlabeled dataset allows achieving accuracy of over 90% with only 50 labeled examples for binary classification.



