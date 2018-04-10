# ML-Project-GAN
Repository for the Machine Learning Project, Spring 2018.

[Project Introduction Presentation](https://docs.google.com/presentation/d/1hQ1Hkw4pFi2SgrW0e6UUQaoTpGMrgFOkR3J3KblcrJg/edit?usp=sharing
)

## GAN, Tensorflow and MNIST

To start, we reproduced the work done by Wiseodd, specifically their [vanilla_gan](https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py) in Tensorflow.

## Goals

+ Flowers Dataset
    - Create fake flower images of the various types of flowers
+ Faces Dataset 
    - Create faces of the various emotions.
    - Advanced: Use a specific starter face and manipulate it into the various emotions
+ GANs are hard – what if we fail?
    - Forget about the GAN and use a CNN to predict facial emotions

## Code Modifications

+ Change the structure of the tensors to match the dataset
+ Change the 2-layer neural net to a CNN
+ Advanced: Face manipulation
    - Use starter images instead of noise

