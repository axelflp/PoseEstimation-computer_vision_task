# Pose Estimation Taks
Implementation of the basic U-net architecture for pose estimation task with the data available in [this link](https://www.kaggle.com/datasets/danielchang2002/openmonkeychallenge)

<div style="display: flex; justify-content: space-between;">
  <img src="images/example.jpg" alt="Example0" width="210"/>
  <img src="images/example1.jpg" alt="Example1" width="180"/>
  <img src="images/example2.jpg" alt="Example2" width="150"/>
  <img src="images/example3.jpg" alt="Example2" width="250"/>
</div>

## Architecture
Proposed in the medical context in 2015, has shown a good performance extracting information of images, it is compose of two parts, a downsampling phase follow by an upsampling phase, connected between the by skip conections and an bottleneck. Convolutions are the biulding block of this architecture.

In this implementation we used the same number of channels to define our model, exept from the first because our images contain 3 channels, and additionally we added a feeed forward neural network at the end to make the predictions.

<div style="display: flex; justify-content: space-between;">

  <img src="images/u-net.png" alt="Example0" width="600"/>

</div>

## Data
The data used to trained our model was released for the competition [OpenMonkeyChallenge](https://competitions.codalab.org/competitions/34342) in 2021. It is divided into train, validation and test data, and contains images and annotations for each image with the target landmarks.

## Preprocessing
As you can see in the images showed above, the images have diferent dimensions, perspectives, resolutions, so we need to stardize the the format. The applied transformations are as follow:

First off, we cut de images so that the figure of the monkey is centered and occupies most of the images. For this we laverage the bbox value contained in the annotations, this is a list of four numeric values that give us the coordinates of two points that delimit the region of the image that contain de monkey figure. We also adjust the landmarks to this transformation

Second, once we have cut the image it is necesary to stardadize the dimensions of the images in order to feed them into out model, so we reescale the images to 200x200 size, and adjusted our respective landmarks.



## Reviewing trained model










