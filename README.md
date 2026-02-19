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

![unet](images/u-net.png){width=200}
