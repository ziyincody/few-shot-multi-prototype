# few-shot-multi-prototype

This multi-prototype method is based off PANet `https://github.com/kaixin96/PANet`

After `git clone`, create a folder named `pretrained_model` and put https://download.pytorch.org/models/vgg16-397923af.pth under it

Download the dataset from https://drive.google.com/file/d/1pY4uFUxXVUModA0AQqMEwc2Qmj-DZcAa/view?usp=sharing and place it in the folder

Follow few-shot.ipynb to setup your training in google colab.

Set 2 1way1shot with vgg backbone + multi-prototype + k-means adjustment + distance loss => 0.504 
https://drive.google.com/file/d/18rnxDxccjyaJ0exNaQmLBsD-CZ_NDdJI/view?usp=sharing

Set 2 1way1shot with vgg backbone + multi-prototype + k-means adjustment => 0.503
https://drive.google.com/file/d/16IZtq3a_Lk_jref9G3hnQWnar22iqLlH/view?usp=sharing
