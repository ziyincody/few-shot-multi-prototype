# few-shot-multi-prototype

This multi-prototype method is based off PANet `https://github.com/kaixin96/PANet`

After `git clone`, create a folder named `pretrained_model` and put https://download.pytorch.org/models/vgg16-397923af.pth under it

Download the dataset from https://drive.google.com/file/d/1pY4uFUxXVUModA0AQqMEwc2Qmj-DZcAa/view?usp=sharing and place it in the folder

Follow few-shot.ipynb to setup your training in google colab.

Snapshot for Set 2 1way1shot with vgg backbone + multi-prototype + k-means adjustment + distance loss => 0.504 
https://drive.google.com/file/d/18rnxDxccjyaJ0exNaQmLBsD-CZ_NDdJI/view?usp=sharing

Snapshot for Set2 1way1shot with vgg backbone + multi-prototype + k-means adjustment => 0.503
https://drive.google.com/file/d/16IZtq3a_Lk_jref9G3hnQWnar22iqLlH/view?usp=sharing

Original paper's snapshot I was only able to achieve 0.477 on colab after training

Other things that have tried which didn't help
- Using saliency classifier to assist
- Training saliency on a dominant dataset then assist
- multi step k-means

Useful resources
- Few-shot semantic segmentation paper https://github.com/xiaomengyc/Few-Shot-Semantic-Segmentation-Papers
- Meta-learning tutorial https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html