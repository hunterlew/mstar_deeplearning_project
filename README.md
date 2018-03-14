mstar_deeplearning_project
==========================
The repository is my graduation project about radar target classification, detection and recognition on public MSTAR using deep learning method. The main framework is based on caffe and faster-rcnn using matlab interface with a bit modification. 

Besides, there is another [repository](https://github.com/hunterlew/convolution_network_on_FPGA) built recently about my graduation project, dealing with network acceleration on FPGA.

![p1.jpg](https://github.com/hunterlew/mstar_deeplearning_project/blob/master/detection_and_recognition/p1.jpg)
![p2.jpg](https://github.com/hunterlew/mstar_deeplearning_project/blob/master/detection_and_recognition/p2.jpg)
![p3.jpg](https://github.com/hunterlew/mstar_deeplearning_project/blob/master/detection_and_recognition/p3.jpg)

Pre-requisites
--------------
The project is supposed to run on win7 or above. Before running the project, please checkout if your PC supports Nvidia GPU computing with compute capability 6.1 like GTX1080 and cuda v8.0, and a certain higher version of Matlab, like Matlab 2015b. Besides, python3.5 is needed and I recommend you directly install Anaconda3-4.2.0-Windows-x86_64.exe and add it to system path. No other installation and compilation is required since the repository is a release version. You can also make your own changes by compiling [caffe](https://github.com/BVLC/caffe/tree/windows) and [faster-rcnn](https://github.com/ShaoqingRen/faster_rcnn) yourselves. 

<code>git clone git@github.com:hunterlew/mstar_deeplearning_project.git</code>

Classification
--------------
The first part of the work focuses on 10-class radar target classification on standard MSTAR dataset. For avoiding overfitting, I fulfilled 96*96 SAR target classification with data augmentation using random cropping, proving that it can easily outperform [traditional machine learning methods](https://github.com/hunterlew/mstar_with_machine_learning). 

Run the commands below and you may get 96% ~ 99% accuracy:

<code>cd classification\caffe</code><br />
<code>(run the data_augmentation.m with matlab)</code><br />
<code>(run the generate_file.m with matlab)</code><br />
<code>create_mstar.bat</code><br />
<code>train_mstar.bat</code><br />

Detection and Recognition
-------------------------
The second part is about how to locate and recognize several SAR targets in a larger background, which may also contain trees and houses, etc. In view of ShaoqingRen's RPN networks, I builded two models with datasets that I made myself. The first model takes only RPN's output as the input of classification network trained before. The second model partially shares the convolution layers between RPN and classification network, which is called faster-rcnn by Ren. You can respectively run the two models and make comparisons. Make sure you have downloaded the pretrained ZF model and mean.mat from [here](https://github.com/ShaoqingRen/faster_rcnn). Then run the commands:

<code>cd detection_and_recognition\core</code><br />
<code>(run the train.m with matlab)</code><br />

It will take more than an hour for overall training and will finally generate output folder with trained model. Remember to copy the RPN's net file and trained model to the output folder and rename them, serving as network files for the run_apart model. 

To run the first model:

<code>cd detection_and_recognition</code><br />
<code>(run the run_apart.m with matlab)</code><br />

To run the second model:

<code>cd detection_and_recognition</code><br />
<code>(run the run_overall.m with matlab)</code><br />

To validate model performance, such as the missing detection rate, false detection rate, recognition rate and the running time:

<code>cd detection_and_recognition</code><br />
<code>(run the run_apart_validation.m or run_overall_validation.m with matlab)</code><br />

Conclusion
----------
The results seemed successful. But it may be doubtful that I directly inserted several targets, under a certain lightness, into the background without considering the reasonability and the characteristics of SAR images. Therefore, the work needs further considerations and research. 
