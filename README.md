# CNN-based coastline detection pipeline
This is a pipeline of training and validating a CNN-based image segmentation which is based on image-segmentation-keras package (https://github.com/divamgupta/image-segmentation-keras).

The extra repositories that you will be needed to utilize during the pipeline execution are (will explain later):
1) https://github.com/divamgupta/image-segmentation-keras
2) https://github.com/wkentaro/labelme
3) https://github.com/aleju/imgaug

## The SW/HW setup
This pipeline runs on Python3. For this reason the user has to setup a python3 virtual environment and installing a set of dependencies inside. Also if you are an NVIDIA user then you have to check the CUDA versions, and cuDNN. Else you will be running the NN prediction on your CPU (either way the payload to the computer is heavy but GPU acceleration helps a lot).
The basic dependencies for the python3 are installed using the bellow commands:
```
$ cd ~
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install build-essential cmake unzip pkg-config
$ sudo apt install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
$ sudo apt install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt install libxvidcore-dev libx264-dev
$ sudo apt install libgtk-3-dev
$ sudo apt install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
$ sudo apt install libhdf5-serial-dev
$ sudo apt install python3-dev python3-tk python-imaging-tk
```
After installing the dependencies you have to install Anaconda 3 (https://docs.anaconda.com/anaconda/install/linux/) for the Ubuntu 18.04 version.
After the Anaconda installation you create an anaconda virtual environment with the following terminal commands:
```
$ source ~/anaconda3/etc/profile.d/conda.sh
$ conda create -n tf-gpu-cuda10 tensorflow-gpu=1.14 cudatoolkit=10.0 python=3.6
$ conda activate tf-gpu-cuda10
$ conda install -c conda-forge keras=2.2.5
$ pip install keras-segmentation
```
Now you have to be inside the virtual environment so you have to install all the pip dependencies:
```
$ pip install numpy
$ pip install scipy matplotlib pillow
$ pip install imutils h5py==2.10.0 requests progressbar2
$ pip install cython
$ pip install scikit-learn scikit-build scikit-image
$ pip install opencv-contrib-python==4.4.0.46
$ pip install tensorflow-gpu==1.14.0
$ pip install keras==2.2.5
$ pip install opencv-python==4.4.0.42
$ pip install keras-segmentation
$ pip install rospkg empy
```
Now you have to check the import of the keras and tensorflow:
```
$ python
$ >>> import tensorflow
$ >>>
$ >>> import keras
$ Using TensorFlow backend.
$ >>>
$ >>> import keras_segmentation
$ >>>
```
If everything succesful you can check your python 3 virtual environment running the image-segmentation-keras tutorial (https://github.com/divamgupta/image-segmentation-keras) with the dataset given from the framework. It takes about 4-6 hours (depending on the PC's we have tested till this day) so you can leave at night. If all the predictions are ok then your python 3 virtual environment is ready for use.
The commands and choice of packages depends on the HW you utilize. We implemented everything on Ubuntu 18.04 with Python 3.6 and a GPU GTX 1070.

## Frames collection and pre-processing
Usually the acquired will come on a video format. You have to split the video in frames and start labeling.
A popular tool for this procedure is the labelme package https://github.com/wkentaro/labelme.
After you complete the coping procedure of by hand labeling which depends on the image segmentation you want to implement (i.e. specific target detection) you should have a folder with the original frames and a folder with the labels on a .json format.
Once you go through with this procedure you are able to execute the whole pipeline.

## Pipeline execution
At first you should create some folders:
```
$ mkdir Masks ResizedFr ResizedMasks AugFr AugMasks DisplayFrMasks_before_classification CMasks TestFr TestMasks Test_output DisplayFrMasks_after_training
```
Then you can begin executing the following commands:
```
$ ipython TakeMasks.ipynb
$ ipython Resize.ipynb
```
After this part you can implement Augmentation tools that are utilized in the Augment.ipynb file according to what you wish about your application and the versatility you wish. This can be a trial and error procedure which tou can test only after training. So you choose the least ones for beginning and then enrich the augmentation tools accordnig to your results:
```
$ ipython Augment.ipynb
```
After this part you can test if something has gone wrong with all the previous steps by running the combined display between you frames and masks:

```
$ ipython DisplayFrMasks.ipynb
```
If you are sure for the steps taken till now then you can proceed with the classification of the images:
```
$ ipython ClassMasks.ipynb
```
When this is completed then you have to split the dataset on training and validation part. From ResizedFr and CMasks you can move the 10-15% to TestFr and TestM respectively.
Then you are rady for training:
```
$ ipython ModelTraing.ipynb
```
After several hours (according to the HW utilized) all the checkpoints of the procedure will be saved on the directory. Each checkpoint corresponds to an epoch of training. You can pick the epoch you deceided that the model converged you can try the validation of the trained model:
```
$ ipython Predict.ipynb
$ ipython DisplayFrMasks_mobilenet_segnet.ipynb
```
According to the results on folder Test_output and DisplayFrMasks_after_training you can decide about the efficiency of the training procedure and what you wish to re-implement and configure.
In our case we implemented these pipeline for coastline detection both on synthetic and on real outdoors datasets. The procedure on the real data was more challenging with a lot of back and forth to decide a final configuration of the whole pipeline to have desire results.
