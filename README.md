# 4DFace

4DFace is an open source face project , which contain face detection,face compare and face classification. 
 
## Requirements

* ubuntu 16.04
* python 2.7
* Dlib 19.10
* Torch
* Opencv 3.4.1

## Setup

### 1.Python and require packages

This project uses python2.7 and following packages can be installed with `pip`  or  `conda`

```
numpy = 1.4
scipy = 0.15
pandas = 0.16
scikit-learn = 0.17
nose = 1.3.1
nolearn == 0.5b1
dlib == 19.10
```

### 2.CUDA and cuDNN installation(optional)

If you have an NVIDIA GPU ,CUDA and cudnn is highly recommended 

**2.1 NVIDIA gpu driver installation**
```
sudo apt-get install nvidia-384
reboot
```
**2.2 CUDA installation**

CUDA can be downloaded from [here](https://developer.nvidia.com/cuda-downloads), click **Legacy Releases** , then choose **CUDA Toolkit 8.0 GA2** and download runfile(local).

```
sudo bash cuda_8.0.61_375.26_linux.run
```

There will be some options follow behind , **gpu driver installation option** choose **no**, because we have already installed ,the rest of other options are all **yes**,
after installation, add following command to `~/.bashrc`. 

```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```
then

```
source ~/.bashrc
```
**1.3 cuDNN installation**

cuDNN can be downloaded from [here](https://developer.nvidia.com/cudnn),choose **cuDNN v5.1**.

```
cd CUDA
cd include
sudo cp cudnn.h /usr/local/cuda-8.0/include/
cd ..
cd lib64
sudo cp lib* /usr/local/cuda-8.0/lib64/
```

### 3.Torch and packages installation

**3.1 Torch**

```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc
```
After this ,the command-line program `th` should be available in your shell,or you should check which step you made a mistake. 

**3.2 Packages**

```
luarocks install dpnn
luarocks install nn
luarocks install csvigo
luarocks install optim
```

If installed CUDA

```
luarocks install cutorch
luarocks install cunn

```

### 4. OpenCV installation

Here we only offered one way to install OpenCV,more specific OpenCV installation should consider official [tutorial](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html).

**4.1 Cmake gui installation**

```
sudo apt-get install cmake-qt-gui
```

**4.2 OpenCV installation**

Required packages 

```
[compiler] sudo apt-get install build-essential
[required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
[optional] sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

OpenCV releases can be found from [here](https://opencv.org/releases.html),download **3.4.1 from  Sources**

```
cd opencv
mkdir build
cd build
```
Configuring by cmake-gui

* open cmake
* set full path to OpenCV source code, e.g. /home/user/opencv
* set full path to <cmake_build_dir>, e.g. /home/user/opencv/build
* set optional parameters
* run: “Configure”
* run: “Generate”

then stay at **build dir**

```
cmake --build .
sudo make install 
```
## How to use

First,you can run following script to download necessary models and put them in `./models`

```
bash ./models/download_models.sh
```

### 1.face detect

This script will detect all the faces in a single image. 

```
python detector.py 'path\to\your\image\dir'
```

### 2.face compare

Face compare will return L2 distance of 2 faces,a small score means these 2 faces are very similar,of course,L2 distance between 2 same images is 0. 

```
python compare.py images/examples/test/{img1,img2}
```
this command will return L2 distance between `img1` and `img2`.

### 3.face classify

The pipeline of face classify has 4 steps or you can simply run

```
bash classifier_train.sh example/train/
```

This is the training step

```
bash classifier_test.sh example/test/wang.jpg
```

**3.1 Detect and align faces**

```
python align-dlib.py $INPUT align outerEyesAndNose ./aligned-placeholder/ --size 96
```
`$INPUT` is the dir of your input images,`./aligned-placeholder/`is where to output your aligned faces.

**3.2 Dimensionality reduction by neural network**

```
th ./network/main.lua -outDir ./represent/ -data ./aligned-placeholder/
``` 

`./represent/`is representation of the aligned faces, which is an 128 dimensions feature vectors.

**3.3 Train classifier**

```
python classifier.py train ./represent/
```
This step produce a SVM classification model for these 128 dimensions feature vector.

**3.4 Inference new faces**

```
python classifier.py infer ./represent/classifier.pkl $IMAGE
```
Inference new faces with confidence.
