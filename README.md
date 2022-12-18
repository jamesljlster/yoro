# YORO: A YOLO Variant for Rotated Object Detection

YORO, extended from YOLO feature map encoding, is an algorithm performimg
simultaneous realtime object detection and rotation detection.

<img width="650" src=".assets/demo.png" />

The documentation project is here:
<https://github.com/jamesljlster/yoro-tutorial>  
But only Chinese version is available currently.  
If you want an English version, please open an issue for it.  
At least let me know you are interested in my project :D

The project is not stable yet.  
There are no guarantes for API compatibility.

## Feature Map Encoding and Decoding

-   YORO use the same bounding box encoding as YOLOv4:  
    <https://github.com/WongKinYiu/ScaledYOLOv4/issues/90>

-   As for degree, please refer to the following methods:

    -   Constants

        Given: <img src="https://latex.codecogs.com/svg.image?d_{min},d_{max},d_{size}" />  
        <img src="https://latex.codecogs.com/svg.image?d_{orig}=d_{min}-d_{size}\div&space;2" />

        -   Where

            <img src="https://latex.codecogs.com/svg.image?d_{min}" /> is the minimum degree.  
            <img src="https://latex.codecogs.com/svg.image?d_{max}" /> is the maximum degree.  
            <img src="https://latex.codecogs.com/svg.image?d_{size}" /> is the degree partition size.  
            <img src="https://latex.codecogs.com/svg.image?d_{orig}" /> is the origin of degree encoding axis.

    -   Encoding

        <img src="https://latex.codecogs.com/svg.image?d_{norm}=(d_{target}-d_{orig})\div&space;d_{size}" /><br>
        <img src="https://latex.codecogs.com/svg.image?d_{label}=\left&space;\lfloor&space;d_{norm}\right&space;\rfloor" /><br>
        <img src="https://latex.codecogs.com/svg.image?d_{shift}=(d_{norm}-d_{label}-0.5)\times&space;2" />

        -   Where

            <img src="https://latex.codecogs.com/svg.image?d_{norm}" /> is normalized degree scalar.  
            <img src="https://latex.codecogs.com/svg.image?d_{label}" /> is degree partition index.  
            <img src="https://latex.codecogs.com/svg.image?d_{shift}" /> is degree shift scalar based on corresponding partition.

    -   Loss

        <img src="https://latex.codecogs.com/svg.image?d_{loss}=cross\_entropy(v_{part},d_{label})&plus;mse(v_{shift}[d_{label}],d_{shift})" />

        -   Where

            <img src="https://latex.codecogs.com/svg.image?d_{loss}" /> is total degree loss.  
            <img src="https://latex.codecogs.com/svg.image?v_{part}" /> is output logits of degree partition.  
            <img src="https://latex.codecogs.com/svg.image?v_{shift}" /> is output vector of degree shift.  
            The lengths of both output logits and vector are the same as <img src="https://latex.codecogs.com/svg.image?\left&space;\lfloor&space;(d_{max}-d_{orig})\div&space;d_{size}&plus;0.5\right&space;\rfloor" />.

    -   Decoding

        <img src="https://latex.codecogs.com/svg.image?p_{label}=argmax(v_{part})" /><br>
        <img src="https://latex.codecogs.com/svg.image?p_{shift}=v_{shift}[p_{label}]\div&space;2&plus;0.5" /><br>
        <img src="https://latex.codecogs.com/svg.image?d_{pred}=d_{size}\times&space;(p_{label}&plus;p_{shift})" />

        -   Where

            <img src="https://latex.codecogs.com/svg.image?p_{label}" /> is degree partition index prediction.  
            <img src="https://latex.codecogs.com/svg.image?p_{shift}" /> is degree shift prediction.  
            <img src="https://latex.codecogs.com/svg.image?d_{pred}" /> is decoded degree prediction.

## Requirement

See [requirements.txt][] for Python package
dependencies.  
The following dependencies need to pay attention:

-   PyTorch 1.11.0 and TorchVision 0.12.0

    Compatibilty with other versions is not guaranteed.

-   CUDA Toolkit & cuDNN

    If your PyTorch was built with CUDA support, please install the
    corresponding version of CUDA toolkit and cuDNN.

Optional dependencies:

-   \[Optional\] OpenCV 4.0.0+

    YORO C++ API optionally requires OpenCV C++ development package for
    providing cv::Mat inference interface support.

### Installation Instructions

The following instructions demonstrate package installation with
conda based environments such as Miniconda or Anaconda.  
You don’t have to follow this if YORO can be setup correctly by yourself.

``` bash
# Create and activate a fresh conda environment
conda create -n yoro-env python==3.8.15
conda activate yoro-env  # This conda environment should be kept activated for the rest of instructions

# Install PyTorch 1.11.0 and corresponding CUDA Toolkit
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit

# Install some build dependendies (cmake, gcc)
conda install -c conda-forge "gxx_linux-64<11"
conda install -c anaconda cmake

# Build and install YORO package
git clone https://github.com/jamesljlster/yoro.git
cd yoro
pip install -v .
cd ..

# Test if everything is fine
python -c "import yoro"
```

## Collaboration Tools

-   [ICANMark][]: Annotation tool for rotated bounding box.

## Acknowledgement

Thanks ICAL Lab <http://www.ical.tw/> for providing a good workstation
for project development.

## Citation

If this project helps your work, please kindly cite it :)

    @unpublished{yoro,
        title={YORO: A YOLO Variant for Rotated Object Detection},
        author={Cheng-Ling Lai},
        note={Project URL: https://github.com/jamesljlster/yoro},
        year={2020}
    }

## Reference

-   <https://github.com/AlexeyAB/darknet>
-   <https://github.com/pjreddie/darknet>
-   <https://github.com/eriklindernoren/PyTorch-YOLOv3>
-   <https://github.com/WongKinYiu/ScaledYOLOv4>
-   <https://github.com/Zzh-tju/CIoU>

  [requirements.txt]: requirements.txt
  [ICANMark]: https://github.com/jamesljlster/ican_mark
