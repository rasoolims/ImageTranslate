# ImageTranslate

This code is used for learning translation and captioning models. More information about how to run this code will be written very soon.



## Installation using virtualenv
Here, I use virtual environment but Conda should be very similar.

### Create virtual environment with Python-3
```bash
python3 -m venv [PATH]
```
### Activate the enviorment
```
source [PATH]/bin/activate
```

### Clone the code
```bash 
git clone https://github.com/rasoolims/ImageTranslate
cd ImageTranslate
```

### Install requirements
In my experiments, I used cuda 10.1 and cudnn 7. To replicate the results, please use the mentioned versions. If things do not work as expected, please use the Docker installation.

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Note that in some machines, [__apex__](https://github.com/NVIDIA/apex) (library for using FP16 in Nvidia) does not install properly. You should try to install in manually throughtout apex its source.
In my case, my __nvcc__ was unrecognized by the machine and I had to update my paths. Take a look at [this](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed). Also, you need to set __CUDA_HOME__.

```bash
export CUDA_HOME=[PATH TO CUDA; e.g. /usr/local/cuda-10.1]
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension```


