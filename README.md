# ImageTranslate

This repository contains a collection of _experimental_ neural machine translation and computer vision codes based on Pytorch. Part of this code is used for the the paper ["Wikily" Neural Machine Translation Tailored to Cross-Lingual Tasks](https://arxiv.org/abs/2104.08384). If you use the models or the code, please cite as following:
```text
@misc{rasooli2021wikily,
      title={"Wikily" Neural Machine Translation Tailored to Cross-Lingual Tasks}, 
      author={Mohammad Sadegh Rasooli and Chris Callison-Burch and Derry Tanti Wijaya},
      year={2021},
      eprint={2104.08384},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## Installation using virtualenv
Here, I use virtual environment but Conda should be very similar.

1. Create virtual environment with Python-3
```bash
python3 -m venv [PATH]
```
2. Activate the enviorment
```
source [PATH]/bin/activate
```

3. Clone the code
```bash 
git clone https://github.com/rasoolims/ImageTranslate
cd ImageTranslate
```

4. Install requirements
In my experiments, I used cuda 10.1 and cudnn 7. To replicate the results, please use the mentioned versions. If things do not work as expected, please use the Docker installation.

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Note that in some machines, [__apex__](https://github.com/NVIDIA/apex) (library for using FP16 in Nvidia) does not install properly. You should try to install in manually throughtout apex its source.
In my case, my __nvcc__ was unrecognized by the machine and I had to update my paths. Take a look at [this](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed). Also, you need to set __CUDA_HOME__.

```bash
export CUDA_HOME=[PATH TO CUDA; e.g. /usr/local/cuda-10.1]
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension
```

## Installation using dockers
__Asuuming that Docker and NVIDIA docker is installed.__, follow the following steps:

1. Download the repository and pretrained models
```
git clone https://github.com/rasoolims/ImageTranslate```


2. Build the docker in command line:
```bash
docker build dockers/gpu/ -t [docker-name] --no-cache
```

3. Start running the docker:

* Run this with screen since training might take a long time.
```bash
docker run --gpus all -it  [docker-name]
```


# Using Pretrained Translation and Captioning Models

## Translation
* Download the model zip files for Arabic-English, Romanian-English, Gujarati-English, and Kazakh-English from [this link](https://drive.google.com/drive/folders/10aojSCqlYCunTv9swDCgkcrVkJ6xP4xE?usp=sharing).
* Unzip the files and use the models for translation.
    
```
unzip ar.zip
CUDA_VISIBLE_DEVICES=0 python3 -u translate.py --tok ar/tok/ --output [output English file] --input [input Arabic file] --src en --target ar --beam 4 --model ar/model --capacity 600 --batch 4000     
```

Here I assumed that I want to translate from Arabic to English. Language abbreviations are ar,en, kk, gu, and ro.
Note that you can change the gpu id (e.g. CUDA_VISIBLE_DEVICES=1), and change batch and capacity for the best fit of your machine. 

## Image Captioning
There are two Wikily models for image captioning for Arabic in which both have similar qualities. One is multi-taksed with translation and English captioning, and the other only with translation. The zipped model folders are available at [this link](https://drive.google.com/drive/folders/1lH3sp3OFerHQ60gsjOPHHBu-wCjGKGjC?usp=sharing). There are two model folders there. You can try either of them.

```bash
unzip caption.py
CUDA_VISIBLE_DEVICES=2 python -u caption.py --input [image-folder] --output [output-file] --target ar --tok caption/tok   --model  caption/caption+mt/   --fp16
```
__[image-folder]__ is a folder containing a collection of __jpg__ of __jpeg__ files. Note that you have to specify the target language __ar__ to do proper captioning. The [output-file] will be a tab-separated file with image path as first and caption as second columns.