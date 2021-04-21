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
```
## Using Pretrained Translation and Captioning Models

## Translation
* Downaload the model zip files for Arabic-English, Romanian-English, Gujarati-English, and Kazakh-English from [this link](https://drive.google.com/drive/folders/10aojSCqlYCunTv9swDCgkcrVkJ6xP4xE?usp=sharing).
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
__[image-folder]__ is a folder containing a collection of __jpg__ of __jpeg__ files. Note that you have to specify the target language __ar__ to do proper captioning.