# ImageTranslate

This repository contains a collection of _experimental_ neural machine translation and computer vision codes based on Pytorch. Part of this code is used for the the paper ["Wikily" Neural Machine Translation Tailored to Cross-Lingual Tasks](https://arxiv.org/abs/2104.08384). If you use the models or the code, please cite the paper with the following details:
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

# Contents
- [Installing Dependencies](#installing-dependencies)
  * [Installation using virtualenv](#installation-using-virtualenv)
  * [Installation using dockers](#installation-using-dockers)
- [Using Pretrained Translation and Captioning Models](#using-pretrained-translation-and-captioning-models)
  * [Translation](#translation)
  * [Image Captioning](#image-captioning)
- [Training a Model](#training-a-model)
  * [Train Machine Translation](#train-machine-translation)
    + [Training MASS Pretraining from Scratch](#training-mass-pretraining-from-scratch)
    + [Train Unsupervised MT](#train-unsupervised-mt)
    + [Train MT on Parallel Data](#train-mt-on-parallel-data)
    + [Training from pre-trained MASS model](#training-from-pre-trained-mass-model)
  * [Train Image Captioning](#train-image-captioning)

# Installing Dependencies 

## Installation using virtualenv
Here, I use a virtual environment but Conda should be very similar.

1. Create a virtual environment with Python-3
```bash
python3 -m venv [PATH]
```
2. Activate the environment
```
source [PATH]/bin/activate
```

3. Clone the code
```bash 
git clone https://github.com/rasoolims/ImageTranslate
cd ImageTranslate/src
```

4. Install requirements
In my experiments, I used cuda 10.1 and cudnn 7. To replicate the results, please use the mentioned versions. If things do not work as expected, please use the Docker installation.

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Note that in some machines, [__apex__](https://github.com/NVIDIA/apex) (library for using FP16 in Nvidia) does not install properly. You should try to install it manually throughout its source.
In my case, my __nvcc__ was unrecognized by the machine and I had to update my paths. Take a look at [this](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed). Also, you need to set __CUDA_HOME__.

```bash
export CUDA_HOME=[PATH TO CUDA; e.g. /usr/local/cuda-10.1]
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension
```

## Installation using dockers
__Asuuming that Docker and NVIDIA docker is installed.__, follow the following steps:

1. Download the repository and pretrained models
```
git clone https://github.com/rasoolims/ImageTranslate
cd ImageTranslate/
```

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
CUDA_VISIBLE_DEVICES=0 python3 -u translate.py --tok ar/tok/ \
--output [output English file] --input [input Arabic file] \
--src en --target ar --beam 4 --model ar/model --capacity 600 --batch 4000     
```

Here I assumed that I want to translate from Arabic to English. Language abbreviations are ar,en, kk, gu, and ro.
Note that you can change the GPU id (e.g. CUDA_VISIBLE_DEVICES=1), and change batch and capacity for the best fit of your machine. 

## Image Captioning
There are two Wikily models for image captioning for Arabic in which both have similar qualities. One is multi-tasked with translation and English captioning, and the other only with translation. The zipped model folders are available at [this link](https://drive.google.com/drive/folders/1lH3sp3OFerHQ60gsjOPHHBu-wCjGKGjC?usp=sharing). There are two model folders there. You can try either of them.

```bash
unzip caption.py
CUDA_VISIBLE_DEVICES=0 python -u caption.py --input [image-folder] \
--output [output-file] --target ar --tok caption/tok \
--model  caption/caption+mt/   --fp16
```
__[image-folder]__ is a folder containing a collection of __jpg__ of __jpeg__ files. Note that you have to specify the target language __ar__ to do proper captioning. The [output-file] will be a tab-separated file with image path as first and caption as second columns.


# Training a Model
Currently, this code only works with one GPU. For working with multiple GPUs, there are some known issues. Please do not use more GPUs until further notice.

## Train Machine Translation
Throughout this guideline, I use the small files in the _sample_ folder. Here the Persian and English files are parallel but the Arabic text is not!

__WARNING__: Depending on data, the best parameters might significantly differ. It is good to try some parameter tuning for finding the best setting.

### Training MASS Pretraining from Scratch
__1. Collect raw text for languages:__

We first add language identifiers to each individual text in order to distinguish different languages.
```bash
python scripts/add_lang_id.py sample/ar.txt ar sample/ar.id.txt
python scripts/add_lang_id.py sample/fa.txt fa sample/fa.id.txt
python scripts/add_lang_id.py sample/en.txt en sample/en.id.txt
```
Then, we concatenate the three files. Note that this could be any number of files or languages more than or equal to two.
```bash
cat sample/*.id.txt > sample/all.id.txt
```

__2. Train a tokenizer on the concatenation of all raw text:__

Now we are ready to train a tokenizer:
```bash
python train_tokenizer.py --data sample/all.id.txt --vocab_size [vocab-size] --model sample/tok
```
The vocab size could be any value but in our paper, we used 60000 since the data was big. For this sample file, try 1000.

__3. Create binarized files on each raw text:__

Each file should consist of content in one language only.
```bash
python create_mt_batches.py --tok sample/tok/ --src sample/en.txt --src-lang en --output sample/en.mass
python create_mt_batches.py --tok sample/tok/ --src sample/ar.txt --src-lang ar --output sample/ar.mass
python create_mt_batches.py --tok sample/tok/ --src sample/fa.txt --src-lang fa --output sample/fa.mass
```
4. Train MASS on binarized files

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u  train_image_mt.py --tok sample/tok/ \
--model sample/mass_model --mass_train sample/en.mass.0,sample/fa.mass.0,sample/ar.mass.0 \
--capacity 2800 --batch 16000 --step 300000 --warmup 100000 --acc 8  --fp16 
```

You can kill the process whenever you want. This process takes a long time to train on large data files. You can use __scree__ and put the standard outputs into a log file in order to run it in the background mode.

The latest model will be saved in [model-path].latest; e.g. sample/mass_model.latest.

### Train Unsupervised MT

Following steps in [the previous section](#Training-MASS-from-Scratch), load the MASS model and train iterative back-translation. 

Assuming that here we are interested in English-Persian, can run the following command to run it.

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u train_image_mt.py --tok sample/tok/ \
--model sample/mass_model \
--mass_train sample/en.mass.0,sample/fa.mass.0,sample/ar.mass.0 \
--capacity 2800 --batch 16000 --step 300000 --warmup 100000   --fp16
```

Similar to the previous step, this step also takes a long time on large datasets. You can change the ``--bt-beam `` option for beam size in back-translation but note that this might affect memory, and you should decrease ``--batch`` and ``--capacity`` options.

If you are interested in observing how the model makes progress in BLEU score on some held-out data, you could also build binaries for development translation data and use the ``--dev_mt`` option for giving the binary files separated by ``,``. Take a look at [the next section](#train-mt-on-parallel-data) for more details on how to built binary files for translation data.

### Train MT on Parallel Data
Parallel data could be gold-standard or mined. You should load pre-trained MASS models for the best performance.

### Training from pre-trained MASS model
This is essentially ...


## Train Image Captioning
