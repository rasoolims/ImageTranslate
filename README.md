# "Wikily" Neural Machine Translation Tailored to Cross-Lingual Tasks

This repository contains a collection of _experimental_ neural machine translation and computer vision codes based on Pytorch. Part of this code is used for the the paper ["Wikily" Neural Machine Translation Tailored to Cross-Lingual Tasks](https://arxiv.org/abs/2104.08384). If you use the models or the code, please cite the paper with the following details:
```bibtex
@misc{rasooli2021wikily,
      title={"Wikily" Neural Machine Translation Tailored to Cross-Lingual Tasks}, 
      author={Mohammad Sadegh Rasooli and Chris Callison-Burch and Derry Tanti Wijaya},
      year={2021},
      eprint={2104.08384},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
We should emphasize that a big portion of this code is still in experimental mode, and any functionality other than what we describe in this README file might not work properly.

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

Note that there is a ``--verbose`` option where it puts the input and output lines separated by ``|||``. This is useful especially if you want to use it for back-translation (to make sure that sentence alignments are completely guaranteed), or for annotation projection in which you might need it for word alignment.

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
--capacity 2800 --batch 16000 --step 300000 --fstep 0 --warmup 100000 --acc 8  --fp16 
```

You can kill the process whenever you want. This process takes a long time to train on large data files. You can use __screen__ and put the standard outputs into a log file in order to run it in the background mode. __Usually training with more steps with bigger batches on larger GPU memories reaches a better model quality.__

The latest model will be saved in [model-path].latest; e.g. sample/mass_model.latest. __If you are using the sample data, you can kill the process after seeing a few epochs being done! We just need to make sure the code works.__

### Train Unsupervised MT

Following steps in [the previous section](#Training-MASS-from-Scratch), load the MASS model and train iterative back-translation. 

Assuming that here we are interested in English-Persian, can run the following command to run it.

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u train_image_mt.py --tok sample/tok/ \
--model sample/umt_model \
--pretrained sample/mass_model.latest \
--mass_train sample/en.mass.0,sample/fa.mass.0 \
--capacity 2800 --batch 16000 --step 0 --fstep 300000 --warmup 100000   --fp16 \
--langs fa,en
```

Similar to the previous step, this step also takes a long time on large datasets. You can change the ``--bt-beam `` option for beam size in back-translation but note that this might affect memory, and you should decrease ``--batch`` and ``--capacity`` options. __In principle, you could merge this step with the previous step by choosing non-zero ``--step`` and ``--fstep--`` options, but it is preferred to do this seperately in order to reuse the pre-trained MASS model in other places.__

If you are interested in observing how the model makes progress in BLEU score on some held-out data, you could also build binaries for development translation data and use the ``--dev_mt`` option for giving the binary files separated by ``,``. Take a look at [the next section](#train-mt-on-parallel-data) for more details on how to built binary files for translation data.

### Train MT on Parallel Data
Parallel data could be gold-standard or mined. You should load pre-trained MASS models for the best performance.

__1. Create binary files for training and dev dataset:__ For simplicity, we use the Persian and English text files as both training and development datasets by using their last 100 sentences as development data. 
```bash
head -9900 sample/fa.txt > sample/train.fa
head -9900 sample/en.txt > sample/train.en
tail -100 sample/en.txt > sample/dev.en
tail -100 sample/fa.txt > sample/dev.fa

python create_mt_batches.py --tok sample/tok/ --src sample/train.fa \
 --dst sample/train.en --src-lang fa --dst-lang en  \
 --output sample/fa2en.train.mt
  
python create_mt_batches.py --tok sample/tok/ --src sample/dev.fa \
 --dst sample/train.en --src-lang fa --dst-lang en  \
 --output sample/fa2en.dev.mt

```

If you create translation data in multiple direction, you can train multilingual translation for which we learn translation from multiple directions. Multiple data files can be separated by ``,`` in the arguments both for ``--train_mt`` and ``--dev_mt`` options.

__2. Train by loading the pretrained MASS model:__
```
 CUDA_VISIBLE_DEVICES=0 python3 -u train_image_mt.py --tok  sample/tok/ \
 --model sample/mt_model  --train_mt sample/fa2en.train.mt \
 --capacity 600 --batch 4000   --beam 4 --step 500000 --warmup 4000 --fstep 0 \
 --lr 0.0001  --dev_mt sample/fa2en.dev.mt \
 --dropout 0.1 --fp16 --pretrained  sample/mass_model.latest
```
Depending on how much you pretrained the MASS model, you might different BLEU scores throughout different epochs. In general, since the training data is super-small the BLEU scores are usually low (less than 1.0) on this sample data.

After you are done, you can use the model path ``sample/mt_model`` for translating text to English (similar to [the section on using the pretrained models in our paper](#translation).

```bash
CUDA_VISIBLE_DEVICES=0 python -u translate.py --tok sample/tok/ \
--model sample/mt_model --input sample/dev.fa \
--output sample/dev.output.en --src fa --target en
```
Note that there is a ``--verbose`` option where it puts the input and output lines separated by ``|||``. This is useful especially if you want to use it for back-translation (to make sure that sentence alignments are completely guaranteed), or for annotation projection in which you might need it for word alignment.


### Training from pre-trained MASS model
In case you need to use our pretrained MASS models, all of those could be downloaded from [this link](https://drive.google.com/drive/folders/18LYZz55Z7YbZCQvLPqBHRqDdMjIhBrRk?usp=sharing)

Note that each package has the following languages:
* ar: ar, fa, en
* gu: gu, hi, en (Note that Hindi (hi) is converted to Gujarati script)
* kk: kk, ru, en
* ro: ro, it, en

You could you the model along with the tokenizers inside them to fine-tune or train new models. Their tokenizer folders are inside the pre-trained translation packages ([this link](https://drive.google.com/drive/u/2/folders/10aojSCqlYCunTv9swDCgkcrVkJ6xP4xE)).


## Train Image Captioning
Assuming that you have a text file that contains a list of image file paths and their captions separated by the tab (``\t``) character, in which if an image has ``n`` captions, it will show up in ``n`` lines in the text file where each line belongs to one of its captions, you should be able to train a captioning model. Note that the development data should have a similar format. After training is done, you can caption all images in a folder (the images could also by soft links).

__1. Convert list text files into binaries__
```bash
python binarize_captions_from_list.py --file [list-file] \
--tok [tokenizer-folder] --lang [language-id] \
--output [output-binary-file]
```
Note that the tokenizer folder has the same format as what we described in the translation section.

__2. Train__: Note that the training can be multi-tasked with different captioning data from multiple languages as well as translation data. Moreover, the captioning model could be initialized by a pre-trained translation model.

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_captioning.py \
--train [caption train binary files separated by ,]  \
--dev [caption dev binary files separated by ,]  --tok [tokenizer-folder]  \
--model [model-folder] --fp16 --no-obj  --img-depth 5  \
--img_capacity 300 --max-image 20 --acc 32  --step 450000 \
--lm [Optional: pretrained translation model folder] \
--train_mt [Optional: translation train binary files separated by ,] \
--dev_mt [Optional: translation development binary files separated by ,] 
```
