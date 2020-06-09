# Reformer

## Requirements
* Install pytorch, huggingface, Apex, numpy, etc. (Virtual environment is preferred)
* Create txt files where is starts with <cls> and each sentences ends with </s>. Each document is in one line
* Hold out a very small part of data as development file (maybe 100 documents).


## Steps
1. Create folders
```bash
mkdir [tokenizer-folder]
mkdir [train-binary-folder]
mkdir [dev-binary-folder]
mkdir [model-folder]

```

2. Train a tokenizer
```bash
python3 -u train_tokenizer.py --data [train-txt-file] --vocab_size [size] --model [tokenizer-folder]
```
3. Tokenize and binarize the training and development file
```bash
python3 -u create_batches.py --data [train-txt-file]  --cache  [train-binary-folder] --tok [tokenizer-folder] --len 4096
```

```bash
python3 -u create_batches.py --data [dev-txt-file]  --cache  [dev-binary-folder] --tok [tokenizer-folder] --len 4096
```

-- Note that this might take a long time. Try to use __screen__ or __nohup__.

4. Train
```bash
CUDA_VISIBLE_DEVICES=[LIST-OF-DEVICES] python3 -u train_reformer_lm.py --train_cache [train-binary-folder] --valid_cache  [dev-binary-folder] --tok [tokenizer-folder] --model [model-folder] --batch [batch-size] --lr [learning-rate] --warmpu [warmup-steps] --step [training-steps]
```

-- Note that this will take a long time. Try to use __screen__ or __nohup__.

* ````[LIST-OF-DEVICES]````: Separated by comma: e.g. 1 or 0,1 or 0,1,2,3,4
* ````[batch-size]````: 64 for one gpu, 128 for 2, ..., 512 for 8 gpus
* ````[learning-rate]````, ````[training-steps]```` and  ````[warmup-steps]````: depends on batch size. A good rule of thumb is ````--lr 0.003 --steps 125000 --warmup 12500````
