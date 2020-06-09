# Reformer

# Requirements
* Install pytorch, huggingface, Apex, numpy, etc. (Virtual environment is preferred)
* Create txt files where is starts with <cls> and each sentences ends with </s>. Each document is in one line
* Hold out a very small part of data as development file (maybe 100 documents).


* Steps
1. Train a tokenizer
2. Tokenize and binarize the training and development file
3. Train