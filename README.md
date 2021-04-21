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
In my experiments, I used cuda 10.1 and cudnn 7. To replicate the results, please use the mentioned versions.

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Note that in some machines, __apex__ does not install properly. You should try to install in manually throughtout apex its source.



