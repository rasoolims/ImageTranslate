FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
LABEL maintainer="Mohammad Sadegh Rasooli"
LABEL repository="root_mt"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip \
                   openssh-client \
                   vim && \
    rm -rf /var/lib/apt/lists

RUN apt-get update && apt-get -y install openssh-client
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install torch==1.4.0
RUN python3 -m pip install numpy==1.18.2
RUN python3 -m pip install sacrebleu
RUN python3 -m pip install future==0.18.2
RUN python3 -m pip install pillow
RUN python3 -m pip install torchvision==0.5.0
RUN python3 -m pip install ipython
RUN python3 -m pip install transformers==2.9.0
RUN python3 -m pip install pandas
RUN python3 -m pip install pyarrow
RUN python3 -m pip install fire
RUN python3 -m pip install boto3
#RUN python3 -m pip install apex

WORKDIR /root
RUN cd /root
RUN mkdir /root/development
RUN mkdir /root/development/global_voices
RUN mkdir /root/development/ted_talks
RUN mkdir /root/models
ENV DATA_FOLDER=/root/development
ENV SAVE_PATH=/root/models
ENV LANG C.UTF-8
RUN echo 'alias python="python3"' >> ~/.bashrc

RUN cd /root
RUN git clone https://github.com/AdityaKashyap423/ImageTranslate

WORKDIR /root/ImageTranslate/
COPY dataset_prep /root/ImageTranslate/dataset_prep
COPY mt_pret /root/ImageTranslate/mt_pret

WORKDIR /root/
# uninstall Apex if present, twice to make absolutely sure :)
RUN python3 -m pip uninstall -y apex || :
RUN python3 -m pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /root/apex/
RUN python3 -m pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension

WORKDIR /root/ImageTranslate/

CMD ["/bin/bash"]