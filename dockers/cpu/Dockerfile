FROM ubuntu:18.04
LABEL maintainer="Mohammad Sadegh Rasooli"
LABEL repository="root_mt"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   wget \
                   tar \
                   ca-certificates \
                   python3 \
                   python3-pip \
                   vim && \
    rm -rf /var/lib/apt/lists

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
RUN git clone https://github.com/rasoolims/ImageTranslate

WORKDIR /root/ImageTranslate/

COPY dataset_prep /root/ImageTranslate/dataset_prep
COPY mt_pret /root/ImageTranslate/mt_pret

CMD ["/bin/bash"]