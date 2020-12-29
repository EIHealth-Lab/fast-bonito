FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get update && \
    apt-get -y install git

RUN cd /workspace && \
    git clone https://github.com/EIHealth-Lab/fast-bonito.git 

RUN cd /workspace/fast-bonito && \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
