# Fast-Bonito

Recently, a new algorithm, [Bonito](https://github.com/nanoporetech/bonito), has been developed and achieved state-of-the-art accuracy. But the speed of Bonito is unsatisfying, therefore limit its application into the production. We utilized systematic methods to optimize it and implement Fast-Bonito. Fast-Bonito archives 53.8% faster than the original version on NVIDIA V100 and could be further speed up by HUAWEI Ascend 910 NPU, achieving 565% faster than the original version. The accuracy of Fast-Bonito is also slightly higher than the original Bonito. 



## Optimization

- We replaced the depthwise separable convolution, which is not friendly supported by the current inference engine, by ResNet50 Bottleneck convolution to improve the inference performance.
- We performed a multi-objective NAS, referring to MnasNet, to search a model friendly supported by our NPU inference engine.
- Some training strategies were used during the training process, data augmentation,label smoothing, knowledge distillation, etc.



## Usage

```python
python basecaller.py --model_directory MODEL_DIRECTORY \
    --reads_directory READS_DIRECTORY \
    --beamsize BEAMSIZE \
    --fastq \
    --chunksize CHUNKSIZE \
    --overlap OVERLAP \
    --batch_size BATCH_SIZE \
    --read_multiprocess READ_MULTIPROCESS \
    --decode_multiprocess DECODE_MULTIPROCESS \
    --queue_size QUEUE_SIZE \
    --output OUTPUT \
    --device DEVICE
```

model_directory: model directory contains the model;

reads_directory: directory contains fast5 files;

beamsize: default 5. The number of search points should be kept at each step. Higher numbers are less likely to discard the true labelling, but also make it slower and more memory intensive;

fastq: whether to output fastq format file;

chunksize: the size of chunk, default 6000;

overlap: the overlap between two chunks, default 300; 

batch_size: batch size of each inference, default 200;

read_multiprocess: number of process for process data before inferencing;

decode_multiprocess: number of process for decoding;

queue_size: size of queue to store the processed data;

output: output file path;

device: inference device, "npu" on D910 or "cuda" on NVIDIA.



## Requirements

- ont_fast5_api==3.0.1
- fast-ctc-decode==0.2.5
- torch==1.4.0
- tqdm==4.31.1
- numpy==1.19.2
- scipy==1.5.2
- tensorflow-gpu==1.14.0
- mappy==2.17
- ***D910***

## License
This software is part of a Huawei internal research project based on [Bonito](https://github.com/nanoporetech/bonito), we make it available to the public under the same license as Bonito in hope it could be useful.

Oxford Nanopore Technologies, Ltd. Public License Version 1.0

## Notice
Fast-bonito was developped based on [Bonito](https://github.com/nanoporetech/bonito) of version 0.2.2. In the process of basecalling, we made some modifacations to speed up Bonito. Firstly, parallel processing was used to preprocess data. Secondly, model was loaded from pb file and data were feed to model with shape of (200, 1, 1, 6000), which can inference several short reads in one batch or a long read in different batchs.  

## Citation
Zhimeng Xu, Yuting Mai, Denghui Liu, Wenjun He, Xinyuan Lin, Chi Xu, Lei Zhang, Xin Meng, Joseph Mafofo, Walid Abbas Zaher, Yi Li, Nan Qiao. Fast-Bonito: A Faster Basecaller for Nanopore Sequencing. bioRxiv 2020.10.08.318535; doi: [https://doi.org/10.1101/2020.10.08.318535](https://doi.org/10.1101/2020.10.08.318535).
