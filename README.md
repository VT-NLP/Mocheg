>📋  End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models

# End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models

This repository is the official implementation of [End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models](https://arxiv.org/abs/2205.12487). [Best Paper Honorable Mention at SIGIR 2023](https://sigir.org/sigir2023/program/best-paper-award/). 

Please use the following citation:
```
@inproceedings{10.1145/3539618.3591879,
author = {Yao, Barry Menglong and Shah, Aditya and Sun, Lichao and Cho, Jin-Hee and Huang, Lifu},
title = {End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591879},
doi = {10.1145/3539618.3591879},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2733–2743},
numpages = {11},
keywords = {explainable fact-checking, multimodal fact-checking, explanation generation, evidence retrieval, stance detection},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```
<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements:

```setup
conda create -n mocheg python=3.8.10
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->


## Dataset - MOCHEG

You can download dataset here:

- [MOCHEG version 1](https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform?usp=sf_link). 

- Dataset Format and structure are explained in document/MOCHEG_dataset_statement.pdf.

## Pre-trained Models

You can download pretrained models here:

- [pretrained models](http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/checkpoint) trained on MOCHEG. 

## Evaluation

To evaluate models on MOCHEG, run:

```eval
bash eval.sh
```
 
-----------------------------------------------------------------------------------------


## Training (Optional)

If you want to train models by yourself, run this command:

```train
bash train.sh
```
<!-- 
>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Dataset Build (Optional)

If you want to build dataset by yourself, run this command:

```build_dataset
bash build_dataseth.sh
``` 

<!-- ## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.  -->



## Credit: 

This repository was build by [Barry Menglong Yao](https://barry-yao.netlify.app/), Aditya Shah.

The data crawler scripts are based on [conll2019-snopes-crawling](https://github.com/UKPLab/conll2019-snopes-crawling).

## Contributing

>📋  Our dataset is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The associated codes are licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
