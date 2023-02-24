>📋  End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models

# End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models

This repository is the official implementation of [End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models](https://arxiv.org/abs/2205.12487). 

Please use the following citation:
```
@article{yao2022end,
  title={End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models},
  author={Yao, Barry Menglong and Shah, Aditya and Sun, Lichao and Cho, Jin-Hee and Huang, Lifu},
  journal={arXiv preprint arXiv:2205.12487},
  year={2022}
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

- [MOCHEG version 1](https://doi.org/10.5281/zenodo.6653771). 

- Dataset Format and structure are explained in document/MOCHEG_dataset_statement.pdf.

- If you just want to reproduce the claim verification or explanation generation experiments, the above dataset is all you need. If you want to run the evidence retrieval experiments to get a sense of the dataset, you can also use the above dataset. However, if you want to reproduce our evidence retrieval experiments, you need to merge the current dataset with our tweets corpus, which contains only 2,916 tweets. You can obtain this merged dataset either by emailing us or running the scripts under dataset_builder/twitter. (In accordance with
[the Twitter developer terms](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases), we will only share the Twitter IDs and scripts to crawl tweets based on Twitter API. However, according to [Twitter policy](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases), they permit sharing up to 50,000 hydrated Twitter content per recipient via non-automated means)


## Pre-trained Models

You can download pretrained models here:

- [pretrained models](http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/checkpoint/) trained on MOCHEG. 

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
