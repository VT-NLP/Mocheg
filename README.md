>📋  End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models

# My Paper Title

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
pip install -r requirements.txt
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->


## Dataset - MOCHEG

You can download dataset here:

- [MOCHEG](http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/). 



## Pre-trained Models

You can download pretrained models here:

- [pretrained models](http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/checkpoint/) trained on MOCHEG. 


<!-- ## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

<!-- ## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). -->
 

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
