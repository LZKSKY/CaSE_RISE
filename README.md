# Learning to Ask Conversational Questions by Optimizing Levenshtein Distance

by Zhongkun Liu, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Maarten de Rijke, Ming Zhou

> @inproceedings{liu2021learning,
>   author    = {Zhongkun Liu and
>                Pengjie Ren and
>                Zhumin Chen and
>                Zhaochun Ren and
>                Maarten de Rijke and
>                Ming Zhou},
>   title     = {A Contextual Hierarchical Attention Network with Adaptive Objective
>                for Dialogue State Tracking},
>   booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, {ACL} 2021},
>   pages     = {6322--6333},
>   year      = {2021}
> }

### Paper summary

#### Task

<img src=".\task.png" alt="image-20210607201940075" style="zoom: 80%;" />

Conversational Question Simplification (CQS) aims to simplify self-contained questions (e.g., SQ4) into conversational ones (e.g., CQ4) by incorporating some conversational characteristics, e.g., anaphora and ellipsis.

#### Model

![image-20210607204645807](.\model.png)



The editing policy network is implemented by the encoder to predict combinatorial edits, and the phrasing policy network is implemented by the decoder to predict phrases.

#### Contribution

1) In this paper, we have proposed a minimum Levenshtein distance (MLD) based Reinforcement Iterative Sequence Editing (RISE) framework for Conversational Question Simplification (CQS). 

2) To train RISE, we have devised an Iterative Reinforce Training (IRT) algorithm with a novel Dynamic
Programming based Sampling (DPS) process. 

3) Extensive experiments show that RISE is more effective and robust than several state-of-the-art CQS
methods.



### Running

---

##### Requirements

​	python >= 3.6

​	pytorch >= 1.6

​	Transformers >= 3.3.0

​	Please setup https://github.com/Maluuba/nlg-eval for evaluation.



##### Download Datasets

​	1. We use [CANARD](https://sites.google.com/view/qanta/projects/canard) based on [Quac](http://quac.ai/) and [CAsT](https://github.com/daltonj/treccastweb/tree/master/2019/data/training) dataset for training and testing.  And we rename the path as '../CANARD_Release', '../QuacN', '../CAsT'.

​	2. Please download pretrain model [bert](https://huggingface.co/bert-base-uncased/tree/main)  in path ../extra/bert/ or you can replace the tokenizer_path in Preprocess/prepare_mld.py and Preprocess/prepare_cast.py

​	3. run in command line 

```bash
	sh preprocess.sh
```

​		you will obatin bert_iter_0.train.pkl, bert_iter_0.dev.pkl, bert_iter_0.test.pkl, bert_iter_0.CAsT_test.pkl in directory ../QuacN/Dataset/.



##### Training and Evaluation

```bash
	python3 run_bert_mld_rl.py
```

​	You can modify the config_n.py for parameter modification or in command line by add argument.

```bash
	python3 run_bert_mld_rl.py --train=2
```

​	When assgin argument 'train' as 2, it can generate the model results and try

```bash
	python3 run_evaluation.py
```

​	for evaluation.