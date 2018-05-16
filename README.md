# NQG
This repository contains code for the  paper "[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/abs/1704.01792)"

## About this code

The experiments in the paper were done with an in-house deep learning tool. Therefore, we re-implement this with PyTorch as a reference.

This code only implements the setting `NQG+` in the paper.
Within 1 hour's training on Tesla P100, the `NQG+` model achieves 12.78 BLEU-4 score on the dev set.

If you find this code useful in your research, please consider citing:

    @article{zhou2017neural,
      title={Neural Question Generation from Text: A Preliminary Study},
      author={Zhou, Qingyu and Yang, Nan and Wei, Furu and Tan, Chuanqi and Bao, Hangbo and Zhou, Ming},
      journal={arXiv preprint arXiv:1704.01792},
      year={2017}
    }



## How to run

### Prepare the dataset and code

Make an experiment home folder for NQG data and code:
```bash
NQG_HOME=~/workspace/nqg
mkdir -p $NQG_HOME/code
mkdir -p $NQG_HOME/data
cd $NQG_HOME/code
git clone https://github.com/magic282/NQG.git
cd $NQG_HOME/data
wget https://res.qyzhou.me/redistribute.zip
unzip redistribute.zip
```
Put the data in the folder `$NQG_HOME/code/data/giga` and organize them as:
```
nqg
├── code
│   └── NQG
│       └── seq2seq_pt
└── data
    └── redistribute
        ├── QG
        │   ├── dev
        │   ├── test
        │   ├── test_sample
        │   └── train
        └── raw
```
Then collect vocabularies:
```bash
python $NQG_HOME/code/NQG/seq2seq_pt/CollectVocab.py \
       $NQG_HOME/data/redistribute/QG/train/train.txt.source.txt \
       $NQG_HOME/data/redistribute/QG/train/train.txt.target.txt \
       $NQG_HOME/data/redistribute/QG/train/vocab.txt
python $NQG_HOME/code/NQG/seq2seq_pt/CollectVocab.py \
       $NQG_HOME/data/redistribute/QG/train/train.txt.bio \
       $NQG_HOME/data/redistribute/QG/train/bio.vocab.txt
python $NQG_HOME/code/NQG/seq2seq_pt/CollectVocab.py \
       $NQG_HOME/data/redistribute/QG/train/train.txt.pos \
       $NQG_HOME/data/redistribute/QG/train/train.txt.ner \
       $NQG_HOME/data/redistribute/QG/train/train.txt.case \
       $NQG_HOME/data/redistribute/QG/train/feat.vocab.txt
head -n 20000 $NQG_HOME/data/redistribute/QG/train/vocab.txt > $NQG_HOME/data/redistribute/QG/train/vocab.txt.20k
```

### Setup the environment
#### Package Requirements:
```
nltk scipy numpy pytorch
```
**PyTorch version**: This code requires PyTorch v0.4.0.

**Python version**: This code requires Python3.

**Warning**: Older versions of NLTK have a bug in the PorterStemmer. Therefore, a fresh installation or update of NLTK is recommended.

A Docker image is also provided.
#### Docker image
```bash
docker pull magic282/pytorch:0.4.0
```
### Run training
The file `run.sh` is an example. Modify it according to your configuration.
#### Without Docker
```bash
bash $NQG_HOME/code/NQG/seq2seq_pt/run_squad_qg.sh $NQG_HOME/data/redistribute/QG $NQG_HOME/code/NQG/seq2seq_pt
```
#### With Docker
```bash
nvidia-docker run --rm -ti -v $NQG_HOME:/workspace magic282/pytorch:0.4.0
```
Then inside the docker:
```bash
bash code/NQG/seq2seq_pt/run_squad_qg.sh /workspace/data/redistribute/QG /workspace/code/NQG/seq2seq_pt
```
