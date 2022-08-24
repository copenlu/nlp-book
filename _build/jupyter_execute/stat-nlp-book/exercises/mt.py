#!/usr/bin/env python
# coding: utf-8

# # Machine Translation Exercises
# In these exercises you will develop a machine translation system that can turn modern English into Shakespeare. 
# <!-- We will use the code from the notes, but within a python package [`mt`](http://localhost:8888/edit/statnlpbook/word_mt.py). -->

# ## <font color='green'>Setup 1</font>: Load Libraries

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import sys, os
_snlp_book_dir = ".."
sys.path.append(_snlp_book_dir) 
import statnlpbook.word_mt as word_mt
# %cd .. 
import sys
sys.path.append("..")
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
from collections import defaultdict 
import statnlpbook.util as util
from statnlpbook.lm import *
from statnlpbook.util import safe_log as log
import statnlpbook.mt as mt



# <!---
# Latex Macros
# -->
# $$
# \newcommand{\Xs}{\mathcal{X}}
# \newcommand{\Ys}{\mathcal{Y}}
# \newcommand{\y}{\mathbf{y}}
# \newcommand{\balpha}{\boldsymbol{\alpha}}
# \newcommand{\bbeta}{\boldsymbol{\beta}}
# \newcommand{\aligns}{\mathbf{a}}
# \newcommand{\align}{a}
# \newcommand{\source}{\mathbf{s}}
# \newcommand{\target}{\mathbf{t}}
# \newcommand{\ssource}{s}
# \newcommand{\starget}{t}
# \newcommand{\repr}{\mathbf{f}}
# \newcommand{\repry}{\mathbf{g}}
# \newcommand{\x}{\mathbf{x}}
# \newcommand{\prob}{p}
# \newcommand{\vocab}{V}
# \newcommand{\params}{\boldsymbol{\theta}}
# \newcommand{\param}{\theta}
# \DeclareMathOperator{\perplexity}{PP}
# \DeclareMathOperator{\argmax}{argmax}
# \DeclareMathOperator{\argmin}{argmin}
# \newcommand{\train}{\mathcal{D}}
# \newcommand{\counts}[2]{\#_{#1}(#2) }
# \newcommand{\length}[1]{\text{length}(#1) }
# \newcommand{\indi}{\mathbb{I}}
# $$

# ## <font color='green'>Setup 2</font>: Download Data

# In[2]:


get_ipython().run_cell_magic('sh', '', 'cd ../data\nif [ ! -d "shakespeare" ]; then\n    git clone https://github.com/tokestermw/tensorflow-shakespeare.git shakespeare    \n    cd shakespeare\n    cat ./data/shakespeare/sparknotes/merged/*_modern.snt.aligned > modern.txt\n    cat ./data/shakespeare/sparknotes/merged/*_original.snt.aligned > original.txt\n    cd ..\nfi\nhead -n 1 shakespeare/modern.txt\nhead -n 1 shakespeare/original.txt \n')


# ## <font color='blue'>Task 1</font>: Preprocessing Aligned Corpus
# Write methods for loading and tokenizing the aligned corpus.

# In[3]:


import re

NULL = "NULL"

def tokenize(sentence):
    return []  # todo

def pre_process(sentence):
    return []  # todo


def load_shakespeare(corpus):
    with open("../data/shakespeare/%s.txt" % corpus, "r") as f:
        return  [pre_process(x.rstrip('\n')) for x in f.readlines()] 
    
modern = load_shakespeare("modern")
original = load_shakespeare("original")

MAX_LENGTH = 6

def create_wordmt_pairs(modern, original):
    alignments = []
    for i in range(len(modern)):
        if len(modern[i]) <= MAX_LENGTH and len(original[i]) <= MAX_LENGTH:
            alignments.append(([NULL] + modern[i], original[i]))
    return alignments
                
train = create_wordmt_pairs(modern, original)

for i in range(10):
    (mod, org) = train[i]
    print(" ".join(mod), "|", " ".join(org))

print("\nTotal number of aligned sentence pairs", len(train))


# ## <font color='blue'>Task 2</font>: Train IBM Model 2
# - Train an IBM Model 2 that translates modern English to Shakespeare
# - Visualize alignments of the sentence pairs before and after training using EM 
# - Do you find interesting cases?
# - What are likely words that "killed" can be translated to?
# - Test your translation system using a beam-search decoder
#   - How does the beam size change the quality of the translation?
#   - Give examples of good and bad translations

# In[4]:


# todo


# ## <font color='blue'>Task 3</font>: Better Language Model
# Try a better language model for machine translation. How does the translation quality change for the examples you found earlier?

# In[5]:


# todo


# ## <font color='blue'>Task 4</font>: Better Decoding
# How can you change the decoder to work to translate to shorter or longer target sequences than the source?

# In[6]:


# todo

