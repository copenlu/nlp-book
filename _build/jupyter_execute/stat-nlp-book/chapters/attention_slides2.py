#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n<style>\n.rendered_html td {\n    font-size: xx-large;\n    text-align: left; !important\n}\n.rendered_html th {\n    font-size: xx-large;\n    text-align: left; !important\n}\n</style>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', 'import sys\nsys.path.append("..")\nimport statnlpbook.util as util\nimport matplotlib\nmatplotlib.rcParams[\'figure.figsize\'] = (10.0, 6.0)\n')


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
# \newcommand{\bar}{\,|\,}
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

# In[3]:


get_ipython().run_line_magic('load_ext', 'tikzmagic')


# # Attention

# ## Schedule
# 
# + Background: recognising textual entailment (5 min.)
# 
# + Math: attention (10 min.)
# 
# + Math: self-attention (10 min.)
# 
# + Background: BERT (15 min.)
# 
# + Background: mBERT (5 min.)
# 
# + Quiz: mBERT (5 min.)
# 

# ## Example Task: Recognising Textual Entailment / Natural Language Inference
# 
# Determining the logical relationship between two sentences.
# 
# - (Pairwise) sequence classification task
# - Requires commonsense and world knowledge
# - Requires general natural language understanding
# - Requires fine-grained reasoning

# ### Recognising Textual Entailment (RTE)
# 
# [Dagan et al., 2005](http://u.cs.biu.ac.il/~nlp/downloads/publications/RTEChallenge.pdf)
# 
# - Text (premise) T
# - Hypothesis H
# 
# T entails H if, typically, a human reading T would infer that H is most likely true.

# > **T:** “Google files for its long awaited IPO.”
# > **H:** “Google goes public.”

# Positive ($\Rightarrow$, entails)

# ### Stanford Natural Language Inference (SNLI) corpus
# 
# [Bowman et al., 2015](https://www.aclweb.org/anthology/D15-1075.pdf): crowdsourced NLI using image captions.
# 
# 570K sentence pairs, two orders of magnitude larger than other NLI resources (1K-10K examples).
# 
# **T**: A wedding party taking pictures
# - **H:** There is a funeral					: **<span class=red>Contradiction</span>** ($\Rightarrow\neg$)
# - **H:** They are outside					    : **<span class=blue>Neutral</span>** (?)
# - **H:** Someone got married				    : **<span class=green>Entailment</span>** ($\Rightarrow$)
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/Wedding_photographer_at_work.jpg" width=1500/> 

# ### Typical Approach: Independent sentence encoding
# 
# [Bowman et al, 2015](https://www.aclweb.org/anthology/D15-1075.pdf): same LSTM encodes premise and hypothesis.
# 
# <img src="dl-applications-figures/rte.svg" width=1500/> 

# Last output vector as sentence representation.
# 
# <img src="dl-applications-figures/rte_encoding.svg" width=1500/>

# #### Problem
# 
# > You can’t cram the meaning of a whole
# %&!\$# sentence into a single \$&!#* vector!
# >
# > -- <cite>Raymond J. Mooney</cite>

# ## Idea
# 
# + Traditional (non-neural) models often perform **alignment** between sequences
# 
# <img  src="./dl-applications-figures/snow.png"/>
# 
# + Can we learn something similar with our neural encoder model?

# ## Attention mechanism
# 
# + Original motivation: Bahdanau et al. 2014, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
# 
# #### Idea
# 
# + Each encoder timestep gives us a **contextual representation** of the corresponding input token
# + A **weighted combination** of those is a differentiable function
# + Computing such a combination for each decoder timestep can give us a **soft alignment**

# ### Word-by-word Attention [<span class=blue>Bahdanau et al. 2015</span>, <span class=blue>Hermann et al. 2015</span>, <span class=blue>Rush et al. 2015</span>]
# 
# <img src="dl-applications-figures/word_attention_encoding.svg" width=800/>

# ### What is happening here?
# 
# **Attention model** takes as input:
# 
# + The matrix $\mathbf{Y} \in\mathbb{R}^{k\times L}$, consisting of all output vectors $(\mathbf{h}_1, \ldots, \mathbf{h}_n)$ of the premise
#    + where $k$ is the hidden size and $L$ are the number of words of the premise

# **Attention model** then:
# 
# + Processes the hypothesis one word at a time
# + While processing it, generates attention weight-vectors $\alpha_t$ overall all output vectors of the premise for every word in the hypothesis

# **Attention model** produces:
# 
# + A probability distribution $\alpha$ over hypothesis and premise output vectors using a softmax
# + A context representation $\mathbf{r}$ by weighting output vectors with the attention $\alpha$, which can be used together with $\mathbf{h}_N$ for prediction

# ### Attention matrix
# 
# <img  src="./dl-applications-figures/snow.png"/>

# More formally:
# 
# <div class=small>
# \begin{align}
#   \mathbf{M}_t &= \tanh(\mathbf{W}^y\mathbf{Y}+(\mathbf{W}^h\mathbf{h}_t+\mathbf{W}^r\mathbf{r}_{t-1})\mathbf{1}^T_L) & \mathbf{M}_t &\in\mathbb{R}^{k\times L}\\
#   \alpha_t &= \text{softmax}(\mathbf{w}^T\mathbf{M}_t)&\alpha_t&\in\mathbb{R}^L\\
#   \mathbf{r}_t &= \mathbf{Y}\alpha^T_t + \tanh(\mathbf{W}^t\mathbf{r}_{t-1})&\mathbf{r}_t&\in\mathbb{R}^k
# \end{align}
# </div>
# 
# where $\mathbf{W}^y$, $\mathbf{W}^h$, $\mathbf{W}^r \in\mathbb{R}^{k\times k}$ are trained projection matrices, $\alpha_t$ is the attention vector, and $\mathbf{r}_t$ is the weighted representation of the premise

# Final pairwise sentence representation:
# 
# <div class=small>
# \begin{align}
#   \mathbf{h}^{*} &= \text{tanh} (\mathbf{W}^p\mathbf{r} + \mathbf{W}^x\mathbf{h}_N)
# \end{align}
# </div>
# 
# Non-linear combination of the attention-weighted representation $\mathbf{r}_t$ and the last output vector $\mathbf{h}_N$, where $\mathbf{h}^{*} \in\mathbb{R}^{k}$ 

# ### An important caveat
# 
# + The attention mechanism was motivated by the idea of aligning inputs & outputs
# + Attention matrices often correspond to human intuitions about alignment
# + But ***producing a sensible alignment is not a training objective!***
# 
# In other words:
# 
# + Do not expect that attention weights will *necessarily* correspond to sensible alignments!

# More recent development:
# 
# ### Transformer models
# 
# + Described in Vaswani et al. (2017) paper famously titled [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
# + Gets rid of RNNs, uses attention calculations everywhere (also called **self-attention**)
# + Used in most current state-of-the-art NMT models
# 

# ## Self-attention
# 
# Forget about Bi-LSTMs, because "Attention is All You Need" (or so Vaswani et al. would have us believe)*
# 
# *Editorial remark: this isn't actually true. LSTMs and CNNs still perform better for many tasks. But let's roll with this for now...

# All encoder tokens attend to each other:
# 
# <center>
#     <img src="http://jalammar.github.io/images/t/transformer_self-attention_visualization.png" width=30%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="http://jalammar.github.io/illustrated-transformer/">The Illustrated Transformer</a>)
# </div>

# ### Scaled Dot-Product Attention
# 
# Use hidden representation $\mathbf{h}_i$ to create three vectors:
# query vector $\color{purple}{\mathbf{q}_i}=W^q\mathbf{h}_i$,
# key vector $\color{orange}{\mathbf{k}_i}=W^k\mathbf{h}_i$,
# value vector $\color{blue}{\mathbf{v}_i}=W^v\mathbf{h}_i$.
# 
# $$
# \mathbf{\alpha}_{i,j} = \text{softmax}\left(
# \frac{\color{purple}{\mathbf{q}_i}^\intercal
# \color{orange}{\mathbf{k}_j}}
# {\sqrt{d_{\mathbf{h}}}}
# \right) \\
# \mathbf{h}_i^\prime = \sum_{j=1}^n \mathbf{\alpha}_{i,j} \color{blue}{\mathbf{v}_j}
# $$
# 
# The three key vectors are all trained.

# In matrix form:
# 
# $$
# \text{softmax}\left(
# \frac{\color{purple}{Q}
# \color{orange}{K}^\intercal}
# {\sqrt{d_{\mathbf{h}}}}
# \right) \color{blue}{V}
# $$

# ### Multi-head self-attention
# 
# <center>
#     <img src="mt_figures/multi_head_self_att.png" width=30%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1706.03762.pdf">Vaswani et al., 2017</a>)
# </div>

# ### Transformer layer
# 
# <center>
#     <img src="mt_figures/transformer_layer.png" width=30%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1706.03762.pdf">Vaswani et al., 2017</a>)
# </div>

# ### Transformer
# 
# <center>
#     <img src="mt_figures/transformer.png" width=30%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1706.03762.pdf">Vaswani et al., 2017</a>)
# </div>

# ### Long-distance dependencies
# 
# <center>
#     <img src="mt_figures/ldd.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1706.03762.pdf">Vaswani et al., 2017</a>)
# </div>

# Unlike RNNs, no inherent locality bias!

# ### Transformers for decoding
# 
# Attends to encoded input *and* to partial output.
# 
# <center>
#     <img src="http://jalammar.github.io/images/xlnet/transformer-encoder-decoder.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="http://jalammar.github.io/illustrated-gpt2/">The Illustrated GPT-2</a>)
# </div>

# Can only attend to already-generated tokens.
# 
# <center>
#     <img src="http://jalammar.github.io/images/gpt2/self-attention-and-masked-self-attention.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="http://jalammar.github.io/illustrated-gpt2/">The Illustrated GPT-2</a>)
# </div>

# The encoder transformer is sometimes called "bidirectional transformer".

# ## BERT
# 
# [Devlin et al., 2019](https://www.aclweb.org/anthology/N19-1423.pdf):
# **B**idirectional **E**ncoder **R**epresentations from **T**ransformers.
# 
# <center>
#     <img src="https://miro.medium.com/max/300/0*2XpE-VjhhLGkFDYg.jpg" width=40%/>
# </center>

# ### BERT architecture
# 
# Transformer with $L$ layers of dimension $H$, and $A$ self-attention heads.
# 
# * BERT$_\mathrm{BASE}$: $L=12, H=768, A=12$
# * BERT$_\mathrm{LARGE}$: $L=24, H=1024, A=16$
# 
# Other pre-trained checkpoints: https://github.com/google-research/bert

# Trained on 16GB of text from Wikipedia + BookCorpus.
# 
# * BERT$_\mathrm{BASE}$: 4 TPUs for 4 days
# * BERT$_\mathrm{LARGE}$: 16 TPUs for 4 days

# ### Training objective (1): masked language model
# 
# Predict masked words given context on both sides:
# 
# <center>
#     <img src="http://jalammar.github.io/images/BERT-language-modeling-masked-lm.png" width=50%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="http://jalammar.github.io/illustrated-bert/">The Illustrated BERT</a>)
# </div>

# <center>
# <a href="slides/mlm.pdf"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Sesame_Street_logo.svg/500px-Sesame_Street_logo.svg.png"></a>
# </center>

# ### Training objective (2): next sentence prediction
# 
# **Conditional encoding** of both sentences:
# 
# <center>
#     <img src="http://jalammar.github.io/images/bert-next-sentence-prediction.png" width=60%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="http://jalammar.github.io/illustrated-bert/">The Illustrated BERT</a>)
# </div>

# ### How is that different from ELMo and GPT-$n$?
# 
# <center>
#     <img src="mt_figures/bert_gpt_elmo.png" width=100%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/N19-1423.pdf">Devlin et al., 2019</a>)
# </div>

# ### Not words, but WordPieces
# 
# <center>
#     <img src="https://vamvas.ch/assets/bert-for-ner/tokenizer.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://vamvas.ch/bert-for-ner">BERT for NER</a>)
# </div>

# * 30,000 WordPiece vocabulary
# * No unknown words!

# ### Using BERT
# 
# <center>
#     <img src="http://jalammar.github.io/images/bert-tasks.png" width=60%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/N19-1423.pdf">Devlin et al., 2019</a>)
# </div>

# Feature extraction (❄️) vs. fine-tuning (🔥)
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/8659bf379ca8756755125a487c43cfe8611ce842/1-Table1-1.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/W19-4302.pdf">Peters et al. 2019</a>)
# </div>

# Don't stop pretraining!
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/e816f788767eec6a8ef0ea9eddd0e902435d4271/1-Figure1-1.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/2020.acl-main.740.pdf">Gururangan et al. 2020</a>)
# </div>

# ### Which layer to use?
# 
# <center>
#     <img src="http://jalammar.github.io/images/bert-feature-extraction-contextualized-embeddings.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="http://jalammar.github.io/illustrated-bert/">The Illustrated BERT</a>)
# </div>

# ### RoBERTa
# 
# [Liu et al., 2019](https://arxiv.org/pdf/1907.11692.pdf): bigger is better.
# 
# BERT with additionally
# 
# - CC-News (76GB)
# - OpenWebText (38GB)
# - Stories (31GB)
# 
# and **no** next-sentence-prediction task (only masked LM).
# 

# Training: 1024 GPUs for one day.

# ## Multilingual BERT
# 
# * One model pre-trained on 104 languages with the largest Wikipedias
# * 110k *shared* WordPiece vocabulary
# * Same architecture as BERT$_\mathrm{BASE}$: $L=12, H=768, A=12$
# * Same training objectives, **no cross-lingual signal**
# 
# https://github.com/google-research/bert/blob/master/multilingual.md

# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/5d8beeca1a2e3263b2796e74e2f57ffb579737ee/3-Figure1-1.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1911.03310.pdf">Libovický et al., 2019</a>)
# </div>

# ### Other multilingual transformers
# 
# + XLM ([Lample and Conneau, 2019](https://arxiv.org/pdf/1901.07291.pdf)) additionally uses an MT objective
# + DistilmBERT ([Sanh et al., 2020](https://arxiv.org/pdf/1910.01108.pdf)) is a lighter version of mBERT
# + Many monolingual BERTs for languages other than English
# ([CamemBERT](https://arxiv.org/pdf/1911.03894.pdf),
# [BERTje](https://arxiv.org/pdf/1912.09582),
# [Nordic BERT](https://github.com/botxo/nordic_bert)...)

# ### Zero-shot cross-lingual transfer
# 
# 1. Pre-train (or download) mBERT
# 2. Fine-tune on a task in one language (e.g., English)
# 3. Test on the same task in another language
# 

# mBERT is unreasonably effective at cross-lingual transfer!
# 
# NER F1:
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/809cc93921e4698bde891475254ad6dfba33d03b/2-Table1-1.png" width=80%/>
# </center>
# 
# POS accuracy:
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/809cc93921e4698bde891475254ad6dfba33d03b/2-Table2-1.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/P19-1493.pdf">Pires et al., 2019</a>)
# </div>

# Why? (poll)

# See also [K et al., 2020](https://arxiv.org/pdf/1912.07840.pdf);
# [Wu and Dredze., 2019](https://www.aclweb.org/anthology/D19-1077.pdf).
# 

# ## Summary
# 
# + The **attention mechanism** alleviates the encoding bottleneck in encoder-decoder architectures
# 
# + Attention can even replace (bi)-LSTMs, giving **self-attention**
# 
# + **Transformers** rely on self-attention for encoding and decoding
# 
# + **BERT**, GPT-$n$ and other transformers are powerful pre-trained contextualized representations
# 
# + **Multilingual** pre-trained transformers enable zero-shot cross-lingual transfer
# 

# ## Further reading
# 
# * Attention:
#   + Lilian Weng's blog post [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
# 
# 
# * Transformers
#   + Jay Alammar's blog posts:
#     + [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
#     + [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
#     + [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/)
# 
# 
# 

# In[ ]:





# In[ ]:




