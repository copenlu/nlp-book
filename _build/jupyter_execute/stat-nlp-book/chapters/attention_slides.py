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
# + Background: neural MT (5 min.)
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

# ## Encoder-decoder seq2seq
# 
# &nbsp;
# 
# <center>
#     <img src="mt_figures/encdec_rnn2.svg" width="70%" />
# </center>

# ## More problems with our approach
# 
# &nbsp;
# 
# <center>
#     <img src="mt_figures/encdec_rnn3.svg" width="70%" />
# </center>

#     You can't cram the meaning of a whole %&!$ing sentence into a single $&!*ing vector!
# 
# — Ray Mooney
# 

# ## Idea
# 
# + Traditional (non-neural) MT models often perform **alignment** between sequences
# 
# <center style="padding: 1em 0;">
#     <img src="mt_figures/align.svg" width="20%" />
# </center>
# 
# + Can we learn something similar with our encoder–decoder model?

# ## Attention mechanism
# 
# + Original motivation: Bahdanau et al. 2014, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
# 
# #### Idea
# 
# + Each encoder timestep gives us a **contextual representation** of the corresponding input token
# + A **weighted combination** of those is a differentiable function
# + Computing such a combination for each decoder timestep can give us a **soft alignment**

# <center>
#     <img src="mt_figures/encdec_att.svg" width="40%" />
# </center>

# ### What is happening here?
# 
# **Attention model** takes as input:
# 
# + Hidden state of the decoder $\mathbf{s}_t^{\textrm{dec}}$
# + All encoder hidden states $(\mathbf{h}_1^{\textrm{enc}}, \ldots, \mathbf{h}_n^{\textrm{enc}})$
# 

# **Attention model** produces:
# 
# + An attention vector $\mathbf{\alpha}_t \in \mathbb{R}^n$ (where $n$ is the length of the source sequence)
# + $\mathbf{\alpha}_t$ is computed as a softmax distribution:
# 
# $$
# \mathbf{\alpha}_{t,j} = \text{softmax}\left(f_{\mathrm{att}}(\mathbf{s}_{t-1}^{\textrm{dec}}, \mathbf{h}_j^{\textrm{enc}})\right)
# $$

# ### How do we compute $f_\mathrm{att}$?
# 
# Usually with a very simple feedforward neural network.
# 
# For example:
# 
# $$
# f_{\mathrm{att}}(\mathbf{s}_{t-1}^{\textrm{dec}}, \mathbf{h}_j^{\textrm{enc}}) =
# \tanh
# \left(
# \mathbf{W}^s \mathbf{s}_{t-1}^{\textrm{dec}} +
# \mathbf{W}^h \mathbf{h}_j^{\textrm{enc}}
# \right)
# $$
# 
# This is called **additive** attention.

# Another alternative:
# 
# $$
# f_{\mathrm{att}}(\mathbf{s}_{t-1}^{\textrm{dec}}, \mathbf{h}_j^{\textrm{enc}}) =
# \frac{\left(\mathbf{s}_{t-1}^{\textrm{dec}}\right)^\intercal
# \mathbf{W} \mathbf{h}_j^{\textrm{enc}}}
# {\sqrt{d_{\mathbf{h}^{\textrm{enc}}}}}
# $$
# 
# This is called **scaled dot-product** attention.
# 
# (But many alternatives have been proposed!)

# ### What do we do with $\mathbf{\alpha}_t$?
# 
# Computing a **context vector:**
# 
# $$
# \mathbf{c}_t = \sum_{i=1}^n \mathbf{\alpha}_{t,i} \mathbf{h}_i^\mathrm{enc}
# $$
# 
# This is the weighted combination of the input representations!

# 
# Include this context vector in the calculation of decoder's hidden state:
# 
# $$
# \mathbf{s}_t^{\textrm{dec}} = f\left(\mathbf{s}_{t-1}^{\textrm{dec}}, \mathbf{y}_{t-1}^\textrm{dec}, \mathbf{c}_t\right)
# $$

# <center>
#     <img src="mt_figures/encdec_att.svg" width="22%" />
# </center>
# 
# > Intuitively, this implements a mechanism of attention in the decoder.  The decoder **decides parts of the source sentence to pay attention to.**  By letting the decoder have an attention mechanism, we relieve the encoder from the burden of having to encode all information in the source sentence into a fixed-length vector.
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1409.0473.pdf">Bahdanau et al., 2014</a>)
# </div>

# We can visualize what this model learns in an
# 
# ## Attention matrix
# 
# &nbsp;
# 
# $\rightarrow$ Simply concatenate all $\alpha_t$ for $1 \leq t \leq m$

# <center>
#     <img src="mt_figures/att_matrix.png" width="40%" />
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1409.0473.pdf">Bahdanau et al., 2014</a>)
# </div>

# ### An important caveat
# 
# + The attention mechanism was motivated by the idea of aligning inputs & outputs
# + Attention matrices often correspond to human intuitions about alignment
# + But ***producing a sensible alignment is not a training objective!***
# 
# In other words:
# 
# + Do not expect that attention weights will *necessarily* correspond to sensible alignments!

# ## Another way to think about attention
# 
# 1. $\color{purple}{\mathbf{s}_{t-1}^{\textrm{dec}}}$ is the **query**
# 2. Retrieve the best $\mathbf{h}_j^{\textrm{enc}}$ by taking
# $\color{orange}{\mathbf{W} \mathbf{h}_j^{\textrm{enc}}}$ as the **key**
# 3. Softly select a $\color{blue}{\mathbf{h}_j^{\textrm{enc}}}$ as **value**
# 
# $$
# \mathbf{\alpha}_{t,j} = \text{softmax}\left(
# \frac{\left(\color{purple}{\mathbf{s}_{t-1}^{\textrm{dec}}}\right)^\intercal
# \color{orange}{\mathbf{W} \mathbf{h}_j^{\textrm{enc}}}}
# {\sqrt{d_{\mathbf{h}^{\textrm{enc}}}}}
# \right) \\
# \mathbf{c}_t = \sum_{i=1}^n \mathbf{\alpha}_{t,i} \color{blue}{\mathbf{h}_i^\mathrm{enc}} \\
# \mathbf{s}_t^{\textrm{dec}} = f\left(\mathbf{s}_{t-1}^{\textrm{dec}}, \mathbf{y}_{t-1}^\textrm{dec}, \mathbf{c}_t\right)
# $$

# Used during **decoding** to attend to $\mathbf{h}^{\textrm{enc}}$, encoded by Bi-LSTM.
# 

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
# Forget about Bi-LSTMs, because Attention is All You Need even for **encoding**!

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
# Use $\mathbf{h}_i$ to create three vectors:
# $\color{purple}{\mathbf{q}_i}=W^q\mathbf{h}_i,
# \color{orange}{\mathbf{k}_i}=W^k\mathbf{h}_i,
# \color{blue}{\mathbf{v}_i}=W^v\mathbf{h}_i$.
# 
# $$
# \mathbf{\alpha}_{i,j} = \text{softmax}\left(
# \frac{\color{purple}{\mathbf{q}_i}^\intercal
# \color{orange}{\mathbf{k}_j}}
# {\sqrt{d_{\mathbf{h}}}}
# \right) \\
# \mathbf{h}_i^\prime = \sum_{j=1}^n \mathbf{\alpha}_{i,j} \color{blue}{\mathbf{v}_j}
# $$

# In matrix form:
# 
# $$
# \text{softmax}\left(
# \frac{\color{purple}{Q}
# \color{orange}{K}^\intercal}
# {\sqrt{d_{\mathbf{h}}}}
# \right) \color{blue}{V}
# $$

# <center>
#     <img src="mt_figures/self_att.png" width=25%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1706.03762.pdf">Vaswani et al., 2017</a>)
# </div>

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




