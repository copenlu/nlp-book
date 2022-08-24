#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n<style>\n.rendered_html td {\n    font-size: xx-large;\n    text-align: left; !important\n}\n.rendered_html th {\n    font-size: xx-large;\n    text-align: left; !important\n}\n</style>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\nimport sys\nsys.path.append("../statnlpbook/")\n\n#util.execute_notebook(\'relation_extraction.ipynb\')\n')


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
# \newcommand{\a}{\alpha}
# \newcommand{\b}{\beta}
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


# # Question Answering

# ## Schedule
# 
# * Question answering (10 min.)
# * Machine reading comprehension (20 min.)
# * Executable semantic parsing (5 min.)
# * Exercise: IR- vs. KB-QA (15 min.)

# Question:
# 
# >Which university did Turing go to?

# Answer:
# 
# >Princeton

# [Passage](https://history-computer.com/people/alan-turing-complete-biography/):
# 
# >Alan Turing graduated from Princeton.

# Knowledge base:
# 
# https://www.wikidata.org/wiki/Q7251

# ## Flavours of Question answering (QA)
# 
# * Factoid QA
#     * Information retrieval (IR)-based QA (by **machine reading comprehension**) on **unstructured** data (text)
#     * Knowledge-based QA (by **semantic parsing** to logical form/SQL/SPARQL) on **structured** data (DB/KB)
# * Non-factoid QA
#     * Math problems ![math](https://d3i71xaburhd42.cloudfront.net/fb1c90806fc5ec72987f58110aa255edbce6620d/1-Figure1-1.png)
# 
# <div style="text-align: right;">
#     (from [Lu et al., 2021](https://aclanthology.org/2021.acl-long.528/))
# </div>
# 
#     * "How" questions
# > How do I delete my Instagram account?
# 
#     * "Why" questions
# > Why is the sky blue?
#     * ...

# ### QA datasets
# 
# * SQuAD ([Rajpurkar et al., 2016](https://www.aclweb.org/anthology/D16-1264.pdf), [Rajpurkar & Jia et al., 2018](https://www.aclweb.org/anthology/P18-2124.pdf))
# * QuAC ([Choi et al., 2018](https://www.aclweb.org/anthology/D18-1241.pdf))
# * CoQA ([Reddy et al., 2019](https://www.aclweb.org/anthology/Q19-1016.pdf))
# * Natural Questions ([Kwiatkowski et al., 2019](https://www.aclweb.org/anthology/Q19-1026.pdf))
# * TyDI-QA ([Clark et al., 2020](https://www.aclweb.org/anthology/2020.tacl-1.30.pdf))
# * ...
# 

# ## Information retrieval (IR)-based QA
# 
# General approach:
# 1. Retrieve relevant **passage**(s)
# 2. Machine reading comprehension: extract the **answer**, which can be
#     * A text span from the passage
#     * Yes/no
#     * `NULL` (unanswerable)
# 

# ### Machine reading comprehension (MRC)
# 
# * Input: (Passage, Question)
# * Output: Answer span
# 
# <center>
#     <img src="https://rajpurkar.github.io/mlx/qa-and-squad/example-squad.png" width="70%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Rajpurkar et al., 2016](https://www.aclweb.org/anthology/D16-1264.pdf))
# </div>

# ### MRC demo
# 
# ### https://demo.allennlp.org/reading-comprehension/MjMzNTgxOA==
# 

# ### MRC modeling
# 
# How to model span selection?
# 
# * As sequence labeling (for each token, is it part of the answer?)
# 
# What may be the possible tags for each token?
# 
# * As span selection (find start and end of the answer span)
# 

# ### MRC evaluation
# 
# Test questions have $k$ gold answers by different human annotators ($k=3$ for SQuAD and TyDI-QA, $k=5$ for NQ)
# 
# Metrics for binary (yes/no) QA:
# * **Accuracy**
# * **F1**

# Metrics for ranking:
# * **MRR** (mean reciprocal rank)

# Metrics for span selection:
# * **Exact match** (EM): text is exactly any of the $k$
# * Word **F1** (see [sequence labeling slides](sequence_labeling_slides.ipynb)) averaged over the $k$ gold answers
#     * Often ignoring punctuation and articles, i.e., `a, an, the`
#     * As bag-of-words, not exact positions (because the same answer may appear multiple times)
#     * Macro-averaged: calculate F1 for each question and average the F1s

# ### SQuAD
# 
# <center>
#     <a href="slides/cs224n-2020-lecture10-QA.pdf"><img src="qa_figures/squad.png"></a>
# </center>
# 
# <div style="text-align: right;">
#     (from [Rajpurkar et al., 2016](https://www.aclweb.org/anthology/D16-1264.pdf))
# </div>

# ### MRC Models
# 
# ![model](https://d3i71xaburhd42.cloudfront.net/1b78ce27180c324f3831f5395a2fdf738e143e74/2-Figure1-1.png)
# 
# <div style="text-align: right;">
#     (from <a href="https://aclanthology.org/2020.aacl-srw.21/">Li et al., 2020</a>)
# </div>

# ### MRC with BERT
# 
# <center>
#     <img src="http://jalammar.github.io/images/bert-tasks.png" width=60%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/N19-1423.pdf">Devlin et al., 2019</a>)
# </div>

# ## Knowledge-based (KB) question answering
# 
# Information is already organized in tables, databases and knowledge bases!
# 
# 1. (Executable) **semantic parsing**: translate natural language question to SQL/SPARQL/logical form **program** (query).
# 2. **Execute** the program on a database/knowledge-base and return the answer.

# ### Knowledge Bases
# 
# ![wikidata](https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Wikidata-logo-en.svg/500px-Wikidata-logo-en.svg.png)
# 
# [Which university did Turing go to?](https://query.wikidata.org/#select%20distinct%20%3Fitem%20%3FitemLabel%20where%20%7B%0A%20%20%20%20%3Fitem%20wdt%3AP31%20wd%3AQ15936437.%0A%20%20%20%20wd%3AQ7251%20wdt%3AP69%20%3Fitem.%0A%20%20%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%20%7D%0A%7D%0AORDER%20BY%20DESC%28%3Fsitelinks%29)

# ### Executable semantic parsing to SPARQL
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/8b65582bcb84b30393c67a2bae71a9e84f45e87c/4-Figure1-1.png" width="100%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Keysers et al., 2020](https://arxiv.org/pdf/1912.09713.pdf))
# </div>

# ### Executable semantic parsing to SQL
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/37882abaec01eba1bf5bda8a36c904aaea0d5642/6-Table1-1.png" width="80%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Oren et al., 2020](https://arxiv.org/pdf/2010.05647.pdf))
# </div>

# ### Executable semantic parsing to SQL
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/23474a845ea4b67f38bde7c7f1c4c1bdba22c50c/1-Figure1-1.png" width="80%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Finegan-Dollak et al., 2018](https://www.aclweb.org/anthology/P18-1033.pdf))
# </div>

# ### Executable semantic parsing to logical form
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/b29447ba499507a259ae9d8f685d60cc1597d7d3/1-Figure1-1.png" width="50%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Berant et al., 2013](https://www.aclweb.org/anthology/D13-1160.pdf))
# </div>

# Why read when you can query?
# 
# ### https://ucph.padlet.org/dh/qa

# ## Summary
# 
# * Relation extraction can be cast as question answering (and vice versa)
# * Information retrieval-based question answering require reading comprehension
# * Knowledge-based question answering requires semantic parsing

# ## Background Material
# 
# * Question Answering. Blog post by Vered Shwartz: http://veredshwartz.blogspot.com/2016/11/question-answering.html
# * Jurafky, Dan and Martin, James H. (2016). Speech and Language Processing, Chapter 25 (Question Answering): https://web.stanford.edu/~jurafsky/slp3/25.pdf

# ## Further Reading
# 
# * Conversational QA: https://abbanmustafa.github.io/slides-deck-1
# * Measuring Compositional Generalization. https://ai.googleblog.com/2020/03/measuring-compositional-generalization.html
# * Multilingual Compositional Wikidata Questions. https://arxiv.org/abs/2108.03509
