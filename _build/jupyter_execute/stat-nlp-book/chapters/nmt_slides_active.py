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


# # Machine Translation

# # What languages do you speak?
# 
# <a href="https://www.mentimeter.com/s/389360b38fa508b4ffd4e40bf47003e4/f43e4a2212e8/edit"><img src="../img/languages-2021.png"></a>

# ## Outline
# 
# + History of machine translation
# 
# + Exercise: pitfalls of machine translation
# 
# + Break
# 
# + Cross-lingual transfer learning
# 

# ## Languages are hard (even for humans)!
# 
# <center style="padding-top:3em;">
#   <img src="mt_figures/whatever.jpg" />
#     
#   <span style="font-size:50%;">(Source: <a href="https://www.flickr.com/photos/98991392@N00/8729849093/sizes/z/in/pool-47169589@N00/">Flickr</a>)</span>
# </center>
# 
# [随便](https://translate.google.com/#view=home&op=translate&sl=zh-CN&tl=en&text=%E9%9A%8F%E4%BE%BF)

# ## Automatic machine translation is hard!
# 
# <center style="padding-top:3em;">
# <img src="../chapters/mt_figures/avocado.png" width="100%"/>
# </center>
# 
# [J'ai besoin d'un avocat pour mon procés de guacamole.](https://translate.google.com/?sl=fr&tl=en&text=J%27ai%20besoin%20d%27un%20avocat%20pour%20mon%20proc%C3%A9s%20de%20guacamole.&op=translate)
# 
# [guacamole lawsuit](https://www.latimes.com/archives/la-xpm-2006-dec-10-fi-letters10.2-story.html)

# ## How to implement?
# 
# <center>
#    <img src="mt_figures/brief_history.png" width="100%" />
#    
#    <span style="font-size:50%;">(Source: <a href="https://www.freecodecamp.org/news/a-history-of-machine-translation-from-the-cold-war-to-deep-learning-f1d335ce8b5/">freeCodeCamp</a>)</span>
# </center>

# ## The Vauquois triangle
# 
# <center>
#    <img src="https://upload.wikimedia.org/wikipedia/commons/3/30/Vauquois%C2%B4_p%C3%BCramiid.PNG" width="50%" />
# </center>

# Many things could go wrong.

# ## Exercise: Pitfalls of Machine Translation
# ### https://ucph.page.link/mt
# 

# ## Outlook
# 
# Competitive machine translation models are *very expensive* to train!
# 
# + Example: [Wu et al. (2016)](https://arxiv.org/pdf/1609.08144.pdf) describe Google's NMT system
# 
# + Encoder–decoder with attention & stack of 8 LSTM layers
#   (plus some other additions)
#   
# + 36 million sentence pairs for English-to-French setting (En→Fr)
# 
# Quote:
# 
# > On WMT En→Fr, it takes around 6 days to train a basic model using 96 NVIDIA K80 GPUs.
# 

# ## Improving efficiency and quality
# 
# + Bigger data
# + Bigger models
# + Better neural network architectures
# + Semi-supervised learning
# + **Transfer learning**

# ## Further reading
# 
# * Non-neural machine translation:
#   + Ilya Pestov's article [A history of machine translation from the Cold War to deep learning](https://www.freecodecamp.org/news/a-history-of-machine-translation-from-the-cold-war-to-deep-learning-f1d335ce8b5/)
#   + [Slides on SMT from this repo](word_mt_slides.ipynb)
#   + Mike Collins's [Lecture notes on IBM Model 1 and 2](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/ibm12.pdf)
# 
# * Sequence-to-sequence models:
#   + Graham Neubig, [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/abs/1703.01619)
# 
# * And beyond...
#   + Philipp Koehn, [Neural Machine Translation, §13.6–13.8](https://arxiv.org/abs/1709.07809) gives a great overview of further refinements and challenges
