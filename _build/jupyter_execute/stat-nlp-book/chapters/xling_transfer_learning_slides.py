#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n<style>\n.rendered_html td {\n    font-size: xx-large;\n    text-align: left; !important\n}\n.rendered_html th {\n    font-size: xx-large;\n    text-align: left; !important\n}\n</style>\n')


# In[2]:


get_ipython().run_line_magic('load_ext', 'tikzmagic')


# # Cross-lingual Transfer Learning

# # Outline
# 
# + Transfer learning overview
# + Cross-lingual learning methods
# + Exercise: cross-lingual learning for machine translation
# + Summary

# <img src="dl-applications-figures/WS_mapping.png" width="100%"/>
# 
# <div style="text-align: right;">
#     Source: http://ai.stanford.edu/blog/weak-supervision/
# </div>

# # Transfer learning
# 
# <img src="https://ruder.io/content/images/2019/08/transfer_learning_taxonomy.png" width=45%>

# ## Multi-task learning
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/161ffb54a3fdf0715b198bb57bd22f910242eb49/19-Figure1.2-1.png" width=45%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Caruana, 1997](https://www.cs.cornell.edu/~caruana/mlj97.pdf))
# </div>

# ## Domain adaptation
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/e505303ba5287f468773fbef22ab1abf6875efca/1-Figure1-1.png" width=45%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Wright & Augenstein, 2020](https://aclanthology.org/2020.emnlp-main.639/))
# </div>

# # Pre-training
# 
# General-purpose representations trained on large datasets:
# - [word2vec](https://arxiv.org/abs/1301.3781)
# - [GloVe](https://www.aclweb.org/anthology/D14-1162)
# - [ELMo](https://www.aclweb.org/anthology/N18-1202)
# - [BERT](https://www.aclweb.org/anthology/N19-1423)
# - [BART](https://aclanthology.org/2020.acl-main.703/)
# - [T5](https://arxiv.org/abs/1910.10683)

# ### Semi-supervised learning + sequential transfer learning
# 
# <center>
#     <img src="https://1.bp.blogspot.com/-89OY3FjN0N0/XlQl4PEYGsI/AAAAAAAAFW4/knj8HFuo48cUFlwCHuU5feQ7yxfsewcAwCLcBGAsYHQ/s640/image2.png" width=50%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Raffel et al., 2019](https://arxiv.org/abs/1910.10683))
# </div>

# ### XTREME
# 
# Collection of multilingual multi-task benchmarks (including **TyDiQA**)
# 
# <center>
#     <a href="https://sites.research.google/xtreme">
#     <img src="https://1.bp.blogspot.com/-5J6e2txWChk/XpSc_BaYFnI/AAAAAAAAFss/QCLROHrEutAN3GvOyfRzK8J7DA9yLY5GACLcBGAsYHQ/s640/XTREME%2BStill%2Bart_04%2Broboto.png" width="90%">
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from [Hu et al., 2020](https://arxiv.org/abs/2003.11080))
# </div>

# ### Strategies for cross-lingual transfer
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/96c7f39c358343b6ea412697ed693fdd04a71516/2-Figure1-1.png" width="90%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Horbach et al., 2020](https://aclanthology.org/W18-0550/))
# </div>

# * **Zero-shot**

# # Zero-shot cross-lingual transfer
# 
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/f6e181c0bd967e9b797d09c579f2ad3fccdbacd2/2-Figure1-1.png" width=100%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Artetxe et al., 2020](https://aclanthology.org/2020.acl-main.421/))
# </div>

# # Multilingual pre-training
# 
# - [MUSE](https://openreview.net/forum?id=H196sainb)
# - [mBERT](https://www.aclweb.org/anthology/N19-1423)
# - [XLM-R](https://aclanthology.org/2020.acl-main.747/)
# - [mBART](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00343/96484/Multilingual-Denoising-Pre-training-for-Neural)
# - [mT5](https://arxiv.org/abs/2010.11934)
# 
# <center>
#     <img src="https://ruder.io/content/images/size/w2000/2016/10/zou_et_al_2013.png" width=50%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Zou et al., 2013](https://aclanthology.org/D13-1141/))
# </div>

# # Multilingual pre-training for MT
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/2106af97d77c45fdbf702bcbea319a185b7b719f/1-Figure1-1.png" width=80%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Chen et al., 2021](https://arxiv.org/abs/2104.08757))
# </div>

# ### Exercise: multilingual pre-training for MT
# Consider a machine translation system using a multilingual pre-trained encoder and decoder.
# <a href="https://docs.google.com/forms/d/14osHSWt2D3cpg4QTbqm3wHkYtSzLWcVeRMhymQdMZj0/edit"><img src="../img/vauquois-2021.png"></a>
# 
# + It depends on the encoding and model.
# + If the performance is good, the attention should incorporate also pragmatic differences. But it might be difficult in practice.
# + Because it encodes the input into a numerical representation of the meaning that is language-independent.

# ### XTREME
# 
# <center>
#     <img src="dl-applications-figures/xtreme.png" width="70%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Hu et al., 2020](https://arxiv.org/abs/2003.11080))
# </div>

# # Further reading
# 
# - [Ruder, 2017. An Overview of Multi-Task Learning in Deep Neural Networks](https://ruder.io/multi-task/)
# - [Ruder, 2019. The State of Transfer Learning in NLP](https://ruder.io/state-of-transfer-learning-in-nlp/)
# - [Ruder, 2016. A survey of cross-lingual word embedding models](https://ruder.io/cross-lingual-embeddings/)
# - [Søgaard, Anders; Vulic, Ivan; Ruder, Sebastian; Faruqui, Manaal. 2019. Cross-lingual word embeddings](https://www.morganclaypool.com/doi/abs/10.2200/S00920ED2V01Y201904HLT042)
