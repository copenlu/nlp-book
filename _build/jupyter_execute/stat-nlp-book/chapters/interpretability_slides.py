#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[2]:


get_ipython().run_line_magic('load_ext', 'tikzmagic')


# # Interpretability
# 
# or: _putting the __science__ in data science (and NLP)_

# # Outline
# 
# * Motivation (10 min.)
# * Probes (20 min.)
# * Adversaries (10 min.)
# * Visualization (10 min.)

# # Motivation
# 
# <center>
#     <a href="slides/acl_2020_interpretability_tutorial-4-12.pdf">
#     <img src="https://4.bp.blogspot.com/-dfHBPg2rXcA/UC4v-5OPXhI/AAAAAAAAHic/EMCX2mOV8Go/s1600/ikea-00-instructions-orig.png" width=30%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from [Belinkov, Gehrmann and Pavlick, 2020](https://www.aclweb.org/anthology/2020.acl-tutorials.1.pdf); [slides](https://sebastiangehrmann.com/assets/files/acl_2020_interpretability_tutorial.pdf))
# </div>

# ## Opening the black box
# 
# <center>
#     <a href="slides/cs224n-2020-lecture20-interpretability-5-9.pdf">
#     <img src="https://imgs.xkcd.com/comics/machine_learning.png" width=30%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from John Hewitt; [slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture20-interpretability.pdf))
# </div>

# ## Probing MT models
# 
# <center>
#     <a href="slides/acl_2020_interpretability_tutorial-24-34.pdf">
#     <img src="https://d3i71xaburhd42.cloudfront.net/fc4bfa761f888806eea985e5fe6d16f83af93a10/4-Figure4-1.png" width=50%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from [Belinkov, Gehrmann and Pavlick, 2020](https://www.aclweb.org/anthology/2020.acl-tutorials.1.pdf); [slides](https://sebastiangehrmann.com/assets/files/acl_2020_interpretability_tutorial.pdf))
# </div>

# ## Language models as linguistic test subjects
# 
# <center>
#     <a href="slides/cs224n-2020-lecture20-interpretability-14-21.pdf">
#     <img src="https://paeaonline.org/wp-content/uploads/2015/09/multiple-choice-757x426.jpg" width=50%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from John Hewitt; [slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture20-interpretability.pdf))
# </div>

# ## Designing probes
# 
# <center>
#     <a href="slides/acl_2020_interpretability_tutorial-80-98.pdf">
#     <img src="https://d3i71xaburhd42.cloudfront.net/9d87300892911275520a4f7a5e5abf4f1c002fec/2-Figure1-1.png" width=50%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from [Belinkov, Gehrmann and Pavlick, 2020](https://www.aclweb.org/anthology/2020.acl-tutorials.1.pdf); [slides](https://sebastiangehrmann.com/assets/files/acl_2020_interpretability_tutorial.pdf))
# </div>

# ## Adversarial examples
# 
# <center>
#     <a href="slides/acl_2020_interpretability_tutorial-94-115.pdf">
#     <img src="https://blog.acolyer.org/wp-content/uploads/2017/09/adversarial-reading-fig-1.jpeg?w=480" width=50%>
#     </a>
# </center>
# 
# (from [Belinkov, Gehrmann and Pavlick, 2020](https://www.aclweb.org/anthology/2020.acl-tutorials.1.pdf); [slides](https://sebastiangehrmann.com/assets/files/acl_2020_interpretability_tutorial.pdf))

# ## Visualization
# 
# <center>
#     <a href="acl_2020_interpretability_tutorial_viz.pdf">
#     <img src="https://d3i71xaburhd42.cloudfront.net/fafb602db42240f5fb1e1b113fa0ed8647b45adc/8-Figure5-1.png" width=50%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from [Belinkov, Gehrmann and Pavlick, 2020](https://www.aclweb.org/anthology/2020.acl-tutorials.1.pdf); [slides](https://sebastiangehrmann.com/assets/files/acl_2020_interpretability_tutorial.pdf))
# </div>

# ## Close inspection
# 
# <center>
#     <a href="slides/cs224n-2020-lecture20-interpretability-26-32.pdf">
#     <img src="https://www.parismou.org/sites/default/files/inspections_1.jpg" width=30%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from John Hewitt; [slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture20-interpretability.pdf))
# 

# ### Attention is not explanation
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/1e83c20def5c84efa6d4a0d80aa3159f55cb9c3f/1-Figure1-1.png" width=80%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Jain and Wallace, 2019](https://www.aclweb.org/anthology/N19-1357.pdf))
# </div>

# ### Attention is not not explanation
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/ce177672b00ddf46e4906157a7e997ca9338b8b9/3-Table1-1.png" width=80%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Wiegreffe and Pinter, 2019](https://www.aclweb.org/anthology/D19-1002.pdf))
# </div>
# 

# ### LIME
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/5091316bb1c6db6c6a813f4391911a5c311fdfe0/4-Figure2-1.png" width=90%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Ribeiro et al., 2016](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
# </div>

# ## Meta-analyses and tools
# 
# <center>
#     <a href="slides/cs224n-2020-lecture20-interpretability-110-112.pdf">
#     <img src="https://images-na.ssl-images-amazon.com/images/I/71ZunWVQ0LL._SL1500_.jpg" width=30%>
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from John Hewitt; [slides](http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture20-interpretability.pdf))
# </div>
# 

# ### MRC demo
# 
# <center>
#     <a href="https://demo.allennlp.org/reading-comprehension/MjMzNTgxOA==">
#     <img src="https://raw.githubusercontent.com/allenai/allennlp/master/docs/img/allennlp-logo-dark.png" width=30%>
#     </a>
# </center>

# # Further reading
# 
# - [Belinkov and Glass, 2020. Analysis Methods in Neural Language Processing: A Survey](https://www.aclweb.org/anthology/Q19-1004.pdf)
# - [Hewitt, 2020. Designing and Interpreting Probes](https://nlp.stanford.edu//~johnhew//interpreting-probes.html)
# - [Lawrence, 2020. Interpretability and Analysis of Models for NLP @ ACL 2020](https://medium.com/@lawrence.carolin/interpretability-and-analysis-of-models-for-nlp-e6b977ac1dc6)
