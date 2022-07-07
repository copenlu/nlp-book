#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# # Introduction

# ## What is NLP?
# 
# * Building computer systems that **understand** and **generate** natural languages.
# * Deep understanding of **broad** language
#     * not just string processing or keyword matching
#     

# Can you think of NLP Applications?

# ## Speech Recognition
# 
# Speech Recognition is usually not considered NLP. We will not cover this topic here.

# ![beach](../img/wreckanicebeach.png)
# 
# recognise speech vs. wreck a nice beach
# 

# ## Sentiment Analysis
# 
# ![sent](../img/sentiment_0.png)
# 

# ## Machine Translation
# 
# ![mt](../img/mt.png)
# 
# http://translate.google.com/

# ## Information Extraction
# ![ie1](../img/ie1.png)

# ## Information Extraction
# ![ie2](../img/ie2.png)

# ## Generation
# ![gen](../img/shirtless2.jpeg)

# ## Generation
# ![gen](../img/shirtless.jpeg)

# ## Summarisation
# <div>
# <img src="../img/summarization2.png" width="1400"/>
# </div>

# ## Question Answering
# <div>
# <img src="../img/qa2.png" width="900"/>
# </div>

# ## Machine Comprehension
# 
# <div>
# <img src="../img/comprehension.png" width="1200"/>
# </div>

# ## Personal Assistants
# <table><tr>
# <td> <img src="../img/siri1.png" alt="Siri1" style="width: 500px;"/> </td>
# <td> <img src="../img/siri2.png" alt="Siri2" style="width: 500px;"/> </td>
# </tr></table>

# ## What is difficult about NLP?
# 
# Discuss and enter your answers here: https://tinyurl.com/kef4ptz4

# ### Ambiguity Everywhere
# 
# * Fed <b>raises</b> interest rates 0.5% in effort to control inflation
# * Fed raises <b>interest</b> rates 0.5% in effort to control inflation
# * Fed raises interest <b>rates</b> 0.5% in effort to control inflation

# ### Ambiguity Everywhere
# 
# "Jane ate spaghetti with a **silver spoon**."

# Do you mean...

# Jane used a silver spoon to eat spaghetti? (**cutlery**)

# Jane had spaghetti and a silver spoon? (**part**)

# Jane exhibited a silver spoon while eating spaghetti? (**manner**)

# Jane ate spaghetti in the presence of a silver spoon? (**company**)

# ### Ambiguity on different linguistic levels
# 
# <img src="../img/nlp_pyramid.png" style="float:left;" width=45%></img>

# ## Core NLP Tasks
# * Tokenisation, Segmentation
# * Part of Speech Tagging
# * Language Modelling
# * Machine Translation
# * Syntactic and Semantic Parsing
# * Document Classification
# * Information Extraction
# * Question Answering

# ## Core NLP Methods
# 
# * Structured Prediction 
# * Preprocessing
# * Generative Learning
# * Discriminative Learning
# * Weak Supervision
# * Representation and Deep Learning
