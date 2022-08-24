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

# ![dogs](../img/whatwesaytodogs.png)
#     

# Can you think of NLP Applications?

# ## Speech Recognition
# 
# Speech Recognition is usually not considered NLP. We will not cover this topic here.

# ![beach](../img/wreckanicebeach.png)
# 
# recognise speech vs. wreck a nice beach
# 

# ## Machine Translation
# 
# ![mt](../img/mt.png)
# 
# http://translate.google.com/

# ## Personal Assistants
# <img src="../img/siri1.png" style="float:left;" width=45%></img><img src="../img/siri2.png" width=45%></img>

# ## Information Extraction
# ![ie1](../img/ie1.png)

# ## Information Extraction
# ![ie2](../img/ie2.png)

# ## Summarisation
# ![sum](../img/summarization2.png)

# ## Generation
# ![gen](../img/shirtless2.jpeg)

# ## Generation
# ![gen](../img/shirtless.jpeg)

# ## Question Answering
# ![qa](../img/qa2.png)

# ## Sentiment Analysis
# 
# ![sent](../img/sentiment_0.png)

# ## Machine Comprehension
# 
# ![mc](../img/comprehension2.png)

# ## Cognitive Science and Psycholinguistics 
# 
# ![cog](../img/psycho.png)

# ## What is difficult about NLP?
# 
# [Play the Structural Ambiguity Game](http://madlyambiguous.osu.edu:1035/)
# 
# Enter sentences you fooled the systems with and sentences it guessed correctly: https://tinyurl.com/ycpd5gaq (submit multiple times for more than one answer) 

# ## Why is it difficult?
# 
# ![sailor_moon1](../img/sailor_moon_1.jpg)

# ## Why is it difficult?
# 
# ![sexcoffee](../img/sailor_moon1.jpg)

# ## Why is it difficult?
# 
# ![sexcoffee](../img/sailor_moon2.jpg)

# ### Ambiguity Everywhere
# 
# * Fed <b>raises</b> interest rates 0.5% in effort to control inflation
# * Fed raises <b>interest</b> rates 0.5% in effort to control inflation
# * Fed raises interest <b>rates</b> 0.5% in effort to control inflation

# ### Fool a [Sentiment Analyser](http://text-processing.com/demo/sentiment/)

# ### Fool a [Machine Translator](http://translate.google.com/?hl=en&tab=TT)

# ### [Count](https://books.google.com/ngrams) N-grams

# ## Core NLP Tasks
# * Tokenisation, Segmentation
# * Part of Speech Tagging
# * Language Modelling
# * Machine Translation
# * Syntactic and Semantic Parsing
# * Document Classification
# * Information Extraction
# * Question Answering

# ## Why is it difficult?
# 
# ![sexcoffee](../img/sailor_moon_2.jpg)

# ## Core NLP Methods
# 
# * Structured Prediction 
# * Preprocessing
# * Generative Learning
# * Discriminative Learning
# * Weak Supervision
# * Representation and Deep Learning
