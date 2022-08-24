#!/usr/bin/env python
# coding: utf-8

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
# ![mt](../img/avocado.png)
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
# <img src="../img/qa.png" width="400"/>
# </div>
# <div>
# <img src="../img/qa2.png" width="900"/>
# </div>
# <div style="text-align: right;">
# <a href="https://doi.org/10.1016/j.artint.2012.06.009">Ferrucci et al. (2013)</a>
# </div>

# ## Reading Comprehension
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
# Discuss and enter your answers here: https://ucph.page.link/nlp_q1
# 
# ([Responses](https://docs.google.com/forms/d/121VI1BeO1TWuWXnAeQbHcBdMazQ3rPivoko29YyrZz4/edit#responses))

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
