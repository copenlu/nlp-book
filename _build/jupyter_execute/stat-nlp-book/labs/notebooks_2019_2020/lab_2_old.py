#!/usr/bin/env python
# coding: utf-8

# The focus of the second lab are details of the [structured prediction recipe](../chapters/structured_prediction.ipynb) and [tokenization](../chapters/tokenization.ipynb).
# 
# We will go through the notes of each of these sections to highlight issues, explain code, and point to exercise which you will work on.

# # Structured prediction
# 
# Solve [exercises](../exercises/structured_prediction.ipynb) in structured prediction.

# # Tokenization
# 
# ## Warm-up questions
# As we've seen, tokenization is an important step in the NLP pipeline, and it is not as trivial as one might imagine.
# 
# - Can you think of 3 domains where one might expect some issues with simple tokenization we considered in the lecture?
# 
# - Why isn't it possible to solve tokenization just by remembering words? What about if we limit ourselves to English only (an estimate says [English has 1,025,109.8 words](http://www.languagemonitor.com/number-of-words/number-of-words-in-the-english-language-1008879/))?
# 
# - In the lecture we went through four iterations of building a simple tokenizer in the lecture:
#   - splitting by blank space
#   - splitting by any whitespace character
#   - tokenization through definition of tokens as sequences of alphanumeric characters and some punctuation
#   - tokenization by addition of words to our tokenizer
# - What are the shortcomings of each one of these approaches?
# 
# Food for thought: we're doing tokenization to split a string into a series of tokens and process them accordingly. After it we're left with an array of symbols whwich we will map into a specific representation, and continue with our machine learning model. Should all tokens be included into further processing? Do all of them contribute to the performance of the model? Are there different tokens which denote the same underlying word? What should we do with them? What about punctuation, is punctuation always important? When it is, when it is not?
# 
# Next, solve [tokenization exercises](../exercises/tokenization.ipynb)
