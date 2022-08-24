#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%cd ..\nimport statnlpbook.tokenization as tok\n')


# # Tokenisation
# 
# Before a program can process natural language, we need identify the _words_ that constitute a string of characters. This, in fact, can be seen as a crucial transformation step to improve the input *representation* of language in the [structured prediction recipe](structured_prediction.ipynb).
# 
# By default text on a computer is represented through `String` values. These values store a sequence of characters (nowadays mostly in [UTF-8](http://en.wikipedia.org/wiki/UTF-8) format). The first step of an NLP pipeline is therefore to split the text into smaller units corresponding to the words of the language we are considering. In the context of NLP we often refer to these units as _tokens_, and the process of extracting these units is called _tokenisation_. Tokenisation is considered boring by most, but it's hard to overemphasize its importance, seeing as it's the first step in a long pipeline of NLP processors, and if you get this step wrong, all further steps will suffer. 
# 
# 
# In Python a simple way to tokenise a text is via the `split` method that divides a text wherever a particular substring is found. In the code below this pattern is simply the whitespace character, and this seems like a reasonable starting point for an English tokenisation approach.

# In[3]:


text = "Mr. Bob Dobolina is thinkin' of a master plan." + \
       "\nWhy doesn't he quit?"
text.split(" ")


# ## Tokenisation with Regular Expressions
# Python allows users to construct tokenisers using [regular expressions](http://en.wikipedia.org/wiki/Regular_expression) that define the character sequence patterns at which to either split tokens, or patterns that define what constitutes a token. In general regular expressions are a powerful tool NLP practitioners can use when working with text, and they come in handy when you work with command line tools such as [grep](http://en.wikipedia.org/wiki/Grep). In the code below we use a simple pattern `\\s` that matches any whitespace to define where to split.

# In[4]:


import re
gap = re.compile('\s')
gap.split(text)


# One shortcoming of this tokenisation is its treatment of punctuation because it considers "plan." as a token whereas ideally we would prefer "plan" and "." to be distinct tokens. It is easier to address this problem if we define what a token token is, instead of what constitutes a gap. Below we have define tokens as sequences of alphanumeric characters and punctuation.

# In[5]:


token = re.compile('\w+|[.?:]')
token.findall(text)


# This still isn't perfect as "Mr." is split into two tokens, but it should be a single token. Moreover, we have actually lost an apostrophe. Both is fixed below, although we now fail to break up the contraction "doesn't".

# In[6]:


token = re.compile('Mr.|[\w\']+|[.?]')
tokens = token.findall(text)
tokens


# ## Learning to Tokenise
# For most English domains powerful and robust tokenisers can be built using the simple pattern matching approach shown above. However, in languages such as Japanese, words are not separated by whitespace, and this makes tokenisation substantially more challenging. Try to, for example, find a good *generic* regular expression pattern to tokenise the following sentence.

# In[7]:


jap = "彼は音楽を聞くのが大好きです"
re.compile('彼|は|く|音楽|を|聞くの|が|大好き|です').findall(jap)


# Even for certain English domains such as the domain of biomedical papers, tokenisation is non-trivial (see an analysis why [here](https://aclweb.org/anthology/W/W15/W15-2605.pdf)).
# 
# When tokenisation is more challenging and difficult to capture in a few rules a machine-learning based approach can be useful. In a nutshell, we can treat the tokenisation problem as a character classification problem, or if needed, as a sequential labelling problem.

# # Sentence Segmentation
# Many NLP tools work on a sentence-by-sentence basis. The next preprocessing step is hence to segment streams of tokens into sentences. In most cases this is straightforward after tokenisation, because we only need to split sentences at sentence-ending punctuation tokens.
# 
# However, keep in mind that, as well as tokenisation, sentence segmentation is language specific - not all languages contain punctuation which denotes sentence boundary, and even if they do, not all segmentations are trivial (can you think of examples?).

# In[8]:


tok.sentence_segment(re.compile('\.'), tokens)


# # Background Reading
# 
# * Jurafsky & Martin, [Speech and Language Processing (Third Edition)](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf): Chapter 2, Regular Expressions, Text Normalization, Edit Distance.
# * Manning, Raghavan & Schuetze, Introduction to Information Retrieval: [Tokenization](http://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)
