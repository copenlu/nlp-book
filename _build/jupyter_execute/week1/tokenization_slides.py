#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%cd ..\nimport statnlpbook.tokenization as tok\n')


# # Tokenisation

# ![nospaces](../img/nospaces.jpg)
# 
# * Identify the **words** in a string of characters.

# In Python you can tokenise text via `split`:

# In[3]:


text = """Mr. Bob Dobolina is thinkin' of a master plan.
Why doesn't he quit?"""
text.split(" ")


# What is wrong with this?

# In Python you can also tokenise using **patterns** at which to split tokens:
# ### Regular Expressions

# A **regular expression** is a compact definition of a **set** of (character) sequences.
# 
# Examples:
# * `"Mr."`: set containing only `"Mr."`
# * `" |\n|!!!"`: set containing the sequences `" "`, `"\n"` and `"!!!"`
# * `"[abc]"`: set containing only the characters `a`, `b` and `c`
# * `"\s"`: set of all whitespace characters
# * `"1+"`: set of all sequences of at least one `"1"` 
# * etc.
# 

# In[4]:


import re
re.compile('\s').split(text)


# Problems:
# * Bad treatment of punctuation.  
# * Easier to **define a token** than a gap. 

# Let us use `findall` instead:

# In[5]:


re.compile('\w+|[.?]').findall(text)


# Problems:
# * "Mr." is split into two tokens, should be single. 
# * Lost an apostrophe. 
# 
# Both is fixed below ...

# In[6]:


re.compile('Mr.|[\w\']+|[.?]').findall(text)


# ## Learning to Tokenise?
# * For English simple pattern matching often sufficient. 
# * In other languages (e.g. Japanese), words are not separated by whitespace.
# 

# In[7]:


jap = "今日もしないといけない。"


# Try lexicon-based tokenisation ...

# In[8]:


re.compile('もし|今日|も|しない|と|けない').findall(jap)


# Equally complex for certain English domains (eg. bio-medical text). 

# In[9]:


bio = """We developed a nanocarrier system of herceptin-conjugated nanoparticles
of d-alpha-tocopheryl-co-poly(ethylene glycol) 1000 succinate (TPGS)-cisplatin
prodrug ..."""


# * d-alpha-tocopheryl-co-poly is **one** token
# * (TPGS)-cisplatin are **five**: 
#   * ( 
#   * TPGS 
#   * ) 
#   * - 
#   * cisplatin 

# In[10]:


re.compile('\s').split(bio)[:15]


# Solution: Treat tokenisation as a **statistical NLP problem** (and as structured prediction)! 
#   * [classification](doc_classify.ipynb)
#   * [sequence labelling](sequence_labelling.ipynb)

# # Sentence Segmentation
# 
# * Many NLP tools work sentence-by-sentence. 
# * Often trivial after tokenisation: split sentences at sentence-ending punctuation tokens.

# In[11]:


tokens = re.compile('Mr.|[\w\']+|[.?]').findall(text)
# try different regular expressions
tok.sentence_segment(re.compile('\.'), tokens)


# What to do with transcribed speech? 
# 
# Discuss and enter your answer(s) here: https://tinyurl.com/yx8vfcom

# # Background Reading
# 
# * Jurafsky & Martin, [Speech and Language Processing (Third Edition)](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf): Chapter 2, Regular Expressions, Text Normalization, Edit Distance.
# * Manning, Raghavan & Schuetze, Introduction to Information Retrieval: [Tokenization](http://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)
