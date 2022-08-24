#!/usr/bin/env python
# coding: utf-8

# # Tokenization Exercises
# 
# In the lecture we took a look at a simple tokenizer and sentence segmenter. In this exercise we will expand our understanding of the problem by asking a few important questions, and looking at the problem from a different perspectives.

# ##  <font color='green'>Setup 1</font>: Load Libraries

# In[1]:


import re


# ## <font color='blue'>Task 1</font>: Improving tokenization
# 
# Write a tokenizer to correctly tokenize the following text:

# In[2]:


text = """'Curiouser and curiouser!' cried Alice (she was so much surprised, that for the moment she quite
forgot how to speak good English); 'now I'm opening out like the largest telescope that ever was! Good-bye,
feet!' (for when she looked down at her feet, they seemed to be almost out of sight, they were getting so far
off). 'Oh, my poor little feet, I wonder who will put on your shoes and stockings for you now, dears? I'm sure I
shan't be able! I shall be a great deal too far off to trouble myself about you: you must manage the best
way you can; —but I must be kind to them,' thought Alice, 'or perhaps they won't walk the way I want to go!
Let me see: I'll give them a new pair of boots every Christmas.'
"""

token = re.compile('Mr.|[\w\']+|[.?]')
tokens = token.findall(text)
#print(tokens)
print(tokens[:2])


# Questions:
# - should one separate 'm, 'll, n't, possessives, and other forms of contractions from the word?
# - should elipsis be considered as three '.'s or one '...'?
# - there's a bunch of these small rules - will you implement all of them to create a 'perfect' tokenizer?

# ## <font color='blue'>Task 2</font>: Twitter Tokenization
# As you might imagine, tokenizing tweets differs from standard tokenization. There are 'rules' on what specific elements of a tweet might be (mentions, hashtags, links), and how they are tokenized. The goal of this exercise is not to create a bullet-proof Twitter tokenizer but to understand tokenization in a different domain.
# 
# Tokenize the following [UCLMR tweet](https://twitter.com/IAugenstein/status/766628888843812864) correctly:

# In[3]:


tweet = "#emnlp2016 paper on numerical grounding for error correction http://arxiv.org/abs/1608.04147  @geospith @riedelcastro #NLProc"
tweet


# In[4]:


token = re.compile('[\w\s]+')
tokens = token.findall(tweet)
print(tokens)


# Questions:
# - what does 'correctly' mean, when it comes to Twitter tokenization?
# - what defines correct tokenization of each tweet element?
# - how will your tokenizer tokenize elipsis (...)?
# - will it correctly tokenize emojis?
# - what about composite emojis?

# ## <font color='blue'>Task 3</font>: Improving sentence segmenter
# 
# Sentence segmentation is not a trivial task either. There might be some cases where your simple sentence segmentation won't work properly.
# 
# First, make sure you understand the following sentence segmentation code used in the lecture:

# In[5]:


import re

def sentence_segment(match_regex, tokens):
    """
    Splits a sequence of tokens into sentences, splitting wherever the given matching regular expression
    matches.

    Parameters
    ----------
    match_regex the regular expression that defines at which token to split.
    tokens the input sequence of string tokens.

    Returns
    -------
    a list of token lists, where each inner list represents a sentence.

    >>> tokens = ['the','man','eats','.','She', 'sleeps', '.']
    >>> sentence_segment(re.compile('\.'), tokens)
    [['the', 'man', 'eats', '.'], ['She', 'sleeps', '.']]
    """
    current = []
    sentences = [current]
    for tok in tokens:
        current.append(tok)
        if match_regex.match(tok):
            current = []
            sentences.append(current)
    if not sentences[-1]:
        sentences.pop(-1)
    return sentences


# Next, modify the following code so that sentence segmentation returns correctly segmented sentences on the following text:

# In[6]:


text = """
Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is the longest official one-word placename in U.K. Isn't that weird? I mean, someone took the effort to really make this name as complicated as possible, huh?! Of course, U.S.A. also has its own record in the longest name, albeit a bit shorter... This record belongs to the place called Chargoggagoggmanchauggagoggchaubunagungamaugg. There's so many wonderful little details one can find out while browsing http://www.wikipedia.org during their Ph.D. or an M.Sc.
"""

token = re.compile('Mr.|[\w\']+|[.?]')

tokens = token.findall(text)
sentences = sentence_segment(re.compile('\.'), tokens)
for sentence in sentences:
    print(sentence)


# Questions:
# - what elements of a sentence did you have to take care of here?
# - is it useful or possible to enumerate all such possible examples?
# - how would you deal with all URLs effectively?
# - are there any specific punctuation not covered in the example you might think of?

# ## Solutions
# 
# You can find the solutions to this exercises [here](tokenization_solutions.ipynb)
