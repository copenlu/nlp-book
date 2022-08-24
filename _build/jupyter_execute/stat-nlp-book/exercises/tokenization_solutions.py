#!/usr/bin/env python
# coding: utf-8

# # Tokenization Exercises - Solutions
# 
# In the lecture we took a look at a simple tokenizer and sentence segmenter. In this exercise we will expand our understanding of the problem by asking a few important questions, and looking at the problem from a different perspectives.

# ##  <font color='green'>Setup 1</font>: Load Libraries

# In[1]:


import re


# ## <font color='blue'>Task 1 solution</font>
# 
# To tokenise the following text:

# In[2]:


text = """'Curiouser and curiouser!' cried Alice (she was so much surprised, that for the moment she quite
forgot how to speak good English); 'now I'm opening out like the largest telescope that ever was! Good-bye,
feet!' (for when she looked down at her feet, they seemed to be almost out of sight, they were getting so far
off). 'Oh, my poor little feet, I wonder who will put on your shoes and stockings for you now, dears? I'm sure I
shan't be able! I shall be a great deal too far off to trouble myself about you: you must manage the best
way you can; —but I must be kind to them,' thought Alice, 'or perhaps they won't walk the way I want to go!
Let me see: I'll give them a new pair of boots every Christmas.'
"""


# We need to expand the list of tokens a bit to account for:
# - additional characters like ! ( ) , ; — - (notice the difference in the last two dashes)
# - separation of `n't` from the rest of the word (question: why?)
# - `'ll` `'m` (question: why?)
# 
# Should you need additional help on regular expressions, check https://regex101.com/

# In[3]:


token = re.compile("[\w]+(?=n't)|n't|\'m|\'ll|[\w]+|[.?!;,\-\(\)—\:']")
tokens = token.findall(text)
print(tokens[:])


# ## <font color='blue'>Task 2 solution</font>
# 
# The following token from [UCLMR tweet](https://twitter.com/IAugenstein/status/766628888843812864):

# In[4]:


tweet = "#emnlp2016 paper on numerical grounding for error correction http://arxiv.org/abs/1608.04147  @geospith @riedelcastro #NLProc"


# can be tokenised in various ways, but the main objective here is to 'catch' hashtags, mentions and URLs. Hashtags and mentions can be extracted easily by just simply adding `#` and `@` to the regular expression part which catches alphanumeric sequences.
# 
# However, catching URLs is a bit more tricky, and the minimum working example for this case would be the coded example:

# In[5]:


# hashtags and user mentions should be included, as well as the hyperlinks - there are more elaborate URL regular expressions, but this one will do for now
token = re.compile('http://[a-zA-Z0-9./]+|[@#\w]+')
tokens = token.findall(tweet)
print(tokens)


# However, bear in mind that this is far from a correct solution for capturing URLs, and that many valid URLs would not be correctly caught with this expression (think https, querystrings, etc.)

# ## <font color='blue'>Task 3 solution</font>
# 
# We first make sure to account for:
# - URLs
# - abbreviations (U.S.A., M.Sc., etc.)
# - elipsis (...)
# - question and exclamation mark, and their composition as a single token
# 
# We modify the tokeniser accordingly and check to verify we're happy with the tokenisation.

# In[6]:


text = """Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is the longest official one-word placename in U.K. Isn't that weird? I mean, someone took the effort to really make this name as complicated as possible, huh?! Of course, U.S.A. also has its own record in the longest name, albeit a bit shorter... This record belongs to the place called Chargoggagoggmanchauggagoggchaubunagungamaugg. There's so many wonderful little details one can find out while browsing http://www.wikipedia.org during their Ph.D. or an M.Sc."""

token = re.compile('http://[a-zA-Z0-9./]+|(?:[A-Za-z]{1,2}\.)+|[\w\']+|\.\.\.|\?\!|[.?!]')
tokens = token.findall(text)
print(tokens)


# This seems to be fine. However, we won't be able to segment sentences here properly because one of the cases `U.K.` in front of `Isn't` cannot be catched with the existing method. Minima working solution to cover that case would be a solutio where we would check pairs of tokens, and check whether a token that ends on `.` is followed by a token starting with a capital letter. In addition, we expand the list of splitting symbols to the symbols we find in our sentence ends. Notice that this is a gross oversimplification which would fail miserably in specific cases (question: can you think of cases where this might happen?)

# In[7]:


def bigrams(tokens):
    """Helper function to extract bigrams of tokens"""
    tokens.append(' ')
    return list(zip(tokens, tokens[1:]))

def new_sentence_segment(match_regex, tokens):
    current = []
    sentences = [current]
    
    for tok, tok2 in bigrams(tokens):
        current.append(tok)
        # we additionally check for . at the end of the first and
        # upper case letter in the beginning of the following token
        if match_regex.match(tok) or (tok[-1]=='.' and tok2[0].isupper()):
            current = []
            sentences.append(current)
    if not sentences[-1]:
        sentences.pop(-1)
    return sentences

sentences = new_sentence_segment(re.compile('\?|\.\.\.|\.'), tokens)
for sentence in sentences:
    print(sentence)

