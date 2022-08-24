#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reveal.js
from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {
        'theme': 'white',
        'transition': 'none',
        'controls': 'false',
        'progress': 'true',
})


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n# %cd ..\nimport sys\nsys.path.append("..")\nimport statnlpbook.util as util\nutil.execute_notebook(\'language_models.ipynb\')\n')


# In[3]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[4]:


from IPython.display import Image
import random


# # Contextualised Word Representations
# 
# 

# ## What makes a good word representation? ##
# 
# 1. Representations are **distinct**
# 2. **Similar** words have **similar** representations

# ## What does this mean? ##
# 
# 
# * "Yesterday I saw a bass ..."

# In[5]:


Image(url='../img/bass_1.jpg'+'?'+str(random.random()), width=300)


# In[6]:


Image(url='../img/bass_2.svg'+'?'+str(random.random()), width=100)


# # Contextualised Representations #
# 
# * Standard embedding representations have *one* representation per word, regardless of the *current* context
# 
# * Contextualised Representations use the context surrounding the word
# 

# ## Contextualised Representations Example ##
# 
# 
# * a) "Yesterday I saw a bass swimming in the lake"

# In[7]:


Image(url='../img/bass_1.jpg'+'?'+str(random.random()), width=300)


# * b) "Yesterday I saw a bass in the music shop"

# In[8]:


Image(url='../img/bass_2.svg'+'?'+str(random.random()), width=100)


# ## Contextualised Representations Example ##
# 
# 
# * a) <span style="color:red">"Yesterday I saw a bass swimming in the lake"</span>.
# * b) <span style="color:green">"Yesterday I saw a bass in the music shop"</span>.

# In[9]:


Image(url='../img/bass_visualisation.jpg'+'?'+str(random.random()), width=500)


# ## What makes a good representation? ##
# 
# 1. Representations are **distinct**
# 2. **Similar** words have **similar** representations

# NEW!
# 
# 3. Representations take **context** into account

# ## How to train contextualised representations ##
# 
# * Make sure the representations of each word differs by its context
#     * Context words by themselves should in turn have different representations depending on *their* context
# * Because defining all possible contexts is infeasible / impraticable, contextualised representations are not static, but are functions that return representations for words given their context
#     * In practice, those are trained models

# Does this remind you of anything?

# \begin{align}
# p(w_1,\ldots,w_d) &= p(w_1) p(w_2|w_1) p(w_3|w_1, w_2) \ldots \\
#  &= p(w_1) \prod_{i = 2}^d p(w_i|w_1,\ldots,w_{i-1})
# \end{align}
# 

# ## Using Bi-LSTMs to learn *contextualised* word embeddings ##
# 
# * **Idea**: the hidden layer corresponding to a single word in a RNN language model can also be thought of as an embedding of that word – but one that is dependent on the context
#     * Bi-LSTMs capture both left and right context of a word
# * **Proposal**: utilise hidden layers in Bi-LSTM to obtain word representations
#     * Deep contextualised word representations ("ELMo", Peters et al., 2018: https://www.aclweb.org/anthology/N18-1202/)

# In[10]:


Image(url='../img/elmo_1.png'+'?'+str(random.random()), width=800)


# Image credit: http://jalammar.github.io/illustrated-bert/

# In[11]:


Image(url='../img/elmo_2.png'+'?'+str(random.random()), width=1200)


# In[12]:


Image(url='../img/elmo_3.png'+'?'+str(random.random()), width=1200)


# In[13]:


Image(url='../img/elmo_4.png'+'?'+str(random.random()), width=1200)


# Peters et al., “Deep contextualized word representations” (2018)

# In[14]:


Image(url='../img/elmo_5.png'+'?'+str(random.random()), width=1200)


# ## Problems with using Bi-LSTMs to learn contextual word embeddings ##
# 
# * Recurrent neural networks are difficult to train
#     * Vanishing gradients (LSTM; Hochreiter & Schmidhuber 1997)
#     * Exploding gradients (norm rescaling; Pascanu et al. 2013)

# ## Proposal: Transformer Networks ##
# 
# * **Idea 1**: Encode words individually with feed-forward neural networks
#     * Shorter path for gradient back-propagation
#     * Parallel computation possible
# * **Idea 2**: Replace recurrence function with a **positional encoding**
#     * Fixed-length vectors, similar word embeddings, that represents the position

# 
# * Current base architecture used for all state-of-the-art NLP models
#     * Yes, also GPT-3
# * You will learn more about popular variants and how to train them later in the course

# ## Transformer Networks ##
# 
# * **Downside**: *Very* complex architecture
#     * We do not expect you to understand every detail of it, only the core ideas

# In[15]:


Image(url='../img/transformers.png'+'?'+str(random.random()), width=400)


# Vaswani et al. (2017), “Attention Is All You Need”. https://arxiv.org/abs/1706.03762

# ## Positional Encoding ##
# 
# * Project positions in sequence to fixed-length vectors, same dimensionality as word embeddings
# * Positional embeddings for similar positions are similar
# * Obtained using a transformation function
# * Is added to each input embedding

# In[16]:


Image(url='../img/positional_1.png'+'?'+str(random.random()), width=1200)


# ## Positional Encoding ##
# 
# Transformation function used in Vaswani et al. (2017) is static  (note: alternative is to jointly learn positional embeddings)

# In[17]:


Image(url='../img/positional_2.png'+'?'+str(random.random()), width=1200)


# Picture source: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

# <center><img src="../img/quiz_time.png"></center>

# # Summary #
# 
# * Traditional neural word representations are static
#     * Do not differ depending on context
# * Contextualised representations are dynamic
#     * Differ by context
#     * Require a function (in practice: trained model) that can return a word embedding given its context
# * Popular neural archtectures for learning contextual representations
#     * ELMo
#         * BiLSTM embeddings for words
#     * Transformers
#         * Feed-forward NNs + positional embeddings

# # Outlook #
# 
# * *Loads* of Transformer variants
#     * Near-impossible to keep up
# * In the machine translation lecture, you will be introduced to
#     * The most important one(s), including BERT
#     * How to train them
#     * How to use them in practice
#     * How to use them for cross-lingual tasks
# * Why in the machine translation lecture?
#     * Many NLP innovations originate in machine translation

# # Additional Reading #
# 
# * Blog posts ["The Illustrated Transformer"](http://jalammar.github.io/illustrated-transformer/) and ["The Illustrated BERT, ELMo, and co."](http://jalammar.github.io/illustrated-bert/) by Jay Alammar
#     * Step-by-step walk-through of architectures
#     * For those who want to get a more in-depth understanding of the architectures
