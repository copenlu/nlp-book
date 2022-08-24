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


# In[ ]:





# In[2]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[3]:


import random
from IPython.display import Image


# # Introduction to Representation Learning
# 
# 

# # Outline
# 
# * Representations of Words
#     * Motivation
#     * Sparse Binary Representations
#     * Dense Continuous Representations
# * Unsupervised Learning of Word Representations
#     * Motivation
#     * Sparse Co-occurence Representations
#     * Neural Word Representations
# * From Word Representations to Sentences and Documents

# # Feed-forward Neural Networks
# <center><img src="../img/mlp.svg"></center>

# ## Word representations ##

# In[4]:


Image(url='../img/word_representations.svg'+'?'+str(random.random()), width=1000)


# ## Why talk about representations? ##
# 
# * Machine Learning, features are representations of the input data
#     * Language is special
# * Better representations, better performance
# * Representation Learning ("Deep Learning"), trendy

# ## What makes a good representation? ##
# 
# 1. Representations are **distinct**
# 2. **Similar** words have **similar** representations

# ## Formal Task ##
# 
# * Words: $w$
# * Vocabulary: $\mathbb{V} (\forall_{i} w_{i} \in \mathbb{V})$
# * Find representation function: $f(w_{i}) = r_{i}$

# ## Sparse Binary Representations ##
# 
# * Map words to unique positive non-zero integers
#     * $f_{id}(w) \mapsto \mathbb{N^{*}}$
#     * $g(w, i) = {\left\lbrace
#     \begin{array}{ll}
#         1 & \textrm{if }~i = f_{id}(w) \\
#         0 & \textrm{otherwise} \\
#     \end{array}\right.}$
# * Word representations as "one-hot" vectors
#     * $f_{sb}(w) = (g(w, 1), \ldots, g(w, |V|))$
#     * $f_{sb}(w) \mapsto \{0,1\}^{|V|}$

# ### Example ###
# 
# * $\mathbb{V} = \{\textrm{apple}, \textrm{orange}, \textrm{rabbit}\}$
# * $f_{id}(\textrm{apple}) = 1, \ldots$, $f_{id}(\textrm{rabbit}) = 3$
# * $f_{sb}(\textrm{apple}) = (1, 0, 0)$
# * $f_{sb}(\textrm{orange}) = (0, 1, 0)$
# * $f_{sb}(\textrm{rabbit}) = (0, 0, 1)$

# ## Sparse Binary Visualised ##
# 
# ![Sparse binary representations visualised](../img/sparse_binary.svg)
# 

# ## Cosine Similarity ##
# 
# * $cos(u, v) = \frac{u \cdot v}{||u|| ||v||}$
# * $cos(u, v) \mapsto [-1, 1]$
# * $cos(u, v) = 1$; identical
# * $cos(u, v) = -1$; opposites
# * $cos(u, v) = 0$; orthogonal

# Note the different formulation in SciPy: $cos(u, v) = 1 - \frac{u \cdot v}{||u|| ||v||}$

# ## Cosine Similarity Visualised ##
# 
# <center><img src="http://blog.christianperone.com/wp-content/uploads/2013/09/cosinesimilarityfq1.png" width="110%"></center>
# 
# http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/

# In[5]:


Image(url='../img/quiz_time.png'+'?'+str(random.random()))


# Link to quiz: https://forms.gle/Atay6apGdL2ZkoAJA

# ## Sparse Binary Similarities ##
# 
# * $cos(f_{sb}(\textrm{apple}), f_{sb}(\textrm{rabbit})) = 0$
# * $cos(f_{sb}(\textrm{apple}), f_{sb}(\textrm{orange})) = 0$
# * $cos(f_{sb}(\textrm{orange}), f_{sb}(\textrm{rabbit})) = 0 $

# ## Dense Continuous Representations ##
# 
# * $f_{id}(w) \mapsto \mathbb{N}^{*}$
# * "Embed" words as matrix rows
# * Dimensionality: $d$ (hyperparameter)
# * $W \in \mathbb{R}^{|\mathbb{V}| \times d}$
# * $f_{dc}(w) = W_{f_{id}(w), :}$
# * $f_{dc}(w) \mapsto \mathbb{R}^{d}$

# ### Example ###
# 
# * $\mathbb{V} = \{\textrm{apple}, \textrm{orange}, \textrm{rabbit}\}$
# * $d = 2$
# * $W \in \mathbb{R}^{3 \times 2}$
# * $f_{id}(\textrm{apple}) = 1, \ldots, f_{id}(\textrm{rabbit}) = 3$
# * $f_{dc}(\textrm{apple}) = (1.0, 1.0)$
# * $f_{dc}(\textrm{orange}) = (0.9, 1.0)$
# * $f_{dc}(\textrm{rabbit}) = (0.1, 0.5)$

# ## Dense Continuous Visualised ##
# 
# <center><img src="../img/dense_continuous.svg" width="80%"></center>

# ## Dense Continuous Similarities ##
# 
# * $cos(f_{dc}(\textrm{apple}),f_{dc}(\textrm{rabbit})) \approx 0.83$
# * $cos(f_{dc}(\textrm{apple}),f_{dc}(\textrm{orange})) \approx 1.0$
# * $cos(f_{dc}(\textrm{orange}),f_{dc}(\textrm{rabbit})) \approx 0.86$

# # Unsupervised Learning of Word Representations #

# ## Why not supervised? ##
# 
# <center><img src="../img/annotated_vs_unannotated_data.svg" width="40%"></center>
# 
# 
# * Current gains from large data sets / more compute
#     * Lack of comparison

# ## Linguistic Inspirations ##
# 
# * "Oculist and eye-doctor … occur in almost the same environments. … If $A$ and $B$ have almost identical environments we say that they are synonyms." – Zellig Harris (1954)
# * "You shall know a word by the company it keeps." – John Rupert Firth (1957)
# * Akin to "meaning is use" – Wittgenstein (1953)

# ## Sparse Co-occurence Representations ##

# ## Co-occurences ##
# 
# * Collected from a large collection of *raw* text
# * E.g. Wikipedia, crawled news data, tweets, ...
# 
# 1. "…comparing an **apple** to an **orange**…"
# 2. "…an **apple** and **orange** from Florida…"
# 3. "…my **rabbit** is not shaped like an **orange**…" (yes, there is always **noise** in the data)
# 

# ## Sparse Co-occurence Representations ##
# 
# * The number of times words co-occur in a text collection
# * $C \in \mathbb{N}^{|V| \times |V|}$
# * $f_{id}(\textrm{apple}) = 1, \ldots, f_{id}(\textrm{rabbit}) = 3$
# * $C = \begin{pmatrix}
#         2 & 2 & 0 \\
#         2 & 3 & 1 \\
#         0 & 1 & 1 \\
#     \end{pmatrix}$
# * $f_{cs}(w) = C_{f_{id}(w), :}$
# * $f_{cs}(w) \mapsto \mathbb{N}^{|V|}$

# ### Example ###
# 
# * $\mathbb{V} = \{\textrm{apple}, \textrm{orange}, \textrm{rabbit}\}$
# * $f_{id}(\textrm{apple}) = 1, \ldots, f_{id}(\textrm{rabbit}) = 3$
# * $f_{cs}(\textrm{apple}) = (2, 2, 0)$
# * $f_{cs}(\textrm{orange}) = (2, 3, 1)$
# * $f_{cs}(\textrm{rabbit}) = (0, 1, 1)$

# ## Sparse Co-occurence Similarities ##
# 
# * $cos(f_{cs}(\textrm{apple}), f_{cs}(\textrm{rabbit})) \approx 0.50$
# * $cos(f_{cs}(\textrm{apple}), f_{cs}(\textrm{orange})) \approx 0.94$
# * $cos(f_{cs}(\textrm{orange}), f_{cs}(\textrm{rabbit})) \approx 0.76$

# # Neural Word Representations #

# ## Learning by Slot Filling ##
# 
# * "…I had some **_____** for breakfast today…"
# * Good: *cereal*
# * Bad: *airplanes*

# <center><img width=1000 src="../img/cbow_sg2.png"></center>

# ## Creating Positive Training Instances ##
# 
# 
# "I had some cereal for breakfast today"
# 
# * **I** had some cereal for breakfast today -> (**I**, had); (**I**, some)
# * I **had** some cereal for breakfast today -> (**had**, I); (**had**, some); (**had**, cereal)
# * I had **some** cereal for breakfast today -> (**some**, I); (**some**, had); (**some**, cereal); (**some**, for)
# * I had some **cereal** for breakfast today -> (**cereal**, had); (**cereal**, some); (**cereal**, for); (**cereal**, breakfast)
# * ...

# Training instance: target word $w \in \mathbb{V}$; context word $c \in \mathbb{V}$
# 
# $D = ((c, w),\ldots)$; observed co-occurences
# 

# ## Sampling Negative Training Instances ##
# 
# 
# "I had some cereal for breakfast today"
# 
# * **Lecture** had some cereal for breakfast today -> (**Lecture**, had); (**Lecture**, some)
# * I **computer** some cereal for breakfast today -> (**computer**, I); (**computer**, some); (**computer**, cereal)
# * I had **word** cereal for breakfast today -> (**word**, I); (**word**, had); (**word**, cereal); (**word**, for)
# * I had some **books** for breakfast today -> (**books**, had); (**books**, some); (**books**, for); (**books**, breakfast)
# * ...

# 
# Create $D' = ((c, w),\ldots)$; "noise samples"

# ## Unsupervised Loss Function ##
# 
# * $w \in \mathbb{V}$; $c \in \mathbb{V}$
# * $D = ((c, w),\ldots)$; observed co-occurences
# * $D' = ((c, w),\ldots)$; "noise samples"
# * $\textrm{max}~p((c, w) \in D | W) - p((c, w) \in D' | W)$

# * Idea: maximise difference between the score for true instances and negative samples
# * How do we get $p(c, w)$?
#    * using a neural network

# ## Neural Skip-Gram Model ##
# 
# <center><img width=650 src="../img/skip-gram_architecture.png"></center>

# * $W^{w} \in \mathbb{R}^{|\mathbb{V}| \times d}$; $W^{c} \in \mathbb{R}^{|\mathbb{V}| \times d}$
# * $D = ((c, w),\ldots)$; $D' = ((c, w),\ldots)$
# * $\sigma(x) = \frac{1}{1 + \textrm{exp}(-x)}$
# * $p((c, w) \in D | W^{w}, W^{c}) = \sigma(W^{c}_{f_{id}(c),:} \cdot W^{w}_{f_{id}(w),:})$
# * $\arg\max\limits_{W^{w},W^{c}} \sum\limits_{(w,c) \in D} \log \sigma(W^{c}_{f_{id}(c),:} \cdot W^{w}_{f_{id}(w),:})  + \sum\limits_{(w,c) \in D'} \log \sigma(-W^{c}_{f_{id}(c),:} \cdot W^{w}_{f_{id}(w),:})$

# ## Neural Representation ##
# 
# * Learned using [word2vec](https://code.google.com/p/word2vec/)
# * Google News data, $~1,000,000,000$ words
# * $|\mathbb{V}| = 3,000,000$
# * $d = 300$

# ## Neural Representation Example ##
# 
# * $f_{n}(\textrm{apple}) = (-0.06, -0.16, \ldots, 0.34)$
# * $f_{n}(\textrm{orange}) = (-0.10, -0.18, \ldots, 0.08)$
# * $f_{n}(\textrm{rabbit}) = (0.02, 0.11, \ldots, 0.11)$

# ## Neural Representation Similarities ##
# 
# * $cos(f_{n}(\textrm{apple}), f_{n}(\textrm{rabbit})) \approx 0.34$
# * $cos(f_{n}(\textrm{apple}), f_{n}(\textrm{orange})) \approx 0.39$
# * $cos(f_{n}(\textrm{orange}), f_{n}(\textrm{rabbit})) \approx 0.20$

# ## Neural Representations Visualised ##

# In[6]:


Image(url='../img/word_representations.svg'+'?'+str(random.random()), width=1200)


# * Dimensionality reduction using [t-SNE](https://lvdmaaten.github.io/tsne/)

# ## Neural Representations Visualised (zoomed) ##

# In[7]:


Image(url='../img/word_representations_zoom.svg'+'?'+str(random.random()), width=1200)


# * Dimensionality reduction using [t-SNE](https://lvdmaaten.github.io/tsne/)

# In[8]:


Image(url='../img/quiz_time.png'+'?'+str(random.random()))


# Link to quiz: https://tinyurl.com/k67je5ut

# ## Word Representation Algebra ##
# 
# * $f_{n}(\textrm{king}) - f_{n}(\textrm{man}) + f_{n}(\textrm{woman}) \approx f_{n}(\textrm{queen})$
# * $f_{n}(\textrm{Paris}) - f_{n}(\textrm{France}) + f_{n}(\textrm{Italy}) \approx f_{n}(\textrm{Rome})$

# In[9]:


Image(url='../img/regularities.png'+'?'+str(random.random()), width=1000)


# # Granularities of representations #
# 
# ## Output
# * Word representations
# * Sentence representations
# * Document representations
# 
# ## Input
# * Words
# * Characters

# # From Words to Sentences to Documents
# 
# * Standard pooling approaches of word embeddings
#     * Sum, Mean, Max

# In[10]:


Image(url='../img/embedding_sum.jpg'+'?'+str(random.random()), width=500)


# # From Words to Sentences to Documents
# 
# * Standard pooling approaches of word embeddings
#     * Sum, Mean, Max
#     * TF-IDF weighting

# In[11]:


Image(url='../img/embedding_weighted.jpg'+'?'+str(random.random()), width=500)


# # Building Sentence and Document Representations
# 
# * Sentence representations from scratch (e.g. with RNNs in next lecture)
# 
# * Doc2vec

# # Doc2vec
# 
# 
# * Simple extension of word2vec (cbow)
# * Paragraph vector

# In[12]:


Image(url='../img/doc2vec_0.png'+'?'+str(random.random()), width=1000)


# Source: https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e

# # Doc2vec
# 
# 
# * Simple extension of word2vec (cbow)
# * Paragraph vector

# In[13]:


Image(url='../img/doc2vec_1.png'+'?'+str(random.random()), width=1000)


# Source: https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e

# # Summary #
# 
# * Moving from features to "representations"
# * Representations limits what we can learn and our generalisation power
# * Many ways to learn representations (there are many more than what we covered)
# * Neural representations most widely used
# * Different linguistic granularities - words, sentences, documents

# # Additional Reading #
# 
# * ["Word Representations: A Simple and General Method for Semi-Supervised Learning"](http://www.aclweb.org/anthology/P/P10/P10-1040.pdf) by Turian et al. (2010)
# * ["Representation Learning: A Review and New Perspectives"](https://arxiv.org/abs/1206.5538) by Bengio et al. (2012)
# * ["Linguistic Regularities in Continuous Space Word Representations"](http://www.aclweb.org/anthology/N/N13/N13-1090.pdf) by Mikolov et al. (2013a) ([video](http://techtalks.tv/talks/linguistic-regularities-in-continuous-space-word-representations/58471/))
# * ["Distributed Representations of Words and Phrases and their Compositionality"](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality) by Mikolov et al. (2013b)
# * ["word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method"](https://arxiv.org/abs/1402.3722) by Goldberg and Levy (2014)
# * ["Neural Word Embedding as Implicit Matrix Factorization"](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) by Levy and Goldberg (2014)
# * [Demystifying Neural Network in Skip-Gram Language Modeling](https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling)

# # Questions
