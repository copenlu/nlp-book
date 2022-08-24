#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


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


# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# # Natural Language Processing (NLP) Lecture 2:
# # Representing Words as Vectors 
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
# * Contextual representations
# * From Word representations to Sentences and Documents

# # Feed-forward Neural Networks
# <center><img src="../img/mlp.svg"></center>

# <center><img src="../img/backprop.svg"></center>

# ## Word representations ##

# ![Word representations visualised in two dimensions](../img/word_representations.svg)

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

# ## Sparse Binary Representations ##
# 
# * Map words to unique positive non-zero integers
# * $f_{id}(w) \mapsto \mathbb{N^{*}}$
# * $g(w, i) = {\left\lbrace
#     \begin{array}{ll}
#         1 & \textrm{if }~i = f_{id}(w) \\
#         0 & \textrm{otherwise} \\
#     \end{array}\right.}$
# * "One-hot" vector
# * $f_{sb}(w) = (g(w, 1), \ldots, g(w, |V|))$
# * $f_{sb}(w) \mapsto \{0,1\}^{|V|}$

# ## Sparse Binary Example ##
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
# 
# Note the different formulation in SciPy: $cos(u, v) = 1 - \frac{u \cdot v}{||u|| ||v||}$

# ## Cosine Similarity Visualised ##
# 
# <center><img src="http://blog.christianperone.com/wp-content/uploads/2013/09/cosinesimilarityfq1.png" width="110%"></center>
# 
# http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/

# ## Sparse Binary Similarities ##
# 
# * $cos(f_{sb}(\textrm{apple}), f_{sb}(\textrm{rabbit})) = 0$
# * $cos(f_{sb}(\textrm{apple}), f_{sb}(\textrm{orange})) = 0$
# * $cos(f_{sb}(\textrm{orange}), f_{sb}(\textrm{rabbit})) = 0 $

# ## Dense Continuous Representations ##

# ## Dense Continuous Representations ##
# 
# * $f_{id}(w) \mapsto \mathbb{N}^{*}$
# * "Embed" words as matrix rows
# * Dimensionality: $d$ (hyperparameter)
# * $W \in \mathbb{R}^{|\mathbb{V}| \times d}$
# * $f_{dc}(w) = W_{f_{id}(w), :}$
# * $f_{dc}(w) \mapsto \mathbb{R}^{d}$

# ## Dense Continuous Example ##
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

# ## Sparse Co-occurence Example ##
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
# * Good: *cereals*
# * Bad: *airplanes*

# <center><img width=1000 src="../img/cbow_sg2.png"></center>

# ## Unsupervised Loss Function ##
# 
# * $w \in \mathbb{V}$; $c \in \mathbb{V}$
# * $D = ((c, w),\ldots)$; observed co-occurences
# * $D' = ((c, w),\ldots)$; "noise samples"
# * $\textrm{max}~p((c, w) \in D | W) - p((c, w) \in D' | W)$
# 

# ## Neural Skip-Gram Model ##
# 
# * $W^{w} \in \mathbb{R}^{|\mathbb{V}| \times d}$; $W^{c} \in \mathbb{R}^{|\mathbb{V}| \times d}$
# * $D = ((c, w),\ldots)$; $D' = ((c, w),\ldots)$
# * $\sigma(x) = \frac{1}{1 + \textrm{exp}(-x)}$
# * $p((c, w) \in D | W^{w}, W^{c}) = \sigma(W^{c}_{f_{id}(c),:} \cdot W^{w}_{f_{id}(w),:})$
# * $\arg\max\limits_{W^{w},W^{c}} \sum\limits_{(w,c) \in D} \log \sigma(W^{c}_{f_{id}(c),:} \cdot W^{w}_{f_{id}(w),:}) \\ + \sum\limits_{(w,c) \in D'} \log \sigma(-W^{c}_{f_{id}(c),:} \cdot W^{w}_{f_{id}(w),:})$
# 

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
# 
# <center><img src="../img/word_representations.svg" width="60%"></center>
# 
# * Dimensionality reduction using [t-SNE](https://lvdmaaten.github.io/tsne/)

# ## Neural Representations Visualised (zoomed) ##
# 
# ![Word representations visualised in two dimensions, zoomed in on a small cluster](../img/word_representations_zoom.svg)
# 
# * Dimensionality reduction using [t-SNE](https://lvdmaaten.github.io/tsne/)

# <center><img src="../img/quiz_time2.png"></center>

# ## Word Representation Algebra ##
# 
# * $f_{n}(\textrm{king}) - f_{n}(\textrm{man}) + f_{n}(\textrm{woman}) \approx f_{n}(\textrm{queen})$
# * $f_{n}(\textrm{Paris}) - f_{n}(\textrm{France}) + f_{n}(\textrm{Italy}) \approx f_{n}(\textrm{Rome})$
# 
# <center><img src="../img/regularities.png" width="130%"></center>

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
# 
# <center><img src="../img/sum_embedding.jpg" width="30%"></center>
# 

# # From Words to Sentences to Documents
# 
# * Standard pooling approaches of word embeddings
#     * Sum, Mean, Max
#     * TF-IDF weighting
# 
# <center><img src="../img/weighted_embedding.jpg" width="30%"></center>
# 

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
# 
# 
# <center><img src="../img/doc2vec_0.png" width=60%></center>
# 
# 
# Source: https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e

# # Doc2vec
# 
# 
# * Simple extension of word2vec (cbow)
# * Paragraph vector
# 
# 
# <center><img src="../img/doc2vec_1.png" width=70%></center>
# 
# 
# Source: https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e

# # Contextualised Representations #
# 
# * Standard embedding representations have *one* representation per word, regardless of the *current* context
# 
# * Contextualised Representations use the context surrounding the word
# 

# ## Contextualised Representations Example ##
# 
# 
# * "Yesterday I saw a bass ..."
# 
# <p float="left">
#   <img src="../img/bass_1.jpg" width="300" />
#   <img src="../img/bass_2.svg" width="100" /> 
# </p>

# ## Contextualised Representations Example ##
# 
# 
# * a) "Yesterday I saw a bass swimming in the lake"
# * b) "Yesterday I saw a bass in the music shop"
# 
# 
# <p float="left">
#   <img src="../img/bass_1.jpg" width="300" />
#   <img src="../img/bass_2.svg" width="100" /> 
# </p>

# ## Contextualised Representations Example ##
# 
# 
# * a) <span style="color:red">"Yesterday I saw a bass swimming in the lake"</span>.
# * b) <span style="color:green">"Yesterday I saw a bass in the music shop"</span>.
# 
# <p float="center">
#   <img src="../img/bass_visualisation.jpg" width="500" />
# </p>

# ## What makes a good representation these days? ##
# 
# 1. Representations are **distinct**
# 2. **Similar** words have **similar** representations
# 3. Representations take **context** into account

# # Using pre-trained representations from BERT #
# 
# 
# <p float="center">
#   <img src="../img/bert_overview2.jpg" width="800" />
# </p>
# 
# Some basic code at https://huggingface.co/pytorch-transformers/
# 

# In[1]:


import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#loss, logits = outputs[:2]


# In[2]:


import torch
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP]"
tokenized_text = tokenizer.tokenize(text)

assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# In[3]:


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to desactivate the DropOut modules
# This is IMPORTANT to have reproductible results during evaluation!
model.eval()

tokens_tensor = tokens_tensor
segments_tensors = segments_tensors

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # PyTorch-Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)

sentence_representation = encoded_layers[0]


# In[4]:


encoded_layers


# <center><img src="../img/quiz_time2.png"></center>

# # Summary #
# 
# * Moving from features to "representations"
# * Representations limits what we can learn and our generalisation power
# * Many ways to learn representations (there are more than what we covered)
# * Neural representations most widely used
#     * 'static' representations are widespread
#     * Contextualised representations usually give much more expressivity
# * Different linguistic granularities - words, sentences, documents

# # Next week
# 
# * Language Modelling
# * Recurrent Neural Networks
#     * LSTMs
#     * GRUs
# * Learning Sentence Representations with RNNs
# * Applications of RNNs

# # Additional Reading #
# 
# * ["Word Representations: A Simple and General Method for Semi-Supervised Learning"](http://www.aclweb.org/anthology/P/P10/P10-1040.pdf) by Turian et al. (2010)
# * ["Representation Learning: A Review and New Perspectives"](https://arxiv.org/abs/1206.5538) by Bengio et al. (2012)
# * ["Linguistic Regularities in Continuous Space Word Representations"](http://www.aclweb.org/anthology/N/N13/N13-1090.pdf) by Mikolov et al. (2013a) ([video](http://techtalks.tv/talks/linguistic-regularities-in-continuous-space-word-representations/58471/))
# * ["Distributed Representations of Words and Phrases and their Compositionality"](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality) by Mikolov et al. (2013b)
# * ["word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method"](https://arxiv.org/abs/1402.3722) by Goldberg and Levy (2014)
# * ["Neural Word Embedding as Implicit Matrix Factorization"](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) by Levy and Goldberg (2014)
# * [Blog post explaining BERT](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) by Horev (2018)

# # Questions
