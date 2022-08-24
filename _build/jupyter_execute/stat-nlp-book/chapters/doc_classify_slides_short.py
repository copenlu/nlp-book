#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# %cd .. \nimport sys\nsys.path.append("..")\nimport random\nimport statnlpbook.util as util\nimport matplotlib\nmatplotlib.rcParams[\'figure.figsize\'] = (10.0, 5.0)\nimport random \nfrom collections import defaultdict\nrandom.seed(2)\n')


# <!---
# Latex Macros
# -->
# $$
# \newcommand{\Xs}{\mathcal{X}}
# \newcommand{\Ys}{\mathcal{Y}}
# \newcommand{\y}{\mathbf{y}}
# \newcommand{\weights}{\mathbf{w}}
# \newcommand{\balpha}{\boldsymbol{\alpha}}
# \newcommand{\bbeta}{\boldsymbol{\beta}}
# \newcommand{\aligns}{\mathbf{a}}
# \newcommand{\align}{a}
# \newcommand{\source}{\mathbf{s}}
# \newcommand{\target}{\mathbf{t}}
# \newcommand{\ssource}{s}
# \newcommand{\starget}{t}
# \newcommand{\repr}{\mathbf{f}}
# \newcommand{\repry}{\mathbf{g}}
# \newcommand{\bar}{\,|\,}
# \newcommand{\x}{\mathbf{x}}
# \newcommand{\prob}{p}
# \newcommand{\Pulp}{\text{Pulp}}
# \newcommand{\Fiction}{\text{Fiction}}
# \newcommand{\PulpFiction}{\text{Pulp Fiction}}
# \newcommand{\pnb}{\prob^{\text{NB}}}
# \newcommand{\vocab}{V}
# \newcommand{\params}{\boldsymbol{\theta}}
# \newcommand{\param}{\theta}
# \DeclareMathOperator{\perplexity}{PP}
# \DeclareMathOperator{\argmax}{argmax}
# \DeclareMathOperator{\argmin}{argmin}
# \newcommand{\train}{\mathcal{D}}
# \newcommand{\counts}[2]{\#_{#1}(#2) }
# \newcommand{\length}[1]{\text{length}(#1) }
# \newcommand{\indi}{\mathbb{I}}
# $$

# In[3]:


get_ipython().run_cell_magic('HTML', '', '<style>\ntd,th {\n    font-size: x-large;\n    text-align: left;\n}\n</style>\n')


# # Text Classification 

# Automatically classify input text into a set of **atomic classes**

# ## Motivation
# * **Information retrieval**: classify documents into topics, such as "sport" or "business"
# * **Sentiment analysis**: classify tweets into being "positive" or "negative"  
# * **Spam filters**: distinguish between "ham" and "spam"
# 
# <!-- TODO: Load Web Corpus, 4 Universities, something were Maxent works -->

# ## Text Classification as Structured Prediction
# Simplest instance of [structured prediction](/template/statnlpbook/02_methods/00_structuredprediction) 
# 
# * Input space $\Xs$ are sequences of words
# * output space $\Ys$ is a set of labels
#     * Example for document classification: $\Ys=\{ \text{sports},\text{business}\}$ 
#     * Example for sentiment prediction: $\Ys=\{ \text{positive},\text{negative}, \text{neutral}\}$ 
# * model $s_{\params}(\x,y)$ 
#     * scores $y$ highly if it fits text $\x$
#     

# ## This Lecture
# 
# * **Discriminative** model: *Logistic Regression*
#     * Learn how to *discriminate best output based on input*

# ## Sentiment Analysis as Text Classification
# Let us focus on a specific task: sentiment analysis
# 
# * load data for this task from the [Movie Review dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/)

# In[4]:


from os import listdir
from os.path import isfile, join
def load_from_dir(directory,label):
    """
    Load documents from a directory, and give them all the same label `label`.
    Params:
        directory: the directory to load the documents from.
        label: the label to assign to each document.
    Returns:
        a list of (x,y) pairs where x is a tokenised document (list of words), and y is the label `label`.
    """
    result = []
    for file in listdir(directory):
        with open(directory + file, 'r') as f:
            text = f.read()
            tokens = [t.strip() for t in text.split()]
            result.append((tokens,label))
    return result
 
data_pos = load_from_dir('../data/rt-2k/txt_sentoken/pos/', 'pos') 
data_neg = load_from_dir('../data/rt-2k/txt_sentoken/neg/', 'neg')
    
data_all = data_pos + data_neg


# Let us look at some example data ...

# In[5]:


" ".join(data_pos[12][0][:200])


# ## Setup
# 
# Divide dataset in **train**, **test** and **development** set

# In[6]:


random.seed(0)
shuffled = list(data_all)
random.shuffle(shuffled)
train, dev, test = shuffled[:1600], shuffled[1600:1800], shuffled[1800:]
len([(x,y) for (x,y) in train if y == 'pos']) # check balance 


# ## Discriminative Text Classification

# How can we set probabilities algorithmically?

# Frame task as an optimisation problem and 
# 
# maximise $\prob(+ \bar \text{...Pulp Fiction...})$ directly

# ### Maximum Conditional Log-Likelihood
# 
# Directly optimise the **conditional likelihood** of the correct labels
# 
# $$
# \mathit{CL}(\params) = \sum_{(\x,y) \in \train} \log(\prob_\params(y|\x)) = \sum_{(\x,y) \in \train} \log\left(\frac{\prob_\params(y,\x)}{\sum_y \prob_\params(y,\x)}\right)
# $$

# ### Feature Function
# 
# For log-linear models we need **feature functions** $f_i(\x)$ mapping input to a real value, e.g.
# 
# $$
# f_{\text{Pulp}}(\x) = \counts{\x}{\text{Pulp}}
# $$
# i.e., the number of times *Pulp* appears in the input $\x$.  

# In[7]:


p,f = "Pulp", "Fiction"
data = [((p,f),True),((p,f),True),((p,f),True), ((f,),False), ((f,),False),((p,f),False)]
from math import log, exp

def cl(data, w_p_true, w_f_true, w_true, l=0.0):
    loss = 0.0
    for x,y in data:
        count_p = len([w for w in x if w==p])
        count_f = len([w for w in x if w==f])
        score_true = w_true + count_p * w_p_true + count_f * w_f_true
        log_z = log(exp(score_true) + exp(0))
#         print(count_p)
#         print(count_f)
#         print(log_z)
        if y:
            loss += score_true - log_z
        else:
            loss += 0 - log_z
    return loss - l * (w_p_true * w_p_true + w_f_true * w_f_true + w_true * w_true)

def jl(data, w_p_true, w_f_true, w_true, w_p_false, w_f_false, w_false):
    loss = 0.0
    for x,y in data:
        count_p = len([w for w in x if w==p])
        count_f = len([w for w in x if w==f])
        score_true = w_true + count_p * w_p_true + count_f * w_f_true
        score_false = w_false + count_p * w_p_false + count_f * w_f_false
        if y:
            loss += score_true
        else:
            loss += score_false
    return loss 


# cl(data,log(0.5),log(0.5),log(0.5))

import matplotlib.pyplot as plt
#import mpld3
import numpy as np

# x = np.linspace(-10, 10, 100)
# cl_loss = np.vectorize(lambda w: cl(data,w,log(0.5),log(0.5),1))


# ## Optimising the Conditional Loglikelihood
# No closed form solution, use **iterative methods** such as 
# 
# * (Stochastic) Gradient Descent
# * L-BFGS (quasi-Newton)

# "Easy" because **concave** in weights $\weights$

# In[8]:


x = np.linspace(-10, 10, 100)
C = 1
cl_loss = np.vectorize(lambda w: cl(data,w,log(0.5),log(0.5),1/C))
plt.plot(x, cl_loss(x)) 


# ### How to Find Optimum?
# 
# In practice one of the following:
# * implement both optimisation and gradient calculation
# * use off-the-shelf gradient descent library, provide the gradients (Example: [factorie](http://factorie.cs.umass.edu/), [scipy](https://www.scipy.org/))
# * You have a back-propagation framework (such as Tensorflow) 
# * You have a library (such as scikit-learn) for a specific model class 
# 
# Here: [scikit-learn](http://scikit-learn.org/)
# 

# ## Regularisation
# Estimating a log-linear models can lead to *overfitting*
# 
# * For example, reviews of movies may often contain the **name** of the movie to reviewed.
# * False *killer feature*: only appears in one training instance 
# * weight can be set in way that we get perfect accuracy on this instance 

# *regularise* the model by **penalising** large weights:
# 
# * add a regularisation penalty to the training objective
# * For example,  $||\weights||_2$ and $||\weights||_1$, the L2 and L1 norm of the weight vector
# 
# $$
# \mathit{RCL}_C(\weights) = \sum_{(\x,y) \in \train} \log(\prob_\weights(y|\x)) - \frac{1}{C} ||\weights||_2
# $$
# 
# $C$ controls inverse strength of regularisation

# Both L1 and L2 have their strength and weaknesses. 
# * L1 regularisation can lead to sparse vectors with zero weights, and this can improve the memory footprint of your model. 
# * L2 seems to often lead to [better results](http://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html#l1_regularized_classification). 
# * More musings on [differences between L1 and L2 regularisation](https://explained.ai/regularization/L1vsL2.html)

# ## Logistic Regression Toolkits
# A log-linear model trained by maximising the CL corresponds to training a **logistic regression** model

# [logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) implementation of [scikit-learn](http://scikit-learn.org/stable/index.html)

# Convert $\x \in \Xs$ to a (sparse) feature vector $\mathbf{f}(\x)$:

# In[9]:


def feats(x):
    result = defaultdict(float)
    for w in x:
        result[w] += 1.0
    return result

feats(['pulp','fiction','fiction'])


# Apply to training and test instances:

# In[31]:


from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer()

train_X = vectorizer.fit_transform([feats(x) for x,_ in train])
dev_X = vectorizer.transform([feats(x) for x,_ in dev])
dev_X


# scikit-learn prefers numbers as classes:
# 
# * $\text{positive}\rightarrow 0$
# * $\text{negative}\rightarrow 1$

# In[13]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

train_Y = label_encoder.fit_transform([y for _,y in train])
dev_Y = label_encoder.transform([y for _,y in dev])

dev_Y[:10] 


# Train the logistic regression with l1 regularisation $C=1000$

# In[14]:


from sklearn.linear_model import LogisticRegression
import numpy as np

lr = LogisticRegression(C=1000, penalty="l1", random_state=1, solver='liblinear')
lr.fit(train_X, train_Y)


# Weights learned:
# * single weight vector $\weights=\weights_\text{positive} - \weights_\text{negative}$

# In[17]:


weights = vectorizer.inverse_transform(lr.coef_)[0]
sorted_weights = sorted(weights.items(), key=lambda t: t[1])
util.plot_bar_graph([w for _,w in sorted_weights[:20]],
                    [f for f,_ in sorted_weights[:20]],rotation=45)


# More obvious **discriminative** features for the negative class 
# 
# * Conditional log-likelihood down-weighs a feature $f_i$ for $y=-$ 
#     * if $f_i$ is active in $\x$
#     * but gold label is $y=+$
# 
# 

# Positive weights? 

# In[18]:


util.plot_bar_graph([w for _,w in sorted_weights[-20:]],
                    [f for f,_ in sorted_weights[-20:]],rotation=45)


# ## Evaluation
# 
# How can we quantify how well our models do?

# Use **accuracy**, the ratio of the 
# 
# * number of correct predictions and the 
# * number of all predictions  
# 
# $\y^*$ is predicted sequence of labels, $\y$ the true labels:

# $$
# \mathrm{Acc}(y_1,\ldots,y_n, y^*_1,\ldots,y^*_n) = \frac{\sum_i \indi [ y_i = y^*_i]}{n}
# $$

# In[19]:


def accuracy(gold, guess):
    correct = 0
    for y_gold,y_guess in zip(gold,guess):
        if y_guess == y_gold:
            correct += 1
    return correct / len(gold)

train_Y_lex = [y for _,y in train]
dev_Y_lex = [y for _,y in dev]


# In[20]:


lr = LogisticRegression(C=1000, penalty="l2",random_state=1, tol=0.00001, solver='liblinear')
lr.fit(train_X, train_Y)
lr_guess = label_encoder.inverse_transform(lr.predict(dev_X))
accuracy(dev_Y_lex, lr_guess)


# ## More on Feature Functions
# 
# Reminder: For log-linear models we need **feature functions** $f_i(\x)$ mapping input to a real value, e.g.
# 
# $$
# f_{\text{Pulp}}(\x) = \counts{\x}{\text{Pulp}}
# $$
# i.e., the number of times *Pulp* appears in the input $\x$.  

# How to get a better feature function?

# ### Stop Words
# 
# Non-content words

# In[22]:


import string

stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', '\n', 'the'] + list(string.punctuation))


# Shouldn't the model have ignored those since they don't discriminate between document classes? 
# Let's see what weights our model learned for non-content words

# In[23]:


stopword_weights = [(w, weight) for w, weight in sorted_weights if w in stop_words]

# Highest negative stopword weights
util.plot_bar_graph([w for _,w in stopword_weights[:20]],
                    [f for f,_ in stopword_weights[:20]],rotation=45)


# In[24]:


# Highest positive stopword weights
util.plot_bar_graph([w for _,w in stopword_weights[-20:]],
                    [f for f,_ in stopword_weights[-20:]],rotation=45)


# Why does this happen?
# 
# Discuss and enter your answer(s) here: https://tinyurl.com/y6ycc6pz

# Let's remove stopwords from the data and re-train our model

# In[25]:


def filter_dataset(data):
    """
    Removes stop words from a dataset of (x,y) pairs.
    """
    return [([w for w in x if w not in stop_words],y) for x,y in data]

train_filtered = filter_dataset(train)
dev_filtered = filter_dataset(dev)
test_filtered = filter_dataset(test)

train_X_filtered = vectorizer.fit_transform([feats(x) for x,_ in train_filtered])
dev_X_filtered = vectorizer.transform([feats(x) for x,_ in dev_filtered])
dev_X_filtered


# In[26]:


lr = LogisticRegression(C=1000, penalty="l2",random_state=1, tol=0.00001, solver='liblinear')
lr.fit(train_X_filtered, train_Y)
lr_guess = label_encoder.inverse_transform(lr.predict(dev_X_filtered))
accuracy(dev_Y_lex, lr_guess)


# ## Bigram Representation
# 
# Features don't have to correspond to single words
# $$
# f_{\text{Pulp Fiction}}(\x) = \counts{\x}{\text{Pulp Fiction}}
# $$
# 
# Use a **bag of bigrams**
# 
# * Count bigrams (like Pulp Fiction) and not just single words

# features don't even have to correspond to n-grams
# $$
# f_{\text{RT}}(\x) = \text{Lowest rotten tomatoes score of all mentioned movies in }\x
# $$

# ## Summary
# 
# * Text classification can be framed as **discriminative** learning problem
#     * Learn how to *discriminate best output based on input*
#     * Increase relative/conditional probability of classes
# * Ingredients
#     * Model (commonly used: logistic regression)
#     * Feature function
#     * Regularisation
# * Evaluation
#     * Quantitative evaluation: measure accuracy
#     * Qualitative evaluation: inspect feature weights

# ### Preview: Next Two Lectures
# 
# * More on the feature function
# * *Learn* features as opposed to manually defining them (**representation learning**)
#     * Benefit: often better performance, less feature engineering
#     * Downside: black box models -- harder to inspect model and interpret feature weights
#     * De-facto approach to NLP currently
#     * Very popular active research topic -- basics here, more in Advanced Topics in NLP course

# ## Background Material
# * Longer version of lecture on text classification: [notes](chapters/doc_classify.ipynb), [slides](chapters/doc_classify_slides.ipynb)
# * [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://www.aclweb.org/anthology/P12-2018), Sida Wang and Christopher D. Manning, ACL 2012 
# * [Thumbs up? Sentiment Classification using Machine Learning
# Techniques](http://www.cs.cornell.edu/home/llee/papers/sentiment.pdf), Bo Pang, Lillian Lee and Shivakumar Vaithyanathan, EMNLP 2002
# * Jurafsky & Martin, [Speech and Language Processing (Third Edition)](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf): Chapter 4, Naive Bayes and Sentiment Classification + Chapter 5, Logistic Regression
