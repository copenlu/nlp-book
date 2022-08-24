#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# <!---
# Latex Macros
# -->
# $$
# \newcommand{\Xs}{\mathcal{X}}
# \newcommand{\Ys}{\mathcal{Y}}
# \newcommand{\y}{\mathbf{y}}
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
# \newcommand{\x}{\mathbf{x}}
# \newcommand{\prob}{p}
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

# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# %cd .. \nimport sys\nsys.path.append("..")\nimport statnlpbook.util as util\n')


# <!---
# Latex Macros
# -->
# $$
# \newcommand{\Xs}{\mathcal{X}}
# \newcommand{\Ys}{\mathcal{Y}}
# \newcommand{\y}{\mathbf{y}}
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
# \newcommand{\x}{\mathbf{x}}
# \newcommand{\X}{\mathbf{X}}
# \newcommand{\parents}{\mathrm{par}}
# \newcommand{\dom}{\mathrm{dom}}
# \newcommand{\prob}{p}
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
# \newcommand{\duals}{\boldsymbol{\lambda}}
# \newcommand{\lagrang}{\mathcal{L}}
# $$

# # Maximum Likelihood Estimator
# 
# The Maximum Likelihood Estimator (MLE) is one of the simplest ways, and often most intuitive way, to determine the parameters of a probabilistic models based on some training data. Under favourable conditions the MLE has several useful properties. On such property is consistency: if you sample enough data from a distribution with certain parameters, the MLE will recover these parameters with arbitrary precision. In our [structured prediction recipe](structured_prediction.ipynb) MLE can be seen as the most basic form of continuous optimization for parameter estimation. 
# 
# In this section we will focus on MLE for _discrete distributions_ and _continuous parameters_ as this is the most relevant scenario within NLP. We will assume a distribution corresponding to a discrete *Bayesian Network*. Here we have a sequence of random variables $\X = \{X_1,\ldots, X_n\}$. For each variable $X_i$ we know its *parents* $\parents(i) \subseteq \{1 \ldots n\}$, and we are given a *conditional probability table* (CPT) $\prob_{i,\params}(x_i|\x_{\parents(i)})=\param^i_{x_i|\x_{\parents(i)}}$. Here $x_i$ is the state of $X_i$ and $x_{\parents(i)}$ the state of the variables $\{X_i:i \in \parents(i)\}$. If the graph induced by the $\parents$ relation is acyclic the we can define a probability distribution $\prob_\params$ over $\X$ as follows:
# 
# \begin{equation}
#   \prob_\params(\x) = \prod_{i \in \{ 1 \ldots n \}} \prob_{i,\params}(x_i|\x_{\parents(i)}) 
#                     = \prod_{i \in \{ 1 \ldots n \}} \param^i_{x_i|\x_{\parents(i)}}.
# \end{equation}
# 
# Notice that in practice we will often want to *tie* the parameters of individual CPTs, such that $\param^i_{x|\x'}=\param^j_{x|\x'}$ for certain pairs $(i,j)$. The following exposition ignores this but it is easy to generalise the findings.
# 
# To make this more concrete, consider the following simple distribution over natural language bigrams. Let $\X=\{X_1, X_2\}$ where $X_1\in\{\text{healthy},\text{fatty}\}$ is a random variable representing the first word of the bigram, and $X_2\in\{\text{pizza},\text{rice}\}$ a random variable representing the second word. We set $\parents(1)=\emptyset$ and $\parents(2)=\{1\}$ to indicate that the second word depends on first word.
# 
# Let us define this model in Python code together with some example parameters $\params$.

# In[3]:


def prob_1(x1, theta_1):
    return theta_1[x1]

def prob_2(x1, x2, theta_2):
    return theta_2[x1,x2]

def prob(x, theta):
    x1, x2 = x
    theta_1, theta_2 = theta
    return prob_1(x1, theta_1) * prob_2(x1,x2, theta_2)

x_1_domain = ['greasy', 'healthy']
x_2_domain = ['pizza', 'rice']

g, h = x_1_domain
p, r = x_2_domain

theta_1 = {g: 0.3, h: 0.7}
theta_2 = {(g,  p): 0.8,  (g, r): 0.2,
           (h,  p): 0.1,  (h, r): 0.9}
theta = theta_1, theta_2

prob((h,p), theta), prob((h,r), theta)


# Let us assume that we are given some training data \\(\train = (\x_1,\ldots, \x_m)\\), and that this data is independently and identically distributed (IID) with respect to a $\prob_\params$ distribution, that is: 
# 
# $$
#   \prob_\params(\train) = \prod_{\x \in \train} \prob_\params(\x). 
# $$
# 
# The Maximum Likelihood estimate \\(\params^*\\) is then defined as the solution to the following optimization problem:
# 
# \begin{equation}
#   \params^* = \argmax_{\params} \prob_\params(\train) = \argmax_{\params} \sum_{\x \in \train} \log \prob_\params(\x) 
# \end{equation}
# 
# In words, the maximum likelihood estimate are the parameters that assign maximal probability to the training sample. Here the second equality stems from the IID assumption and the monotonicity of the \\(\log\\) function. The latter is useful because the \\(\log\\) expression is easier to optimize. The corresponding objective 
# 
# $$
# L_\params(\train) = \log \prob_\params(\train) = \sum_{\x \in \train} \log \prob_\params(\x) = \sum_{\x \in \train} \log \prob_\params(\x)
# $$  
# 
# is called the *log-likelihood* of the data sample $\train$.
# 
# Let us write down this objective in Python, for our running example defined above. Notice that we normalise the objective by the size of the data. This will make it easier to compare objective values for different datasets, and does not arguments that maximise the objectives.

# In[4]:


from math import log
def ll(data, theta):
    return sum([log(prob(x, theta)) for x in data]) / len(data)

ll([(g,p),(h,r)], theta)


# As we will show below, the MLE can be calculated in closed form for this type of model. Roughly speaking, the solution amounts to counting how often a certain child value $x_i$ has been seen together with a certain parents value $\x_{\parents(i)}$, normalised by how often the parents value $\x_{\parents(i)}$ has been seen.
# 
# More formally, we have:
# 
# \begin{equation}
#   \param^i_{x|\x} = \frac{\counts{\train}{ x,i,\x',\parents(i)}}{\counts{\train}{\x', \parents(i)}}
# \end{equation}
# 
# Here 
# 
# $$
# \counts{\train}{x,i,\x',\mathbf{j}} = \sum_{\x \in \train} \indi(x_i = x \wedge \x_{\mathbf{j}} = \x')
# $$ 
# 
# is number of times $X_i$ was in state $x$ while $\X_{\parents(i)}$ was in state $\x'$. Likewise, 
# 
# $$
# \counts{\train}{\x',\mathbf{j}} = \sum_{\x \in \train} \indi(\x_{\mathbf{j}} = \x')
# $$ 
# 
# is the number of times $\X_{\parents(i)}$ was in state $\x'$. 
# 
# Let us calculate this solution for our running example in Python.

# In[5]:


from collections import defaultdict
def mle(data):
    counts = defaultdict(float)
    norm = defaultdict(float)
    for x1, x2 in data:
        counts[1, x1] += 1.0
        norm[1] += 1.0
        counts[2, x1, x2] += 1.0
        norm[2, x1] += 1.0
    theta_1 = dict([(w1, counts[1,w1] / norm[1]) for w1 in x_1_domain])
    theta_2 = dict([((w1,w2), counts[2,w1,w2] / norm[2,w1]) for w1 in x_1_domain for w2 in x_2_domain])
    return (theta_1, theta_2)
mle([(h,p),(g,r)])


# ## Derivation
# 
# Understanding that the MLE solution arises from optimising a mathematical objective is crucial. It not only shows that this intuitive way to set model parameters is formally grounded, it also enables us to understand counting and normalising as one instantiation of the [structured prediction](structured_prediction.ipynb) where we have some training objective defined on a training set, and determine our parameters by maximising the objective (or equivalently, minimising a loss). The MLE estimator is one of its simplest instantiations, and later we will see more complex but also often more empirically successful examples. 
# 
# Let us write out the log-likelihood further, taking into account the individual terms of the distribution:
# 
# $$
# L_\params(\train) = \sum_{\x \in \train} \log \prob_\params(\x) = \sum_{\x \in \train}\sum_{i \in 1\ldots n } \log \prob_\params(\x) = \sum_{\x \in \train}\sum_{i \in 1\ldots n } \log \param^i_{x_i|\x_{\parents(i)}}
# $$  
# 
# It is tempting to optmise this function directly, simply choosing (using some optimisation technique) the parameters $\params$ that maximise it. However, there are several constraints on $\params$ that need to be fulfilled for $\prob_\params$ to be a valid distribution. First of all, all $\param^i_{x|\x'}$ need to be non-negative. Second, for a given parent configuration $\x'$ the parameters $\param^i_{x|\x'}$ for all $x\in\dom(X_i)$ need to sum up to one. The actual, now constrained, optimisation problem we have to solve is therefore:
# 
# \begin{equation}
#   \params^* = \argmax_{\params} \sum_{\x \in \train}\sum_{i \in 1\ldots n } \log \param^i_{x_i|\x_{\parents(i)}} \\ \text{so that } \forall x \in \dom(X_i), \x' \in \dom(\X_\parents(i)): \param^i_{x|\x'} \geq 0 \,  \\
#   \text{ and } \forall \x' \in \dom(\X_\parents(i)): \sum_{x \in \mathrm{dom}(X_i)} \param^i_{x|\x'} = 1 
# \end{equation}
# 
# Notice that in the above objective no terms or constraints involve parameters $\param^i_{x|\x'}$ from different parent configurations $\x'$ or different variable indices $i$. We can hence optimise the parameters $\params^i_{\cdot|\x'}$ for each parent configuration $\x'$ and variable index $i$ in isolation. Let us hence focus on the following problem:
# 
# \begin{equation}
#   \params^{i,*}_{\cdot|\x'} = \argmax_{\params^{i}_{\cdot|\x'}} \sum_{\x \in \train \wedge \x_{\parents(i)} = \x'}\log \param^i_{x_i|\x'} \\ \text{so that } \forall x \in \dom(X_i): \param^i_{x|\x'} \geq 0 \,  \\
#   \text{ and } \sum_{x \in \mathrm{dom}(X_i)} \param^i_{x|\x'} = 1 
# \end{equation}
# 
# Let us define this sub-objective in Python for a particular variable ($X_2$) and parent configuration.

# In[6]:


def ll_2_greasy(data, greasy_theta):
    return sum([log(greasy_theta[x2]) for x1, x2 in data if x1 == g])

def mle_2_greasy(data):
    greasy_data = [x2 for x1, x2 in data if x1 == g]
    return {
        r: len([x2 for x2 in greasy_data if x2 == r]) / len(greasy_data),
        p: len([x2 for x2 in greasy_data if x2 == p]) / len(greasy_data)
    }
mle_2_greasy([(g,p),(g,p)])


# We can visualise this objective and constraint for some datasets. We will plot the value of the objective with respect to the two parameters $\param^2_{\text{rice}|\text{greasy}}$ and $\param^2_{\text{pizza}|\text{greasy}}$, and also plot the line $\param^2_{\text{rice}|\text{greasy}} + \param^2_{\text{pizza}|\text{greasy}} = 1$ to visualise the equality constraint.

# In[7]:


import matplotlib.pyplot as plt
import mpld3
import numpy as np

N = 100
eps = 0.0001
x = np.linspace(eps, 1.0 - eps, N)
y = np.linspace(eps, 1.0 - eps, N)

xx, yy = np.meshgrid(x, y)

def create_ll_plot(data):
    np_ll = np.vectorize(lambda t1,t2: ll_2_greasy(data, {r:t1, p:t2}))
    z = np_ll(xx,yy)
    fig = plt.figure()
    levels = np.arange(-2., -0.1, 0.25 )
    optimal_theta = mle_2_greasy(data)
    optimal_loss = ll_2_greasy(data, optimal_theta)
    levels_before = np.arange(optimal_loss - 2.0, optimal_loss, 0.25)
    levels_after = np.arange(optimal_loss, min(optimal_loss+2.0,-0.1), 0.25)
    contour = plt.contour(x, y, z, levels=np.concatenate([levels_before,levels_after]))
    plt.xlabel('rice')
    plt.ylabel('pizza')
    plt.plot(x,1 - x)
    plt.plot([optimal_theta[r]],[optimal_theta[p]],'ro')
    plt.clabel(contour)
    return mpld3.display(fig)

datasets = [
        [(g,p)] * 1 + [(g,r)] * 3,
        [(g,p)] * 1 + [(g,r)] * 1,
        [(g,p)] * 3 + [(g,r)] * 1
]

util.Carousel([create_ll_plot(d) for d in datasets])


# The core observation to make in these graphs is that the optimal pair of weights has to lie on the line that fulfills the equality constraint, and that at this point the gradient of the loss function is collinear to the gradient of the constraint function $g(\params^2_{\cdot|\text{greasy}})=\param^2_{\text{rice}|\text{greasy}} + \param^2_{\text{pizza}|\text{greasy}}$ (which is orthogonal to the line we see in the figure). 
# 
# This observation can be generalised: At the optimal solution to the (partial) MLE problem we require:
# 
# $$
# \frac{\partial \sum_{\x \in \train \wedge \x_{\parents(i)} = \x'}\log \param^i_{x_i|\x'} }{\partial \param^i_{x|\x'}} = \lambda \frac{\partial \sum_{x \in \mathrm{dom}(X_i)} \param^i_{x|\x'}}{\param^i_{x|\x'}} 
# $$
# 
# Taking the derivates gives:
# 
# $$
#   \frac{1}{\param^i_{x|\x'}} \sum_{\x \in \train} \indi(x_i = x \wedge \x_{\mathbf{j}} = \x') = \lambda 
# $$
# 
# and hence
# 
# $$
#   \param^i_{x|\x'} = \frac{\sum_{\x \in \train} \indi(x_i = x \wedge \x_{\mathbf{j}} = \x')}{\lambda} 
# $$
# 
# Since we need to fulfil the constraint $\sum_{x \in \mathrm{dom}(X_i)} \param^i_{x|\x'} = 1$ we have to set 
# 
# $$
# \lambda = \sum_{x \in \mathrm{dom}(X_i)}\sum_{\x \in \train} \indi(x_i = x \wedge \x_{\mathbf{j}} = \x') = \sum_{\x \in \train} \indi(\x_{\mathbf{j}} = \x').
# $$ 
# 
# This gives the counting solution we defined above. 

# ## Background Material
# * Introduction to MLE in [Mike Collin's notes](http://www.cs.columbia.edu/~mcollins/em.pdf)
