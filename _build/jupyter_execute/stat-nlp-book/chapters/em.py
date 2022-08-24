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


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# %cd .. \nimport sys\nsys.path.append("..")\nimport statnlpbook.util as util\nimport statnlpbook.em as em\nimport matplotlib.pyplot as plt\nimport mpld3\nimport numpy as np\n')


# <!---
# Latex Macros
# -->
# $$
# \newcommand{\Xs}{\mathcal{X}}
# \newcommand{\Ys}{\mathcal{Y}}
# \newcommand{\y}{\mathbf{y}}
# \newcommand{\z}{\mathbf{z}}
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

# # Expectation Maximisation Algorithm
# 
# [Maximum likelihood estimation](mle.ipynb) is an effective tool to learn model parameters when your data is fully observed. For example, when training a [language model](language_models.ipynb) we fully observe all words in the training set, and when training a [syntactic parser](parsing.ipynb) on a treebank we have access to the full parse tree of each training instance. However, in many scenarios this is not the case. When training [machine translation](word_mt.ipynb) models we often require alignments between the source sentence words and those in the target sentence. However, the standard corpora out there do not provide these alignments as they are too expensive to annotate in scale. Likewise, we might want to classify text documents into different document classes, and do have some training documents, but without annotated document classes.
# 
# Let us consider a model $\prob_\params(\x,\z)$, and a dataset $\train = \x_1 \ldots \x_n$ consisting only of $\x$ data but no information about the latent $\z$. For example, $\x$ could be a pair of sentences, and $\z$ an alignment between the sentences, as in chapter [machine translation](word_mt.ipynb). In unsupervised text classification $\x$ would be a document, and $\z=z$ a document label.
# 
# To make this more concrete, let us consider the following scenario. We would like to classify (or cluster) documents about food into two classes $\mathrm{c}_1$ and $\mathrm{c}_2$. To keep things simple, each word in the document is either $\mathrm{nattoo}$, a [healthy but slightly smelly Japanese food](https://en.wikipedia.org/wiki/Natt%C5%8D) made of fermented soybeans, $\mathrm{pizza}$ or $\mathrm{fries}$. The dataset we are looking at has a tendency to either mostly talk about healthy food (nattoo), or mostly about unhealty food (fries and pizza), and we would like our model to distinguish between these to cases. The model $\prob^\mathrm{food}_\params()$ itself is a [Naive Bayes](TODO) model and generates the words of a document independently conditioned on the document class $z\in\{\mathrm{c}_1,\mathrm{c}_2\}$:
# 
# $$
# \newcommand{\foodprob}{\prob^\mathrm{food}_\params}
# \foodprob(\x,z) = \foodprob(z) \prod_{x \in \x} \foodprob(x|z) = \theta_z \prod_{x \in \x} \theta_{x|z}
# $$
# 
# In python we can formulate this as follows. Notice how we implement the model in log-space for numerical stability.

# In[3]:


from math import log, exp

# Domains and values
z_domain = ['c1','c2']
x_domain = ['nattoo','pizza','fries']
c1, c2 = z_domain
n, p, f = x_domain

def prob(x, z, theta):
    """
    Calculate probability of p_\theta(x,z).
    Args:
        x: list of words of the document, should be in `x_domain`
        z: class label, should be in `z_domain`
    Returns:
        probability p(x,z) given the parameters.
    """
    theta_x, theta_z = theta
    bias = log(theta_z[z])
    ll = sum([log(theta_x[x_i, z]) for x_i in x])
    return exp(bias + ll)

def create_theta(prob_c1, prob_c2, 
                 n_c1, p_c1, f_c1, 
                 n_c2, p_c2, f_c2):
    """
    Construct a theta parameter vector. 
    """
    theta_z = { c1: prob_c1, c2: prob_c2}
    theta_x = { (n, c1): n_c1, (p, c1): p_c1, (f, c1): f_c1, 
                (n, c2): n_c2, (p, c2): p_c2, (f, c2): f_c2}
    return theta_x, theta_z
    

theta = create_theta(0.5,0.5, 0.3, 0.5, 0.2, 0.1, 0.4, 0.5)
prob([p,p,f], c2, theta)


# ## Marginal Log-Likelihood
# As before using our [structured prediction recipe](structured_prediction.ipynb), we would like to find good parameters $\params^*$ by defining some training objective over $\train$ and $\params$. Inspired by the [Maximum likelihood estimation](mle.ipynb) approach, a natural candidate for this objective is the *marginal* log-likelihood of the data. This likelihood arises by marginalising out the latent variable $\z$ for each training instance. Assuming again that the sample is generated IID, we get:
# 
# $$
#   M_\params(\train) = \log(\prob_\params(\train)) = \sum_{\x \in \train} \log(\sum_\z \prob_\params(\x,\z))
# $$
# 
# Unfortunately this objective has two problems when compared to the standard log-likelihood. First, there is no closed-form solution to it. In the case of the log-likelihood we could find an optimal $\params^*$ simply by counting, but no such solution exists for the marginal log-likelihood. Second, the objective is non-concave, meaning that there can be several local optima. This means that any iterative solution one can apply to maximising it (such as [SGD](sgd.ipynb)) is not guaranteed to find a globally optimal $\params^*$.   
# 
# Let us visualise this objective for our running example. We will do so by choosing two points in the parameter space $\params_1$ and $\params_2$, and then visualise the behaviour of the loss on the line between these points. 
# 
# First let us define the marginal log likelihood:

# In[4]:


def marginal_ll(data, theta):
    """
    Calculate the marginal log-likelihood of the given `data` using parameter `theta`.
    Args:
        data: list of documents, where each document is a list of words. 
        theta: parameters to use.  
    """
    return sum([log(prob(x,c1,theta) + prob(x,c2,theta)) for x in data]) / len(data)

marginal_ll([[p,p,f],[n,n]], theta)


# Let us plot the marginal log-likelihood on the line between a $\params_1$ and $\params_2$.

# In[5]:


theta1 = create_theta(0.3, 0.7, 0.0, 0.3, 0.7, 1.0, 0.0, 0.0)
theta2 = create_theta(0.3, 0.7, 1.0, 0.0, 0.0, 0.0, 0.3, 0.7)

em.plot_1D(lambda theta: marginal_ll(dataset, theta), theta1, theta2, ylim=[-8.5,-5.5])


# You can see the non-concavity of the objective, as there are two local optima. These essentially stem from the symmetry of the model: whether you call one cluster $\mathrm{c}_1$ or $\mathrm{c}_1$ will make no difference in the probability you assign to the data. 
# 
# How can we at least find one of these local optima? A classic approach to this problem relies on a deriving a lower bound to the marginal log-likelihood. 

# ## A Lower Bound on the Marginal Log-Likelihood
# Let us assume an arbitrary distribution $q(\z|\x)$. With this distribution, and using Jenssen's inequality, we can define a lower bound as follows: 
# $$
# M_\params(\train) 
#   = \sum_{\x \in \train} \log(\sum_\z \prob_\params(\x,\z)) \\ 
#   = \sum_{\x \in \train} \log(\sum_\z q(\z|\x) \frac{\prob_\params(\x,\z)}{q(\z|\x)})
#   \geq \sum_{\x \in \train} \sum_\z q(\z|\x) \log(\frac{\prob_\params(\x,\z)}{q(\z|\x)}) =: B_{q,\params}(\train).
# $$
# 
# What $q$ to choose? The one that gets $B$ maximally close to $M$. But given that $M$ is non-concave and $B_{q_\params}(\train)$ is concave, the bound cannot be tight everywhere. We need to choose a point $\params$ at which the bound is as close as possibly. Let $\params'$ be such a point. 
# 
# We want to find a $q$ that maximises $B_{q,\params'}(\train)$. Since $\prob_\params(\x,\z) = \prob_\params(\z|\x) \prob_\params(\x)$ we can maximise 
# 
# $$
# \sum_{\x \in \train} \sum_\z q(z) \log(\frac{\prob_{\params'}(\z|\x)}{q(\z)}) + \log(\prob_{\params'}(\x)).
# $$ 
# 
# The second term in the sum is constant with respect to $q$, and the first one is the negative KL divergence between $q$ and $\prob_\params(\z|\x)$. The distribution that minimises KL divergence (and hence maximises the negative KL divergence) is the distribution itself. Hence the closest lower bound can be determined by setting $q(\z|\x)=\prob_{\params'}(\z|\x)$, the conditional distribution over $\z$ based on the given parameters $\params'$. 
# 
# Let us plot this bound for given $\params'$.

# In[84]:


current_theta = add_theta(0.4, theta1, 0.6, theta2)

def calculate_class_distributions(data, theta):
    result = []
    for x in data:
        norm = prob(x,c1,theta) + prob(x,c2,theta)
        # E Step
        q = {
            c1: prob(x,c1,theta) / norm,
            c2: prob(x,c2,theta) / norm
        }
        result.append(q)
    return result

current_q = calculate_class_distributions(dataset, current_theta)

def marginal_ll_bound(data, theta, q_data = current_q):
    loss = 0.0
    for x,q in zip(data,q_data):
        loss += q[c1] * log(prob(x,c1,theta) / q[c1]) + q[c2] * log(prob(x,c2,theta) / q[c2])
    return loss / len(data)

em.plot_1D(lambda theta: marginal_ll(dataset, theta), theta1, theta2, 
           loss2=lambda theta:marginal_ll_bound(dataset,theta), ylim=[-8.5,-5.5])


# This lower bound seems to be concave, with a single maximum. In fact, it is easy to see that $B_{q_\params}(\train)$ is a weighted version of the (joint) log-likelihood as defined in the [MLE chapter](mle.ipynb). As such it has a single maximum, and we can find the optimum easily, but more on that later.
# 
# As can be seen in the figure, the bound is not just close at $\params'$, it coincides with the original objective. This can be easily shown to always hold:
# 
# $$
# \sum_\z \prob_\params(\z|\x) \log\left(\frac{\prob_\params(\x,\z)}{\prob_\params(\z|\x)}\right) = \sum_\z \prob_\params(\z|\x) \log\left(\frac{\prob_\params(\z|\x) \prob_\params(\x)}{\prob_\params(\z|\x)} \right) = \log(\prob_\params(\x)) \sum_\z \prob_\params(\z|\x) = \log(\prob_\params(\x)) )
# $$

# ## Maximising the Marginal Log-likelihood
# The fact that the lower bound coincides with the objective at the chosen point $\params'$, and that the lower bound itself is easy to optimise given that it is a weighted log-likelihood with close-form solution, suggests a simple algorithm to find a (local) optimum of the objective. Simply choose an initial $\params'$, determine the lower bound, find the optimum of this lower bound, call it $\params'$ and repeat until convergence. This algorithm is the Expectation Maximisation (EM) algorithm, and it is named that way for reasons we explain below. It is relatively easy to show that this algorithm increases the marginal likelihood in each step, see for example Mike Collin's note. It is a little more involved to show that under mild conditions this algorithm converges to a local optimum, and we point the reader to [Wu](http://web.stanford.edu/class/ee378b/papers/wu-em.pdf)'s in-depth analysis.
# 
# Before we describe the algorithm in more detail we should point out that when choosing $\params$ to optimise the bound $B_{q,\params}(\train)$ we can simplify the bound to:
# $$
# \sum_{\x \in \train} \sum_\z q(\z|\x) \log(\frac{\prob_\params(\x,\z)}{q(\z|\x)}) \propto 
# \sum_{\x \in \train} \sum_\z q(\z|\x) \log(\prob_\params(\x,\z)) = \sum_{\x \in \train} E_{q(\z|x)}\left[\prob_\params(\x,\z)\right].
# $$
# That is, we can drop terms that do not involve $\params$, and then get an *expected* version of the standard log-likelihood. It is this expectation that gives the algorithm its name.
# 
# Let us formalise the EM algorithm:
# 
# * **Input**:
#     * Initial parameter $\params_1$
# * **Initialisation**
#     * $i\leftarrow 1$
# * **while** not converged:
#     * Expectation-Step:
#         * Calculate the lower bound. This means calcuating the **expected** log-likelihood, and generally involves calculating a representation of the conditional probabilities $\prob_{\params_i}(\z|\x)$ that is convenient for optimising the expected log-likelihood in the M-Step.
#     * Maximisation-Step:
#         * Maximise the lower bound. This means finding the parameters that **maximise** the current expected log-likelihood. Often there is a closed form solution to this problem, and it involves weighted counting. 
#     * $i\leftarrow i + 1$
#         

# Let us implement the EM algorithm for our example task and model. 
# 
# #### E-Step 
# Calculating the conditional probabilities $q(z|x)=\prob_{\params_i}(z|\x)$ that make up the expectation is easy: we can calculate $\prob_{\params_i}(\x,z)$ for each class $z$, and then normalise these values to sum up to one:
# $$
#   q(z|x) = \frac{\prob_{\params_i}(\x,z)}{\sum_{z'} \prob_{\params_i}(\x,z')} 
# $$
# 
# #### M-Step
# The M-Step requires us to maximise the expected log-likelihood, using the distributions calculated in the E-Step. The solution is similar to the closed form solution of the [MLE](mle.ipynb) objective (and can be derived accordingly by the reader). It only differs by using weighted counts instead of hard counts:
# 
# $$
# \theta_{x|z} = \frac{ \sum_{\x \in \train} q(z|\x) \sum_{x_i \in \x }\indi(x_i = x)}{\sum_{\x \in \train} q(z|\x) |\x|}
# $$
# 
# The $\theta_z$ parameters can be calculated accordingly.
# 
# In Python we can implement this algorithm as follows. Notice that we are re-using the `calculate_class_distributions` function from above.

# In[104]:


from collections import defaultdict

def e_step(data,theta):
    return calculate_class_distributions(data, theta)

def m_step(x_data,q_data):
    counts = defaultdict(float)
    norm = defaultdict(float)
    class_counts = defaultdict(float)
    for x,q in zip(x_data, q_data):
        for z in z_domain:
            class_counts[z] += q[z]
            for x_i in x:
                norm[z] += q[z]
                counts[x_i, z] += q[z]
    theta_c = dict([(z,class_counts[z] / len(x_data)) for z in z_domain])
    theta_x = dict([((x,z),counts[x,z] / norm[z]) for z in z_domain for x in x_domain])
    return theta_x, theta_c

def em_algorithm(data, init_theta, iterations = 10):
    current_theta = init_theta
    current_q = None
    result = []
    for _ in range(0, iterations):
        current_q = e_step(data, current_theta)
        current_theta = m_step(data, current_q)
        current_marg_ll = marginal_ll(data, current_theta)
        current_bound = marginal_ll_bound(data, current_theta, current_q)
        result.append((current_q, current_theta, current_marg_ll, current_bound))
    return result


# Let us run the EM algorithm on some example data and observe the result.

# In[105]:


data = [[p,p,p,p,p,n],[n,n,n,n,n,n,f,p],[f,f,f,f,p,p,p,n]]
iterations = em_algorithm(data, current_theta, 5)
iterations[-1]


# We can see that the model learnt to assign the 'unhealthy' documents 1 and 3, consisting primarily of pizza and fries, to class `c1` and the healthy 'natto' document to cluster `c2`. 
# 
# To see how quickly the algorithm converges we can observe the behaviour of the lower bound and the marginal log-likelihood. 

# In[106]:


fig = plt.figure()
plt.plot(range(0, len(iterations)), [iteration[2] for iteration in iterations], label='marg_ll')
plt.plot(range(0, len(iterations)), [iteration[3] for iteration in iterations], label='bound')
plt.legend()
mpld3.display(fig)


# We see that the bound indeed is a lower bound of the of the marginal log-likelihood, and both objectives are converging very quickly to a local optimum. 

# ## Background Material
# * Michael Collin's [lecture notes on EM](http://www.cs.columbia.edu/~mcollins/em.pdf)
# * K. Nigal, A. McCallum, S. Thrun, T. Mitchell, [Text Classification from Labeled and Unlabeled Documents using EM](http://www.kamalnigam.com/papers/emcat-mlj99.pdf), Machine Learning, 2000
# * Jurafsky & Martin, [Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/8.pdf)
