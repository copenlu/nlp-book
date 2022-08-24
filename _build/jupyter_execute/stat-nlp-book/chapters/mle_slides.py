#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\nimport sys\nsys.path.append("..")\nimport statnlpbook.util as util\nimport statnlpbook.mle as smle\nfrom statnlpbook.util import safe_log as log\nutil.execute_notebook(\'mle.ipynb\')\n')


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
# \newcommand{\a}{\alpha}
# \newcommand{\b}{\beta}
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
# \newcommand{\china}{\text{China}}
# \newcommand{\mexico}{\text{Mexico}}
# \newcommand{\paramc}{\param_\china}
# \newcommand{\paramm}{\param_\mexico}
# \newcommand{\countc}{\counts{\train}{\china}}
# \newcommand{\countm}{\counts{\train}{\mexico}}
# $$

# # Maximum Likelihood Estimation
# for **ShallowDrumpf**!

# What does 
# $$
# \argmax_\params \sum_{(\x,\y) \in \train} \log \prob_\params(\x,\y)
# $$
# have to do with counting?

# ## Application: ShallowDrumpf
# 
# Develop **unigram language model** for generating simplified Trump speeches
# 
# > China, China, China, Mexico, China, Mexico ...

# ## Model
# 
# $$
# \prob_\params(w) = \params_w
# $$
# 
# $$
# \prob_\params(\text{China}) = \params_\text{China}  \qquad \prob_\params(\text{Mexico}) = \params_\text{Mexico} 
# $$
# 
# 

# In[3]:


m = "Mexico"
c = "China"
def prob(th_china, th_mexico, word):
    return th_china if word == 'China' else th_mexico

prob(0.7, 0.3, 'China')


# ## Maximum Likelihood Objective
# 
# $$
# l(\params) = \sum_{w \in \train} \log \prob_\params(w)
# $$

# $$
# l(\params)  = \countc  \log \paramc +  \countm \log \paramm
# $$

# Solution is **counting**:
# 
# $$
# \paramc = \frac{\countc}{\countc + \countm}
# $$

# In[4]:


def mle(data):
    theta_china = len([w for w in data if w == 'China']) / len(data)
    return theta_china, 1.0 - theta_china 

mle([c,c,m,c])


# ### Loss  Surface

# In[5]:


def ll(th_china, th_mexico, data):
    return sum([log(prob(th_china, th_mexico, w)) for w in data])

data = [c,c,m,c] # how does this graph look with all Cs?
smle.plot_mle_graph(lambda x,y: ll(x,y, data), mle(data), 
                    x_label='China',y_label='Mexico')


# Solution trivial (and useless) without **constraints**

# Constraints:
# 
# * $0 \leq \paramc \leq 1 $
# * $0 \leq \paramm \leq 1 $
# * $\paramc + \paramm = 1$
#     * Isoline of $g(\paramc,\paramm)=\paramc + \paramm$ 

# In[5]:


smle.plot_mle_graph(lambda x,y: ll(x,y, data), mle(data), 
                    show_constraint=True)


# ## Gradients at Optimum

# In[6]:


smle.plot_mle_graph(lambda x,y: ll(x,y, data), mle(data), 
                    show_constraint=True, show_optimum=True)


# $$
# \nabla_\params l(\params) = \alpha \nabla_\params g(\params)
# $$

# $$
# l(\params)  = \countc  \log \paramc +  \countm \log \paramm
# $$

# $$
# \frac{\partial l(\params)}{\partial \paramc} = \frac{\counts{D}{China}}{\paramc}
# $$

# $$
# g(\params) = \paramc + \paramm
# $$

# $$
# \frac{\partial g(\params)}{\partial \paramc} = 1
# $$

# $$
# \frac{\partial l(\params)}{\partial \paramc} = \alpha \frac{\partial g(\params)}{\partial \paramc}
# $$

# $$
# \frac{\countc}{\paramc} = \alpha 
# $$

# $$
# \paramc = \frac{\countc}{\alpha} = \ldots
# $$
# $$
# \paramm = \frac{\countm}{\alpha} = \ldots
# $$

# $$
# \paramc = \frac{\countc}{\countc + \countm}
# $$

# ## Summary
# 
# * Derive MLE by 
#     * equating loss and constraint gradient
#     * using constraint equation
# * Easy to extend to any discrete generative model with conditional probability tables
# * Learning goal: be able to derive the equation for new models 

# ## Background Material
# * Introduction to MLE in [Mike Collin's notes](http://www.cs.columbia.edu/~mcollins/em.pdf)
