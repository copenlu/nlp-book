#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# # Configuration

# <!---
# Latex Macros
# -->
# $$
# \newcommand{\Xs}{\mathcal{X}}
# \newcommand{\Ys}{\mathcal{Y}}
# \newcommand{\y}{\mathbf{y}}
# \newcommand{\repr}{\mathbf{f}}
# \newcommand{\repry}{\mathbf{g}}
# \newcommand{\x}{\mathbf{x}}
# \newcommand{\vocab}{V}
# \newcommand{\params}{\boldsymbol{\theta}}
# \newcommand{\param}{\theta}
# \DeclareMathOperator{\perplexity}{PP}
# \DeclareMathOperator{\argmax}{argmax}
# \DeclareMathOperator{\argmin}{argmin}
# \newcommand{\train}{\mathcal{D}}
# \newcommand{\counts}[2]{\#_{#1}(#2) }
# \newcommand{\indi}{\mathbb{I}}
# $$

# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%load_ext tikzmagic\n%autoreload 2\nimport sys\nsys.path.append("..")\nimport numpy as np\n\n#reveal configuration\nfrom notebook.services.config import ConfigManager\ncm = ConfigManager()\ncm.update(\'livereveal\', {\n        \'theme\': \'white\',\n        \'transition\': \'none\',\n        \'controls\': \'false\',\n        \'progress\': \'true\',\n})\n')


# In[3]:


get_ipython().run_cell_magic('html', '', '<style>\n.red { color: #E41A1C; }\n.orange { color: #FF7F00 }\n.yellow { color: #FFC020 }         \n.green { color: #4DAF4A }                  \n.blue { color: #377EB8; }\n.purple { color: #984EA3 }       \n       \nh1 {\n    color: #377EB8;\n}\n       \nctb_global_show div.ctb_hideshow.ctb_show {\n    display: inline;\n} \n         \ndiv.tabContent {\n    padding: 0px;\n    background: #ffffff;     \n    border: 0px;                        \n}  \n         \n.left {\n    float: left;\n    width: 50%;\n    vertical-align: text-top;\n}\n\n.right {\n    margin-left: 50%;\n    vertical-align: text-top;                            \n}    \n               \n.small {         \n    zoom: 0.9;\n    -ms-zoom: 0.9;\n    -webkit-zoom: 0.9;\n    -moz-transform:  scale(0.9,0.9);\n    -moz-transform-origin: left center;  \n}          \n         \n.verysmall {         \n    zoom: 0.75;\n    -ms-zoom: 0.75;\n    -webkit-zoom: 0.75;\n    -moz-transform:  scale(0.75,0.75);\n    -moz-transform-origin: left center;  \n}         \n   \n        \n.tiny {         \n    zoom: 0.6;\n    -ms-zoom: 0.6;\n    -webkit-zoom: 0.6;\n    -moz-transform:  scale(0.6,0.6);\n    -moz-transform-origin: left center;  \n}         \n         \n         \n.rendered_html blockquote {\n    border-left-width: 0px;\n    padding: 15px;\n    margin: 0px;    \n    width: 100%;                            \n}         \n         \n.rendered_html th {\n    padding: 0.5em;  \n    border: 0px;                            \n}         \n         \n.rendered_html td {\n    padding: 0.25em;\n    border: 0px;                                                        \n}    \n     \n#for reveal         \n.aside .controls, .reveal .controls {\n    display: none !important;                            \n    width: 0px !important;\n    height: 0px !important;\n}\n    \n.rise-enabled .reveal .slide-number {\n    right: 25px;\n    bottom: 25px;                        \n    font-size: 200%;     \n    color: #377EB8;                        \n}         \n         \n.rise-enabled .reveal .progress span {\n    background: #377EB8;\n}     \n         \n.present .top {\n    position: fixed !important;\n    top: 0 !important;                                   \n}                  \n    \n.present .rendered_html * + p, .present .rendered_html p, .present .rendered_html * + br, .present .rendered_html br {\n    margin: 0.5em 0;                            \n}  \n         \n.present tr, .present td {\n    border: 0px;\n    padding: 0.35em;                            \n}      \n         \n.present th {\n    border: 1px;\n}\n         \npresent .prompt {\n    min-width: 0px !important;\n    transition-duration: 0s !important;\n}     \n         \n.prompt {\n    min-width: 0px !important;\n    transition-duration: 0s !important;                            \n}         \n         \n.rise-enabled .cell li {\n    line-height: 135%;\n}\n         \n</style>\n')


# <center><h1>An Introduction to Deep Learning for Natural Language Processing</h1></center><br>
# 

# # A Subjective History of Deep Learning

# ## Disclaimer
# - The field of Deep Learning is young but fast-changing and diverse due to very active research
# - I can only give you a small overview on Deep Learning
# - I won't talk about vision, convolutional networks etc.
# - Many things that I explain today will be outdated next year/month

# <center><h2>A More or Less Objective View</h2></center>
# <br><br><br>
# <img  src="../img/schmidthuber.png"/>

# <center><h2>A Personal View</h2></center>
# <br>
# <span class=red>Feature Engineering</span>, Classification, Support Vector Machines
# 
# <img  src="../img/personal_2011.png"/>

# <img  src="../img/features.png"/>

# <img  src="../img/margin.png"/>

# <center><h2>Machine Learning</h2></center>
# <img  src="../img/ml.png" width=1000/>

# Graphical Models, Structured Prediction, Probabilistic Inference, <span class=red>Feature Engineering</span>

# <img  src="../img/sequence.png"/>

# <img  src="../img/factor.png" width=700/>

# <img  src="../img/feature_classes.png" width=700/>

# Relation Extraction, Matrix Factorization, <span class=green>Representation Learning</span>
# <img  src="../img/factorisation.png" width=800/>

# <span class=green>Representation Learning</span>, **<span class=blue>Deep Learning</span>**

# <img src="../img/lstm.svg">

# ## Deep Learning in a Nutshell

# <img  src="../img/alexnet.png" width=900/>

# <img  src="../img/filters.png"/>

# ## The Success Story of Deep Learning
# - State of the art performance for countless real-world tasks (too much to list)
# - Huge investements from industry (Google, Facebook, Apple etc.)
# - Many new Deep Learning start-ups
# - Very active and open research community
# - "There's something magical about Recurrent Neural Networks" -- Andrej Karpathy 

# <img  src="../img/aihires.png" width="1000"/>

# <img  src="../img/countries_capitals.png"/>

# <img  src="../img/w2v.png" width=1000/>

# <img  src="../img/emojis.png"/>

# <img  src="../img/atari.gif" width=1000/>

# <img  src="../img/kubrick.jpg" width=1000/>

# <img  src="../img/style_transfer_1.png"/>

# <img  src="../img/style_transfer_2.png"/>

# <img  src="../img/alphago.jpg" width="600"/>

# # Continuous Optimization, Modularity and Backpropagation

# ### Preliminaries: Model
# 
# Change of notation: 
# $$
# s_\params(\x,y) \in \mathbb{R}
# $$
# becomes
# $$
# f_\params(\x)_y \in  \mathbb{R}
# $$
# 
# where $f_\params(\x) \in \mathbb{R}^{|\Ys|}$ represents the scores for each possible solution $y$
# 
# 

# ### Preliminaries: Model
# 
# - Model: some function $f_\theta$ parameterized by $\theta$ that we want to learn from data $\mathcal{D}=\{(x_i,y_i)\}$, for example
#   - Linear Regression
# $$
# f_\theta(\mathbf{x}) = \mathbf{Wx} + \mathbf{b} \quad\text{with }\theta = \{\mathbf{W}, \mathbf{b}\}
# $$
# 
#   - Logistic Regression
# $$
# f_\theta(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{Wx} + \mathbf{b})}} \quad\text{with }\theta = \{\mathbf{W}, \mathbf{b}\}
# $$
#   - 3-layer Perceptron
# $$
# f_\theta(\mathbf{x}) = \text{tanh}(\mathbf{W}_3\text{tanh}(\mathbf{W}_2\text{tanh}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)+\mathbf{b}_2)+\mathbf{b}_3)\\ \quad\text{with }\theta = \{\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3, \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3\}
# $$  
# 

# ### Preliminaries: Loss Functions
# A function $\mathcal{L}$ that given a model $f_\theta$, input $x$ and gold output $y$ measures how far we are away from the truth, for example 
#   - Squared distance
# $$
# \mathcal{L}(f_\theta, x, y) = ||f_\theta(x) - y||^2
# $$
#   - Logistic
# $$
# \mathcal{L}(f_\theta, x, y) = \log(1 + f_\theta(yx))
# $$  
#   - Hinge
# $$
# \mathcal{L}(f_\theta, x, y) = \max(0,1-yf_\theta(x))
# $$

# ### Stochastic Gradient Descent
# 
# Goal: find parameters $\theta$ of model $f_\theta$ that minimize loss function $\mathcal{L}$
# 
# 1. Initialize parameters $\theta$
# 2. Shuffle training data $\mathcal{D}$
#   - For every example $(x_i,y_i) \in \mathcal{T}$ 
#     1. Find direction of parameters that improves loss 
#       - Calculate gradient of parameters w.r.t. loss $\frac{\partial \mathcal{L}(f_\theta, x_i, y_i)}{\partial \theta}$
#     2. Update parameters with learning rate $\alpha$  
#       - $\theta := \theta - \alpha*\frac{\partial \mathcal{L}(f_\theta, x_i, y_i)}{\partial \theta}$
#   - Go to 2.    
#     

# <img  src="../img/surface.png" width=1000/>

# <img  src="../img/momentum.gif" width=800/>

# ## Perceptron: A Single Neuron

# <img src="../img/single_neuron.svg">

# \begin{align}
# z &= \text{sigmoid}(x_1*w_1 + x_2*w_2 + x_3*w_3 + x_4*w_4 + b)\\
#   &= \text{sigmoid}(\mathbf{x}\cdot\mathbf{w} + b) \quad\text{with }\mathbf{x},\mathbf{w}\in\mathbb{R}^4
# \end{align}

# ## Multiple Neurons

# <img src="../img/multiple_neurons.svg" width=800/>

# \begin{align}
# z_1 &= \text{sigmoid}(\mathbf{x}\cdot\mathbf{w_1} + b_1)\\
# z_2 &= \text{sigmoid}(\mathbf{x}\cdot\mathbf{w_2} + b_2)
# \end{align}

# ## Multiple Neurons
# 
# $f_\theta: \mathbb{R}^4 \to \mathbb{R}^2$

# <img src="../img/multiple_neurons_2.svg">

# \begin{align}
# \mathbf{z} &= \text{sigmoid}(\mathbf{W}\mathbf{x} + \mathbf{b}) \quad\text{ with } \mathbf{W}\in\mathbb{R}^{2\times4}, \mathbf{b}\in\mathbb{R}^{2}
# \end{align}

# ## Modularity: Multi-layer Perceptron

# <img src="../img/mlp.svg"/>

# <div class=right><div class=top><div class=small>
# <div style="margin-bottom: 60%;"></div>
# \begin{align}
# f_{1,\theta} &: \mathbb{R}^5 \to \mathbb{R}^3\\
# f_{2,\theta} &: \mathbb{R}^3 \to \mathbb{R}^3\\
# f_{3,\theta} &: \mathbb{R}^3 \to \mathbb{R}^1\\
# g_\theta &= f_{3,\theta} \circ f_{2,\theta} \circ f_{1,\theta}\\
# g_\theta(\mathbf{x}) &= f_{3,\theta}(f_{2,\theta}(f_{1,\theta}(\mathbf{x})))\\
# g_\theta &: \mathbb{R}^5 \to \mathbb{R}^1
# \end{align}
# </div>
# </div>

# ## Calculation of Gradients
# <br>
# <div class=verysmall>
# \begin{align}
# g_\theta(\mathbf{x}) &= \text{sigmoid}(\mathbf{W}^{1\times 3}_3\text{sigmoid}(\mathbf{W}^{3\times 3}_2\text{sigmoid}(\mathbf{W}^{3\times 5}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2)+\mathbf{b}_3)\\
# \frac{\partial \mathcal{L}(f_\theta, \mathbf{x}, \mathbf{y})}{\partial \mathbf{W}^{1\times 3}_3} &= \text{ ?}\\
# \frac{\partial \mathcal{L}(f_\theta, \mathbf{x}, \mathbf{y})}{\partial \mathbf{b}_3} &= \text{ ?}\\
# \frac{\partial \mathcal{L}(f_\theta, \mathbf{x}, \mathbf{y})}{\partial \mathbf{W}^{3\times 3}_2} &= \text{ ?}\\
# \frac{\partial \mathcal{L}(f_\theta, \mathbf{x}, \mathbf{y})}{\partial \mathbf{b}_2} &= \text{ ?}\\
# \frac{\partial \mathcal{L}(f_\theta, \mathbf{x}, \mathbf{y})}{\partial \mathbf{W}^{3\times 5}_1} &= \text{ ?}\\
# \frac{\partial \mathcal{L}(f_\theta, \mathbf{x}, \mathbf{y})}{\partial \mathbf{b}_1} &= \text{ ?}
# \end{align}
# </div>

# ## Chain Rule
# 
# \begin{align}
# \frac{\partial f \circ g}{\partial \theta} &= \frac{\partial f \circ g}{\partial g} \frac{\partial g}{\partial \theta}\\
# \end{align}

# ## Example

# <div class=small>
# \begin{align}
# \frac{\partial \mathcal{L}(\text{sigmoid}(\mathbf{W}\mathbf{x}),\mathbf{y})}{\partial \mathbf{W}} &= \frac{\partial \mathcal{L}(\text{sigmoid}(\mathbf{W}\mathbf{x}),\mathbf{y})}{\partial \text{ sigmoid}(\mathbf{W}\mathbf{x})} \frac{\partial \text{ sigmoid}(\mathbf{W}\mathbf{x})}{\partial \mathbf{Wx}} \frac{\partial{\mathbf{Wx}}}{\partial\mathbf{W}}
# \end{align}
# </div>

# \begin{align}
# \mathbf{h} &= \mathbf{W}\mathbf{x}\\
# \mathbf{z} &= \text{sigmoid}(\mathbf{h})\\
# \mathcal{L}(\mathbf{z},\mathbf{y}) &= \frac{1}{2}||\mathbf{z} - \mathbf{y}||^2
# \end{align}

# \begin{align}
# \frac{\mathcal{\partial \frac{1}{2}||\mathbf{z} - \mathbf{y}||^2}}{\partial \mathbf{W}} &= \frac{\partial \frac{1}{2}||\mathbf{z} - \mathbf{y}||^2}{\partial\mathbf{z}} \frac{\partial\mathbf{z}}{\partial \mathbf{h}} \frac{\partial \mathbf{h}}{\partial \mathbf{W}}
# \end{align}

# ## Example cont.

# \begin{align}
# \mathbf{h} &= \mathbf{W}\mathbf{x}\\
# \mathbf{z} &= \text{sigmoid}(\mathbf{h})\\
# \mathcal{L}(\mathbf{z},\mathbf{y}) &= \frac{1}{2}||\mathbf{z} - \mathbf{y}||^2
# \end{align}

# \begin{align}
# \frac{\mathcal{\partial \frac{1}{2}||\mathbf{z} - \mathbf{y}||^2}}{\partial \mathbf{W}} &= \frac{\partial \frac{1}{2}||\mathbf{z} - \mathbf{y}||^2}{\partial\mathbf{z}} \frac{\partial\mathbf{z}}{\partial \mathbf{h}} \frac{\partial \mathbf{h}}{\partial \mathbf{W}}\\
# \partial \mathbf{z} &= \mathbf{z}-\mathbf{y}\\
# \partial \mathbf{h} &= \partial \mathbf{z}\,\text{sigmoid}(\mathbf{h})\,(1 - \text{sigmoid}(\mathbf{h}))\\
# \partial \mathbf{W} &= \partial\mathbf{h}\otimes\mathbf{x}
# \end{align}

# ## Module

# <img src="../img/dl_module.svg">

# ## Backpropagation

# <img src="../img/backprop.svg">

# ## Deep Learning Libraries
# - pytorch
# - dynet
# - Theano
# - DeepLearning4J
# - autograd
# - **TensorFlow**
# - ...

# <img  src="../img/tensorflow.jpg"/>

# ## Logistic Regression

# In[4]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
seed = 0
#input
input_sz = 3
output_sz = 1
x = tf.placeholder("float")
#parameters
W = tf.Variable(tf.random_uniform([output_sz,input_sz], -0.1, 0.1, seed=seed))
b = tf.Variable(tf.zeros(output_sz))
#f_theta
z = tf.nn.sigmoid(tf.matmul(W,x) + b) #sigmoid(Wx + b)


# In[4]:


sess = tf.Session()
sess.run(tf.global_variables_initializer()) #initialize W and b
sess.run(W)


# In[5]:


sess.run(b)


# ## Logistic Regression cont.

# Forward: $\mathbf{z} = f_\theta(\mathbf{x})$

# In[6]:


sess.run(z, feed_dict={x: [[-5.5],[2.0],[-0.5]]})


# Backward: $\partial\mathbf{W},\partial\mathbf{b},\partial\mathbf{x}$ given upstream gradient $\partial\mathbf{z}$

# In[7]:


sess.run(tf.global_variables_initializer())
gradz = [[0.1]] 
grad = tf.gradients(z,[W, b, x], grad_ys=gradz)
sess.run(grad, feed_dict={x: [[-5.5],[2.0],[-0.5]]})


# ## Multi-layer Perceptron

# In[8]:


#input
x = tf.placeholder(tf.float32, shape=[5,1])
#parameters
W1 = tf.Variable(tf.random_uniform([3,5], seed=seed))
b1 = tf.Variable(tf.zeros([3,1]))
W2 = tf.Variable(tf.random_uniform([3,3], seed=seed))
b2 = tf.Variable(tf.zeros([3,1]))
W3 = tf.Variable(tf.random_uniform([1,3], seed=seed))
b3 = tf.Variable(tf.zeros([1,1]))
#model
h1 = tf.nn.sigmoid(tf.matmul(W1,x) + b1) 
h2 = tf.nn.sigmoid(tf.matmul(W2,h1) + b2)
mlp_z = tf.matmul(W3,h2) + b3 

sess.run(tf.global_variables_initializer())
x_value = [[-5.5], [2.0], [-0.5], [2.0], [4.0]]
sess.run(mlp_z, feed_dict={x: x_value})


# ## Training

# In[9]:


target_z = tf.constant([[1.0]]) # what the output should be
loss = tf.square(target_z - mlp_z) # the loss function 
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
opt_op = optimizer.minimize(loss) # the TF operation that performs optimisation steps
sess.run(tf.global_variables_initializer())
for epoch in range(0,5):
    _, loss_value = sess.run([opt_op, loss], feed_dict={x: x_value})
    if epoch % 1 == 0:
        print(loss_value)


# It learned!

# In[10]:


sess.run(mlp_z, feed_dict={x: x_value})


# ## Next
# 
# Input are always (continuous) **vectors**. 
# 
# What vectors to use in NLP?  

# In[ ]:




