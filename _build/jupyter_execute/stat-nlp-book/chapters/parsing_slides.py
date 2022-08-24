#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\nimport sys\nsys.path.append("..")\nimport statnlpbook.util as util\nimport statnlpbook.parsing as parsing\nutil.execute_notebook(\'parsing.ipynb\')\n')


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
# $$

# # Parsing

# In[3]:


get_ipython().run_cell_magic('HTML', '', '<style>\ntd,th {\n    font-size: x-large;\n    text-align: left;\n}\n</style>\n')


# ##  Motivation 
# 
# Say want to automatically build a database of this form
# 
# | Brand   | Parent    |
# |---------|-----------|
# | KitKat  | Nestle    |
# | Lipton  | Unilever  |  
# | ...     | ...       |  
# 
# or this [graph](http://geekologie.com/image.php?path=/2012/04/25/parent-companies-large.jpg)

# Say you find positive textual mentions in this form:
# 
# > <font color="blue">Dechra Pharmaceuticals</font> has made its second acquisition after purchasing <font color="green">Genitrix</font>.

# 
# > <font color="blue">Trinity Mirror plc</font> is the largest British newspaper after purchasing rival <font color="green">Local World</font>.

# Can you find a pattern? 

# How about this sentence 
# 
# > <font color="blue">Kraft</font> is gearing up for a roll-out of its <font color="blue">Milka</font> brand after purchasing  <font color="green">Cadbury Dairy Milk</font>.
# 

# Wouldn't it be great if we knew that
# 
# * Kraft is the **subject** of the phrase **purchasing Cadbury Dairy Milk** 

# Check out [enju parser](http://www.nactem.ac.uk/enju/demo.html#2)

# Parsing is is the process of **finding these trees**:
# 
# * very important for downstream applications
# * the "celebrity" sub-field of NLP 
#     * partly because it  marries linguistics and NLP
# * researched bigly in academia and [industry](http://www.telegraph.co.uk/technology/2016/05/17/has-googles-parsey-mcparseface-just-solved-one-of-the-worlds-big/)

# How is this done?

# ## Syntax
# from the Greek syntaxis (arrangement):
# 
# * **Constituency**: groups of words act as single units.
# * **Grammatical Relations**: object, subject, direct object etc. 
# * **Subcategorization**: restrictions on the type of phrases that go with certain words.
# 

# ### Constituency
# 
# * **Noun Phrase** (NP)
#     * a roll-out of its Milka brand
#     * Cadbury Dairy Milk
#     * a roll-out
# * **Verb Phrase** (VP) 
#     * is gearing up
#     * purchasing Cadbury Dairy Milk 
# * **Prepositional Phrase** (PP)
#     * of its Milka brand
#     * after purchasing Cadbury Dairy Milk

# ### Grammatical Relations
# > <font color="blue">Kraft</font> is gearing up for a roll-out of its <font color="blue">Milka</font> brand after purchasing  <font color="green">Cadbury Dairy Milk</font>.
# 
# * *Subject* of purchasing: **Kraft**
# * *Object* of purchasing: **Cadbury Dairy Milk**

# ### Subcategorization
# 
# There are more complex (sub) categories of verbs (and other types of words)
# 
# * Intransitive Verbs: must not have objects
#     * the student works
# * Transitive Verbs: must have exactly one object
#     * Kraft purchased Cadbury Dairy Milk
# * Ditransitive Verbs: must have two objects
#     * Give me a break! 
# 

# ## Context Free Grammars 
# 
# Formalise syntax by describing the hierarchical structure of sentences
# 

# A **Context Free Grammar** (CFG) is a 4-tuple \\(G=(N,\Sigma,R,S)\\) where
# 
#   * \\(N\\) is a set of _non-terminal symbols_.
#   * \\(\Sigma\\) is a set of _terminal symbols_.
#   * \\(R\\) is a finite set of _rules_ \\(X \rightarrow Y_1 Y_2\ldots Y_n\\) where \\(X \in N\\) and \\(Y_i \in N \cup \Sigma\\). 
#   * \\(S \in N\\) is a _start symbol_. 
# 

# Simple example grammar:
# * NP_p : plural Noun Phrase
# * NP_s : singular Noun Phrase
# * VP_s/p: same for verb phrases

# In[4]:


cfg = CFG.from_rules([('S',    ['NP_p','VP_p']),('S',['NP_s','VP_s']), 
                      ('NP_p', ['Matko', 'raps']),
                      ('VP_p', ['are', 'ADJ']),
                      ('NP_s', ['Matko']),
                      ('VP_s', ['raps', 'in', 'StatNLP']),
                      ('ADJ',  ['silly'])
                     ])
cfg


# ## (Left-most) Derivation
# The structure of a sentence with respect to a grammar can be described by its **derivation** (if it exists) 

# Sequence of sequences \\(s_1 \ldots s_n\\) such that 
# 
# * \\(s_1 = S\\)
#     * first sequence is the start symbol
# * \\(s_n \in \Sigma^*\\)
#     * last sequence consists of only terminals.
# * \\(s_i\\) for \\(i > 1\\)
#     * replace left-most non-terminal \\(\alpha\\) in $s_{i-1}$ with right-hand of $\alpha\rightarrow \beta_1,\ldots,\beta_n$

# In[4]:


util.Table(generate_deriv(cfg, [cfg.s]))


# ## Parse Trees
# Represent derivations as trees

# In[5]:


tree = ('S', [('NP_p',['Matko','raps']), ('VP_p',['are','silly'])])
parsing.render_tree(tree)


# In[6]:


parsing.render_tree(generate_tree(cfg,'S'))        


# ## Parsing
# The inverse problem: given a sentence 
# 
# > Matko raps in StatNLP
# 
# What's the derivation for it?  

# There are a couple of approaches to find a legal parse tree given a sentence and grammar:
# 
# * **Top-Down**: Start with the start symbol and generate trees
#     * backtrack if they do not match observed sentence
# * **Bottom-Up**: Start with the sentence, find rules that generate parts of it
#     * backtrack if you can't reach the start symbol
# * **Dynamic Programming**: Explore several trees in parallel and re-use computations

# ### Bottom-Up Parsing with Backtracking
# 
# Incrementally build up a tree **left-to-right**, and maintain ...

# a **buffer** of remaining words

# In[7]:


parsing.render_transitions(transitions[0:1])


# a **stack** of trees build so far

# In[8]:


parsing.render_transitions(transitions[13:14])


# Perform three types of **actions**:

# ### Shift
# Put first word from buffer to stack (as singleton tree)

# In[9]:


parsing.render_transitions(transitions[0:2])


# ### Reduce
# For rule $X \rightarrow Y \: Z$ and stack $Y \: Z$, create new tree headed with $X$

# In[10]:


parsing.render_transitions(transitions[11:13])


# ### Backtrack
# If no rule can be found and the buffer is empty, go back to last decision point

# In[11]:


parsing.render_transitions(transitions[10:13])


# ### Example

# In[12]:


sentence = ['Matko', 'raps', 'are', 'silly']
transitions = bottom_up_parse(cfg, sentence)
cfg


# In[13]:


parsing.render_transitions(transitions[10:14])


# In[14]:


parsing.render_forest(transitions[-1][0].stack)


# ## Dynamic Programming for Parsing
# Bottom-up parser repeats the same work several times

# In[15]:


parsing.render_transitions(transitions[7:8]) 


# In[16]:


parsing.render_transitions(transitions[10:13])


# In[17]:


parsing.render_transitions(transitions[-2:-1])


# Fortunately we can **cache** these computations

# ### Chomsky Normal Form
# Algorithm for caching requires **Chomsky Normal Form**
# 
# Rules have form:
# 
# * \\(\alpha \rightarrow \beta \gamma\\) where \\(\beta,\gamma \in N \setminus S \\). 
#     * rule with exactly two non-terminals on RHS
# * \\(\alpha \rightarrow t\\) where \\(t \in \Sigma\\)
#     * rule that expands to single 
#    terminal

# ## Conversion to CNF
# We can convert every CFG into an equivalent CFG in CNF
# 
# Replace left rules by right rules:  
# 
# * $\alpha \rightarrow \beta \gamma \delta \Rightarrow \alpha \rightarrow \beta\alpha', \alpha' \rightarrow \gamma \delta$
# * $\alpha \rightarrow \beta t \Rightarrow \alpha \rightarrow \beta \alpha', \alpha' \rightarrow t$ where $t \in \Sigma$
# * $\alpha \rightarrow \beta, \beta \rightarrow \gamma \delta \Rightarrow \alpha \rightarrow \gamma \delta, \beta \rightarrow \gamma \delta$ 
# 

# ## Example
# 
# $S \rightarrow NP \: VP \: PP$ 

# becomes $S \rightarrow S' \: PP$ and $S' \rightarrow NP \: VP$

# $VP \rightarrow \text{are} \: ADJ$ 

# becomes $VP \rightarrow X \: ADJ$ and $X \rightarrow \text{are}$

# In[18]:


cnf_cfg = to_cnf(cfg)
cnf_cfg


# ### Cocke–Younger–Kasami (CYK) Algorithm
# 
# **Incrementally** build all parse trees for **spans of increasing length**

# Like the one for "are silly" and "Matko Raps":

# In[19]:


parsing.render_transitions(transitions[16:17]) 


# ### CYK Algorithm
# Populate chart with non-terminal $l$ for span $(i,j)$ 
# 
# if $j=i$
# * Add label $l$ if $l \rightarrow x_i \in R$ 

# if $j>i$
# * Consider all *middle* indices $m$   
# * **combine trees** of span $(i,m)$ and $(m+1,j)$ with labels $l_1$ and $l_2$
# * if there is a rule $l \rightarrow l_1 \: l_2 \in R$

# Best done in a **chart** to store 
# * legal non-terminals per span 
# * and back-pointers to child spans

# In[20]:


chart = parsing.Chart(sentence)
chart.append_label(0,0,'NP_s')
chart.append_label(0,0,'NP_p_0')
chart.append_label(1,1,'VP_s_6')
chart.append_label(1,1,'NP_p_1')
chart.append_label(0,1,'NP_p_2', [(0,0,'NP_p_0'),(1,1,'NP_p_1')]) 
chart.mark(0, 1, 'NP_p_2')
chart.mark_target(0,1)
chart


# In[21]:


cnf_cfg


# In[22]:


trace = cyk(cnf_cfg, sentence)
util.Carousel(trace)


# The chart can be **traversed backwards** to get all trees

# In[23]:


util.Carousel([trace[i] for i in [35,33,22,13]])


# In[24]:


parse_result = trace[-1].derive_trees()[0]
parsing.render_tree(parse_result)


# Collapse **CNF non-terminals**

# In[25]:


parsing.render_tree(parsing.filter_non_terminals(parse_result, cfg.n))


# ## Ambiguity 
# For real world grammars many phrases have **several legal parse trees**

# Consider the following grammar and sentence

# In[26]:


amb_cfg = CFG.from_rules([
        ('S',    ['Subj','VP']),
        ('Subj', ['He']),
        ('Verb', ['shot']),
        ('VP',   ['Verb', 'Obj']),        ('VP', ['Verb', 'Obj', 'PP']),
        ('PP',   ['in','his','pyjamas']),
        ('Obj',  ['the','elephant']),     ('Obj', ['the','elephant','PP'])
    ])
amb_cnf_cfg = to_cnf(amb_cfg)
amb_sentence = ["He", "shot", "the", "elephant", "in", "his", "pyjamas"]


# In[27]:


amb_cfg


# In[28]:


" ".join(amb_sentence)


# In[29]:


amb_trace = cyk(amb_cnf_cfg, amb_sentence)
amb_parse_results = amb_trace[-1].derive_trees()
def ambiguous_tree(num):
    return parsing.render_tree(parsing.filter_non_terminals(amb_parse_results[num],amb_cfg.n)) # try results[1]


# In[30]:


ambiguous_tree(1) # try tree 1 


# **prepositional phrase attachment ambiguity**: "in his pyjamas" could be 
# 
# * in verb phrase (in pyjamas when shooting)
# * in noun phrase (elephant in pyjamas)
# 

# Both readings are grammatical, but one is **more probable**

# ## Probabilistic Context Free Grammars
# [Probabilistic Context Free Grammars](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf) (PFCGs) are Context Free Grammars in which rules have probabilities 
# 
# * A Context Free Grammar \\(G(N,\Sigma,R,S)\\)
# * A parameter \\(\param(\alpha \rightarrow \beta) \in [0,1]\\) for each rule  \\(\alpha \rightarrow \beta \in R\\) 
# * For each left hand side \\(\alpha \in N\\) we require \\(\sum_\beta \param(\alpha \rightarrow \beta) = 1\\)

# A PCFG defines probability for parse tree \\(\mathbf{t}\\) containing the rules \\(\alpha_1 \rightarrow \beta_1, \ldots, \alpha_n \rightarrow \beta_n\\):
# $$
#   \newcommand{parse}{\mathbf{t}}
#   p_{\param}(\parse) = \prod_i^n \param(\alpha_i \rightarrow \beta_i) 
# $$
# 

# Example PCFG:

# In[31]:


pcfg = PCFG.from_rules([
        ('S',    1.0, ['Subj','VP']),
        ('Subj', 1.0, ['He']),
        ('Verb', 1.0, ['shot']),
        ('VP',   0.3, ['Verb', 'Obj']),        ('VP',  0.7, ['Verb', 'Obj', 'PP']),
        ('PP',   1.0, ['in','his','pyjamas']),
        ('Obj',  0.5, ['the','elephant']),     ('Obj', 0.5, ['the','elephant','PP'])
    ])
pcfg


# ## Parsing
# 
# For given sentence $\x$, let $\Ys(\x,G)$ be all trees $\mathbf{t}$ with $\x$ as terminals:
# 
# $$
# \argmax_{\mathbf{t} \in \Ys(\x,G)} \prob_\params(\mathbf{t}) 
# $$
# 

# ## CYK for PCFGs
# We can use a variant of the CYK algorithm to solve the prediction problem

# Populate chart with non-terminal $l$ for span $(i,j)$ **and score $s$**
# 
# if $j=i$
# * Add label $l$ **with score $\theta(l \rightarrow x_i )$** if $l \rightarrow x_i \in R$ 

# if $j>i$
# * Consider all *middle* indices $m$   
# * combine trees of span $(i,m)$ and $(m+1,j)$ with labels $l_1$ and $l_2$ and scores $s_1$ and $s_2$
#     * **and score $\theta(l \rightarrow l_1 \: l_2) \times s_1 \times s_2$**
# * if there is a rule $l \rightarrow l_1 \: l_2 \in R$

# In[32]:


cnf_pcfg


# In[33]:


util.Carousel(pcyk_trace)


# Runtime with respect to sentence length? 

# Resolve parse by going backwards ... 

# In[34]:


pcyk_trace = pcyk(cnf_pcfg, amb_sentence)
parsing.render_tree(parsing.filter_non_terminals(pcyk_trace[-1].derive_trees()[0],pcfg.cfg.n))


# ## Learning
# 
# Learning for PCFGs :
# 
# 1. What should the rules in the grammar be?
# 2. What should the probabilities associated with these rules be?

# Need corpus of parse trees $\train=(\parse_1, \ldots, \parse_n)$ 
# 
# * English: [Penn Treebank Project](https://www.cis.upenn.edu/~treebank/) parses for the 1989 Wall Street Journal (among other sources). 
# * Other languages: e.g. [Chinese](https://catalog.ldc.upenn.edu/LDC2013T21)
# * Other domains: e.g. [Biomedical Papers](www.nactem.ac.uk/aNT/genia.html)
# 
# Annotation expensive and need experts, major bottleneck in parsing research. 

# To learn the parameters $\params$ of the model we can again use the maximum-likelihood criterium:
# 
# $$
# \params^* = \argmax_\params \sum_{\parse \in \train} \log \prob_\params(\parse)
# $$

# Amounts to **counting**
# 
# $$
#   \param(\alpha \rightarrow \beta) = \frac{\counts{\train}{\alpha \rightarrow \beta}}{\counts{\train}{\alpha}}
# $$
# 
# Details omitted here, as you have seen this before

# ## Advanced: Parent Annotation
# 
# In practice 
# 
# * Let $X^Y$ be a non-terminal $X$ with parent $Y$
# * **Grandparents** matter
#     * $NP^{VP} \rightarrow NP \: PP$ vs 
#     * $NP^{PP} \rightarrow NP \: PP$  
# * Can be captured by labelling nodes in training trees with their parent
#     * Same machinery

# ## Advanced: Head Driven PCFG
# 
# In practice 
# 
# * **VP NP** is not necessarily less or more likely than **VP NP PP**
# * But **elephant** in **pyjamas** is very unlikely
# * PCFGs must model relations between important words ("heads")
#     * $PP^{NP(\text{elephant})} \rightarrow IN \: NP(\text{pyjamas})$ vs
#     * $PP^{VP(\text{shot})} \rightarrow IN \: NP(\text{pyjamas})$
# * Needs more complex model and search algorithms

# ## Background Material
# 
# * [Mike Collins' PCFG lecture](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf)
# * Jurafsky & Martin, Chapter 12, Statistical Parsing
