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
# \newcommand{\a}{\mathbf{a}}
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


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# preamble\nimport sys\nsys.path.append("..")\nimport statnlpbook.transition as transition\n')


# # Transition-based dependency parsing
# 
# This chapter is influenced by the [EACL 2014 tutorial](http://stp.lingfil.uu.se/~nivre/eacl14.html) by Ryan McDonald and Joakim Nivre.
# 
# In the [parsing chapter](parsing.ipynb) we saw how to develop a syntactic parser based on context-free grammars (CFG). In this chapter we will see how to develop a syntactic parser based on a different paradigm, dependency parsing.
# 
# The key idea in dependency parsing is that syntactic structure of lexical items, linked by binary
# asymmetric relations called dependencies [[Nivre, 2008]](http://www.mitpressjournals.org/doi/pdf/10.1162/coli.07-056-R1-07-027). More simply, syntax is represented as directed edges between words, commonly referred to arcs in the literature. Thus unlike the trees in [CFG parsing](parsing.ipynb), dependency trees have only terminal nodes (the words of the sentences), which can appear as leaves as and non-leaf nodes. Here is the dependency graph for the sentence:
# 
# > "Economic news had little effect on financial markets"

# In[3]:


tokens = ["Economic", "news", "had", "little", "effect", "on", "financial", "markets", "."]
arcs = {(1,0,"amod"), (2,1,"nsubj"), (2, 4, "dobj"), (4,3,"amod"), (4,5, "prep"), (5,7,"pmod"), (7,6,"amod")}

#transition.render_tree(tokens, arcs)

transition.render_displacy(*transition.to_displacy_graph(arcs, tokens))


# Following [Nivre (2008)](http://www.mitpressjournals.org/doi/pdf/10.1162/coli.07-056-R1-07-027), the dependency parse of a sentence is a graph \\(\y = (\x, \a) \\) where:
# - \\(\x = \\{0, 1, ... N\\}\\) are the nodes, each of them representing one of the N words in the sentence
# - \\(\a \subseteq \x \times \x \times L \\) are labeled directed arcs between the words, with labels coming from a predefined set \\(L\\).
# 
# For a graph \\(\y\\) to be a valid dependency tree, the following constrains must be obeyed:
# 
# - rooted: node 0 is the root and there can be no incoming arcs to it
# - acyclic: no directed cycles exist in \\(\y\\) 
# - single-headed: each node can only one head node, i.e. only one incoming arc
# - connected: there is an undirected path between every pair of nodes in the graph. 
# 
# Note that the dependency parse tree shown above is not connected since the period is left without an arc (it is a dependency *forrest*). To ensure that dependency trees are well-formed, we introduce a ROOT node which points to the main verb of the sentence, as well as the punctuation.

# In[3]:


tokens = ["ROOT", "Alice", "saw", "Bob"]
arcs = {(0,2, "root"), (2,1,"nsubj"), (2,3,"dobj")}

transition.render_displacy(*transition.to_displacy_graph(arcs, tokens))


# A straightforward approach to dependency parsing would be given a sentence \\(\x\\), to enumerate over all valid graphs for the sentence $\y\in\Ys_x$ and score them using an appropriate function \\(s_\params(\x,\y)\\), a case of a [structured prediction](structured_prediction.ipynb) problem. While such an approach is possible and there has been a lot of work often referred to as graph-based parsing, e.g. [McDonald et al. (2006)](http://www.ryanmcd.com/papers/MS-CIS-05-11.pdf), in this note we will focus on transition-based approaches, which decompose the task into a sequence of label predictions that can be learned with a classifier.

# To perform transition-based parsing we first need to define a transition system consisting of the following elements:
# 
# **Configuration**:
# - Stack \\(S\\): a last-in, first-out memory to keep track of words to process later
# - Buffer \\(B\\): words not processed so far
# - Arcs \\(A\\): the dependency edges predicted so far
# 
# We further define two special configurations:
# - initial: buffer is initialised to the words in the sentence, stack and arks are empty
# - terminal: buffer is empty
# 
# **Actions**:
# - shift: push the word at the top of the buffer to the stack \\((S, i|B, A)\rightarrow(S|i, B, A)\\)
# - reduce: pop the word at the top of the stack if it has a head \\((S|i, B, A)\rightarrow(S, B, A)\\)
# - rightArc-label: create a labeled arc from the token at the top of the stack \\(i\\) to the token at the top of the buffer \\(j\\) \\((S|i, j|B, A) \rightarrow (S|i|j, B, A\cup\{(i,j,l)\})\\). Shift the token on top of the buffer to the stack.
# - leftArc-label: create a labeled arc from the token at the top of the buffer \\(j\\) to the token at the top of the stack \\(i\\) if \\(i\\) has no head \\((S|i, j|B, A) \rightarrow (S, j|B, A\cup\{(j,i,l)\})\\). Reduce the token on top of the stack.
# 
# Below we show a simple implementation of this transition system:

# In[4]:


from collections import deque

class Configuration():
    def __init__(self, tokenized_sentence):
        # This implements the initial configuration for a sentence
        self.arcs = set()
        self.buffer = deque()
        self.sentence = tokenized_sentence
        for idx, token in enumerate(tokenized_sentence[1:], start=1):
            self.buffer.append({"index": idx, "form": token})
        self.stack = [{"index": 0, "form": "ROOT"}]
        
import copy
def parse(tokenized_sentence, actions):
    # This stores the (configuration, action) tuples generated
    transitions = []
    
    # Initialize the configuration
    configuration = Configuration(tokenized_sentence)
    transitions.append((copy.deepcopy(configuration), ""))
    
    for action in actions:
        if action == "shift":
            token = configuration.buffer.popleft()
            configuration.stack.append(token)
        elif action.startswith("leftArc"):
            head = configuration.stack[-1]
            dependent = configuration.stack.pop(-2)
            label = action.split("-")[1]
            configuration.arcs.add((int(head["index"]), int(dependent["index"]), label))
        elif action.startswith("rightArc"):
            head = configuration.stack[-2]
            dependent = configuration.stack.pop()
            label = action.split("-")[1]
            configuration.arcs.add((int(head["index"]), int(dependent["index"]), label))            
        
        transitions.append((copy.deepcopy(configuration), action))
    
    if len(configuration.buffer) == 0 and len(configuration.stack) <= 1:
        transitions.append((copy.deepcopy(configuration), ""))
    return transitions


# Let's see how we can parse the example sentence using this transition system defined above assuming we are given the correct sequence of actions:

# In[5]:


tokenized_sentence = ["ROOT", "Alice", "saw", "Bob"]
actions = ["shift","shift", "leftArc-nsubj", "shift", "rightArc-dobj", "rightArc-root"]

transitions = parse(tokenized_sentence, actions)

transition.render_transitions_displacy(transitions, tokenized_sentence)


# The key idea in transition dependency parsing is that we converted graph prediction, a structured prediction problem, into a sequence of classification predictions that are guaranteed to give us a valid dependency tree. Thus a transition based dependency parser is a classifier that predicts the correct action for the current configuration. 
# 
# The choice of classifier is free; we can use any classifier we like, for example [loglinear classification models](doc_classify.ipynb). The features are defined in order to describe the configuration and the previous actions taken in a way that helps the classifier predict the correct action. For example, encoding that the words on top of the buffer and the stack are "on" and "effect" respectively is highly indicative of the rightArc-prep action to create an arc between them. Such lexicalized features though can be quite sparse, which is why recent work has looked into continuous representations for them ([Chen and Manning, 2014](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)).
# 
# The next question is where we get the training data to train the classifier, i.e. configurations labeled with the correct action. However the training data we are typically provided with consists of sentences labeled with the final dependency tree. Thus, we need to develop a function that given a sentence and its dependency tree can "reverse-engineer" the parsing process to recover the sequence of actions that was used construct it. This function is often referred to as the oracle (name inspired by [the ones in antiquity](https://en.wikipedia.org/wiki/Oracle)), and it is usually a set of heuristics that returns the correct sequence of actions by looking at the dependency tree. A different way to think about it is that of a human annotator demonstrating how to construct the parse tree using the transition system defined.
# 
# The transition system we defined above is known as the arc-eager system due to [Nivre (2003)](http://stp.lingfil.uu.se/~nivre/docs/iwpt03.pdf). Different transition systems have been proposed, another popular choice being the arc-standard transition system that has three actions, left-arc, right-arc and shift that are defined differently compared to the arc-eager ones. As expected, different transition systems have different oracles to extract configurations labeled with the correct transition action from sentences annotated with dependency trees.

# An important restriction that both the arc-eager and arc-standard transition systems have is that they can only produce  projective dependency trees, i.e. trees that when they are drawn having the words on a fixed left-to-right order their arcs do not cross. However this restriction is violated when long-distance dependencies and free word order need to be taken into account, as in the sentence below in which  (

# In[6]:


tokens = ["ROOT", "What", "did", "economic", "news", "have", "little", "effect", "?"]
arcs = {(0,5, "root"), (0,9,"p"), (8,1,"pobj"), (5,2,"aux"), (4,3,"amod"), (5,4,"nsubj"), (5, 7, "dobj"), (7,6,"amod"), (5,8, "prep"), (6,8,"pmod"), (8,7,"amod")}

transition.render_displacy(*transition.to_displacy_graph(arcs, tokens))


# To produce non-projective dependency trees such as the on in the example above, more complex transition systems employing mulitple stacks have been developed ([Gomez-Rodriguez and Nivre, 2010](http://www.aclweb.org/anthology/P10-1151)).
