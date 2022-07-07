#!/usr/bin/env python
# coding: utf-8

# # Constituent Parsing Exercises
# 
# 
# 
# In the lecture we took a look at a simple tokenizer and sentence segmenter. In this exercise we will expand our understanding of the problem by asking a few important questions, and looking at the problem from a different perspectives.

# ##  <font color='green'>Setup 1</font>: Load Libraries

# In[1]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# %cd .. \nimport sys\nsys.path.append("..")\nimport math \nimport statnlpbook.util as util\nimport statnlpbook.parsing as parsing\n')


# ## <font color='blue'>Task 1</font>: Understanding parsing
# 
# Be sure you understand [grammatical categories and structures](http://webdelprofesor.ula.ve/humanidades/azapata/materias/english_4/grammatical_categories_structures_and_syntactical_functions.pdf) and brush up on your [grammar skils](http://www.ucl.ac.uk/internet-grammar/intro/intro.htm).
# 
# Then re-visit the [Enju online parser](http://www.nactem.ac.uk/enju/demo.html), and parse the following sentences...
# 
# What is wrong with the parses of the following sentences? Are they correct?
# - Fat people eat accumulates.
# - The fat that people eat accumulates in their bodies.
# - The fat that people eat is accumulating in their bodies.
# 
# What about these, is the problem in the parser or in the sentence?
#   - The old man the boat.
#   - The old people man the boat.  
# 
# These were examples of garden path sentences, find out what that means.
# 
# What about these sentences? Are their parses correct?
#   - Time flies like an arrow; fruit flies like a banana.
#   - We saw her duck.

# ## <font color='blue'>Task 2</font>: Parent Annotation
# 
# 

# Reminisce the lecture notes in parsing, and the mentioned parent annotation. (grand)*parents, matter - knowing who the parent is in a tree gives a bit of context information which can later help us with smoothing probabilities, and approaching context-dependent parsing.
# 
# in that case, each non-terminal node should know it's parent. We'll do this exercise on a single tree, just to play around a bit with trees and their labeling.
# 
# 
# Given the following tree:

# In[2]:


x = ('S', [('Subj', ['He']), ('VP', [('Verb', ['shot']), ('Obj', ['the', 'elephant']), ('PP', ['in', 'his', 'pyjamas'])])])
parsing.render_tree(x)


# Construct an `annotate_parents` function which will take that tree, and annotate its parents. The final annotation result should look like this:

# In[3]:


y = ('S^?', [('Subj^S', ['He']), ('VP^S', [('Verb^VP', ['shot']), ('Obj^VP', ['the', 'elephant']), ('PP^VP', ['in', 'his', 'pyjamas'])])])
parsing.render_tree(y)


# ## Solutions
# 
# You can find the solutions to this exercises [here](parsing_solutions.ipynb)
