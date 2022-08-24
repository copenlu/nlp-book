#!/usr/bin/env python
# coding: utf-8

# # Relation Extraction Exercises
# 
# 
# 
# In the lecture we took a look at [four different types of relation extraction](chapters/relation_extraction.ipynb). In this exercise we will expand our understanding of those methods by improving on some of the issues we observed with the basic solutions presented in the lecture.

# ##  <font color='green'>Setup 1</font>: Load Libraries

# In[1]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# %cd .. \nimport sys\nsys.path.append("../statnlpbook/")\nimport math \nimport tfutil,ie\n')


# ## <font color='blue'>Task 1</font>: Shortest Path for Relation Extraction
# 
# Recall that for all the presented relation extraction methods, we were interested in determining the relation between two entities.
# A core component of all the relation extraction methods presented in the lecture was to determine the shortest path between those two entities. Features for the relation extraction model were then based on that shortest path.  
# 
# The solution to determining the shortest path presented in the lecture was to define it as the words occurring between the two entities. 
# 
# - What is a possible problem with this way of determining the shortest path?
# - What is a better way of producing such a shortest path?
# 
# - Improve the relation extraction methods by implementing an alternative shortest path extraction method, the old one is "sentenceToShortPath(sent)".
# - Apply the new shortest path extraction method to the relation extraction methods presented with the lecture. 
#     - Note that in order for this to work, the new shortest path extraction method will need to return a list of words which define the shortest path.
# - Observe the differences between the old and new path shortening method.
# 
# Hint: revisit the lecture materials on parsing.

# ## <font color='blue'>Task 2</font>: Shortest Path Features
# 
# 

# Recall that, in order to obtain features for supervised relation extraction, we transformed the shortest path between the two entities to word features using the built-in sklearn "CountVectorizer()".  
# 
# As discussed, this can lead to obtaining features which are too general (e.g. stopwords such as "a", "of"). A better approach would be to have features which are based on the syntax of the sentence.
# 
# - Implement a method which, for each sentence, returns a syntactic representation of that sentence
# - Narrow this syntactic representation down to words on the shortest path between the two entities
# 
# Hint: if you have completed Task 1, you should already have a solution for those the two exercises above.
# 
# - Use the syntactic representation as features for supervised learning. To do this, replace the "featTransform()" method. 
#     - Note that you define the whole syntactic path as a feature, or split it into several words. 
#     - For the latter, pass the syntactic path to the CountVectorizer() and set the n-gram range appropriately. It is set to "1, 1" by default, meaning it returns single words only. See the [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for the CountVectorizer.
# - Use a combination of the syntactic representation and words as features. In the literature, this is typically referred to as "lexico-syntactic features". Proceed in the same way as for syntax-only features above.
# - Observe the effect of choosing different features on the supervised relation extraction output.

# ## <font color='blue'>Task 2</font>: Helper Function
# 
# The current state of the art parser is [SyntaxNet](https://github.com/tensorflow/models/tree/master/syntaxnet), also available as Docker container [here](http://www.whycouch.com/2016/07/how-to-install-and-use-syntaxnet-and.html).
# It parses sentences into the [CoNLL-U format](http://universaldependencies.org/format.html).
# 
# The relation extraction data is already parsed with this parser, and you can load it like so:

# In[2]:


def loadSyntRepr(path="../data/ie/ie_training_data.sents.parse"):
    file = open(path, "r")
    
    sents = []
    s = []
    for l in file:
        l = l.strip()
        data = l.split("\t")
        print(data)
        if len(data) < 10:
            sents.append(s)
            s = []
            continue
            
        wid, token, lemma, upostag, xpostag, feats, head, deprel, deps, misc = data
        s.append(data)
        
    if len(s) != 0:
        sents.append(s)
    
    return sents
        
sents = loadSyntRepr()


# In[3]:


sents

