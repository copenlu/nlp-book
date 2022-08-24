#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n<style>\n.rendered_html td {\n    font-size: xx-large;\n    text-align: left; !important\n}\n.rendered_html th {\n    font-size: xx-large;\n    text-align: left; !important\n}\n</style>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\nimport sys\nsys.path.append("../statnlpbook/")\n#import util\nimport ie\nimport tfutil\nimport random\nimport numpy as np\nimport tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()\nnp.random.seed(1337)\ntf.set_random_seed(1337)\n\n#util.execute_notebook(\'relation_extraction.ipynb\')\n')


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

# In[3]:


get_ipython().run_line_magic('load_ext', 'tikzmagic')


# # Relation Extraction

# ## Relation Extraction Approaches
# * **Pattern-Based** Relation Extraction:
#     * Extract relations via manually defined textual patterns
# * **Bootstrapping**:
#     * Iterative pattern-based relation extraction

# * **Supervised** Relation Extraction:
#     * Train supervised model from manually labelled training examples
# * **Distantly Supervised** Relation Extraction:
#     * Supervised model with automatically annotated training data

# * **Universal Schema** Relation Extraction:
#     * Model relation types and surface forms in same semantic space
# * **Transfer Learning** for Relation Extraction:
#     * Use word or sentence embeddings trained on larger dataset, see representation learning lecture slides ([intro](dl-representations.ipynb), [language models](language_models_slides.ipynb), [RNN applications](rnn_slides_ucph.ipynb))

# 
# ## Relation Extraction: Running Example
# * Extracting "method used for task" relations from sentences in computer science publications
# * The first step would normally be to detect pairs of arguments $\mathcal{E}$. For simplicity, our training data already contains those annotations.
# 

# ## Pattern-Based Extraction
# * The simplest relation extraction method
# * Set of textual patterns for each relation
# * Assign labels to entity pairs whose sentences match that pattern
#     * Labels: relation types or "NONE"
# * Data: entity pairs $\mathcal{E}$, patterns $A$, labels $Y$

# In[4]:


training_patterns, training_entpairs = ie.readLabelledPatternData()
# Training patterns and entity pairs for relation 'method used for task'
list(zip(training_patterns[:3], training_entpairs[:3]))


# * Patterns: sentences where entity pairs are blanked with placeholder 'XXXXX'
# * Here: 
#     * Only one relation, 'method used for task'
#     * Manually defined patterns
# * Labels for training data, no labels for test data
# * Task: 'predict' labels for test data

# In[ ]:


testing_patterns, testing_entpairs = ie.readPatternData()
# Testing patterns and entity pairs
list(zip(testing_patterns[0:3], testing_entpairs[:3]))


# **Scoring model**: determine which instance belongs to which relation
#   * A pattern scoring model \\(s_{\params}(\x,y)\\) only has one parameter
#   * Assignes scores to each relation label \\(y\\) proportional to the matches with the set of textual patterns
#   * The final label assigned to each instance is then the one with the highest score

# * Here, our pattern scoring model is even simpler since we only have patterns for one relation
#     * Final label: 'method used for task' if there is a match with a pattern, 'NONE' if no match

# ### Closer look at pattern matching
# * Patterns in the training data: sentences where entity pairs are blanked with 'XXXXX'
# * Suggested improvement:
#     * We could use those patterns to find more sentences
#     * However, we are not likely to find many since patterns are very specific to the example
# * We need to **generalise** those patterns to less specific ones
#     * e.g. define sequence of words between each entity pair as a pattern

# In[ ]:


def sentence_to_short_seq(sent):
    """
    Returns the sequence between two arguments in a sentence, where the arguments have been masked
    Args:
        sent: the sentence
    Returns:
        the sequence between to arguments
    """
    sent_toks = sent.split(" ")
    indeces = [i for i, ltr in enumerate(sent_toks) if ltr == "XXXXX"]
    pattern = " ".join(sent_toks[indeces[0]+1:indeces[1]])
    return pattern

print(training_patterns[0])
sentence_to_short_seq(training_patterns[0])


# * There are many different alternatives to this method for shortening patterns
# * **Thought exercise**: 
#     * what is a possible problem with this way of shortening patterns and what are better ways of generalising patterns?
#     
# Enter answer: https://tinyurl.com/y5y9fwpe

# ### Revised pattern extraction approach
#   * Define sentence shortening / **pattern generalisation method**
#   * Apply patterns to testing instances to classify them into 'method used for task' and 'NONE'
# 
# Example: return instances which contain a 'method used for task' pattern

# In[ ]:


def pattern_extraction(training_sentences, testing_sentences):
    """
    Given a set of patterns for a relation, searches for those patterns in other sentences
    Args:
        sent: training sentences with arguments masked, testing sentences with arguments masked
    Returns:
        the testing sentences which the training patterns appeared in
    """
    # convert training and testing sentences to short paths to obtain patterns
    training_patterns = set([sentence_to_short_seq(train_sent) for train_sent in training_sentences])
    testing_patterns = [sentence_to_short_seq(test_sent) for test_sent in testing_sentences]
    # look for match of training and testing patterns
    testing_extractions = []
    for i, testing_pattern in enumerate(testing_patterns):
        if testing_pattern in training_patterns: # look for exact matches of patterns
            testing_extractions.append(testing_sentences[i])
    return testing_extractions

pattern_extraction(training_patterns[:300], testing_patterns[:300])


# * Problems with approach: 
#     * set of patterns has to be defined manually
#     * the model does not learn new patterns
#     
# 
# * Next: approach which addresses those two shortcomings
# 

# ## Bootstrapping
# 
# * Input: a set of entity pairs
# * Overall idea: extract patterns and entity pairs **iteratively**
# * One of the first algorithms: [DIPRE (Sergey Brin, 1999)](http://ilpubs.stanford.edu:8090/421/1/1999-65.pdf)
# * Two helper methods: 
#     * *use entity pairs* to find/generate (more) patterns
#     * *apply patterns* to find entity pairs
# 

# In[ ]:


# use patterns to find more entity pairs
def search_for_entpairs_by_patterns(training_patterns, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_pattern in enumerate(testing_patterns): # iterate over patterns
        if testing_pattern in training_patterns: # if there is an exact match of a pattern
            testing_extractions.append(testing_sentences[i]) # add the corresponding sentence
            appearing_testing_patterns.append(testing_pattern) # add the pattern
            appearing_testing_entpairs.append(testing_entpairs[i]) # add the entity pairs
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs


# In[ ]:


# use entity pairs to find more patterns
def search_for_patterns_by_entpairs(training_entpairs, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_entpair in enumerate(testing_entpairs): # iterate over entity pairs
        if testing_entpair in training_entpairs: # if there is an exact match of an entity pair
            testing_extractions.append(testing_sentences[i]) # add the corresponding sentence
            appearing_testing_entpairs.append(testing_entpair) # add the entity pair
            appearing_testing_patterns.append(testing_patterns[i]) # add the pattern
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs


# The two helper functions are then applied iteratively:

# In[ ]:


def bootstrapping_extraction(train_sents, train_entpairs, test_sents, test_entpairs, num_iter=10):
    """
    Given a set of patterns and entity pairs for a relation, extracts more patterns and entity pairs iteratively
    Args:
        train_sents: training sentences with arguments masked
        train_entpairs: training entity pairs
        test_sents: testing sentences with arguments masked
        test_entpairs: testing entity pairs
    Returns:
        the testing sentences which the training patterns or any of the inferred patterns appeared in
    """
    # convert training and testing sentences to short paths to obtain patterns
    train_patterns = set([sentence_to_short_seq(s) for s in train_sents])
    train_patterns.discard("in") # too general, remove this
    test_patterns = [sentence_to_short_seq(s) for s in test_sents]
    test_extracts = []

    # iteratively get more patterns and entity pairs
    for i in range(1, num_iter):
        print("Number extractions at iteration", str(i), ":", str(len(test_extracts)))
        print("Number patterns at iteration", str(i), ":", str(len(train_patterns)))
        print("Number entpairs at iteration", str(i), ":", str(len(train_entpairs)))
        # get more patterns and entity pairs
        test_extracts_e, ext_test_patterns_e, ext_test_entpairs_e = search_for_patterns_by_entpairs(train_entpairs, test_patterns, test_entpairs, test_sents)
        test_extracts_p, ext_test_patterns_p, ext_test_entpairs_p = search_for_entpairs_by_patterns(train_patterns, test_patterns, test_entpairs, test_sents)
        # add them to the existing patterns and entity pairs for the next iteration
        train_patterns.update(ext_test_patterns_p)
        train_patterns.update(ext_test_patterns_e)
        train_entpairs.extend(ext_test_entpairs_p)
        train_entpairs.extend(ext_test_entpairs_e)
        test_extracts.extend(test_extracts_p)
        test_extracts.extend(test_extracts_e)

    return test_extracts, test_entpairs


# In[ ]:


test_extracts, test_entpairs = ie.bootstrappingExtraction(training_patterns[:20], training_entpairs[:20], testing_patterns, testing_entpairs, 10)


# Problem: 
# * with each iteration, the number of pattern, entity pairs and extractions increases
# * however, they are less correct

# In[ ]:


train_patterns = set(sentence_to_short_seq(s) for s in training_patterns[:20])
test_patterns = set(sentence_to_short_seq(s) for s in test_extracts)

# patterns that do not co-occur with first set of entity pairs
for p in test_patterns:
    if p not in train_patterns:
        print(p)


# * One of the reasons is that the semantics of the pattern shifts ("**semantic drift**")
#     * here we try to find new patterns for 'method used for task'
#     * but because the instances share a similar context with other relations, the patterns and entity pairs iteratively move away from the 'method used in task' relation
#     * Another example: 'employee-at', 'student-at' -> many overlapping contexts
# 

# * One solution: confidence values for each entity pair and pattern
#     * E.g., we might want to avoid entity pairs or patterns which are too general and penalise them

# In[ ]:


from collections import Counter
te_cnt = Counter()
for te in test_extracts:
    te_cnt[sentence_to_short_seq(te)] += 1
print(te_cnt)


# * Such a 'noisy' pattern is e.g. 'in': it matches many contexts that are not 'method used for task' 
# * **Thought exercise**: 
#     * how would a confidence weighting for patterns work here?
#     
# Enter answer: https://tinyurl.com/y3dxzjuz

# ## Supervised Relation Extraction
# * Follow the supervised learning paradigm
#     * We have already seen for other structured prediction tasks
# * Scoring model \\(s_{\params}(\x,y)\\) is estimated based on training sentences $\mathcal{X}$ and their labels $\mathcal{Y}$
# * We can use range of different classifiers, e.g. a logistic regression model or an SVM
# * At testing time, the predict highest-scoring label for each testing instance, i.e. $$ \y^* = \argmax_{\y\in\Ys} s(\x,\y) $$
# 

# ### Example
# * The training data consists again of patterns, entity pairs and labels
# * This time, the given labels for the training instances are 'method used for task' or 'NONE', i.e. we have positive and negative training data

# In[ ]:


training_sents, training_entpairs, training_labels = ie.readLabelledData()
print("Manually labelled data set consists of", training_labels.count("NONE"), 
          "negative training examples and", training_labels.count("method used for task"), "positive training examples\n")
list(zip(training_sents[:3], training_entpairs[:3], training_labels[:3]))


# Feature extraction
# * Transform training and testing data to features 
# * Typical features: shortest dependency path between two entities (see [parsing slides](dependency_parsing_slides.ipynb))
# * We assume again that entity pairs are already recognised
# 
# * Example shown with sklearn's built-in feature extractor to transform sentences to n-grams

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def feat_transform(sents_train, sents_test):
    cv = CountVectorizer()
    cv.fit(sents_train)
    print(cv.get_params())
    features_train = cv.transform(sents_train)
    features_test = cv.transform(sents_test)
    return features_train, features_test, cv


# Define a model, here: sklearn, using one of their built-in classifiers and a prediction function

# In[ ]:


from sklearn.linear_model import LogisticRegression

def model_train(feats_train, labels):
    model = LogisticRegression(penalty='l2', solver='liblinear')  # logistic regression model with l2 regularisation
    model.fit(feats_train, labels) # fit the model to the transformed training data
    return model

def predict(model, features_test):
    """Find the most compatible output class"""
    preds = model.predict(features_test) # this returns the predicted labels
    #preds_prob = model.predict_proba(features_test)  # this returns probablities instead of labels
    return preds


# Helper function for debugging that determines the most useful features learned by the model

# In[ ]:


def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


# Supervised relation extraction algorithm:

# In[ ]:


def supervised_extraction(train_sents, train_entpairs, train_labels, test_sents, test_entpairs):
    """
    Given pos/neg training instances, train a logistic regression model with simple BOW features and predict labels on unseen test instances
    Args:
        train_sents: training sentences with arguments masked
        train_entpairs: training entity pairs
        train_labels: labels of training instances
        test_sents: testing sentences with arguments masked
        test_entpairs: testing entity pairs
    Returns:
        predictions for the testing sentences
    """

    # convert training and testing sentences to short paths to obtain patterns
    train_patterns = [sentence_to_short_seq(test_sent) for test_sent in train_sents]
    test_patterns = [sentence_to_short_seq(test_sent) for test_sent in test_sents]

    # extract features
    features_train, features_test, cv = feat_transform(train_patterns, test_patterns)

    # train model
    model = model_train(features_train, train_labels)

    # show most common features
    show_most_informative_features(cv, model)

    # get predictions
    predictions = predict(model, features_test)

    return predictions


# In[ ]:


testing_preds = supervised_extraction(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)
list(zip(testing_preds, testing_patterns, testing_entpairs))[:10]


# ### Model Inspection
# * Some the features are common words (i.e. 'stop words', such as 'is') and very broad
# * Others are very specific and thus might not appear very often
# * Typically these problems can be mitigated by using more involved features, e.g. based on syntax
# * Current model ignores entity pairs, only featurises the path between the entity pairs
#     * We will later examine a model that also takes entity pairs into account
# 

# * Finally, the model requires manually annotated training data, which might not always be available
# * Next, we will look at a method that provides a solution for the latter

# ## Distant Supervision
# * Supervised learning typically requires large amounts of hand-labelled training examples
# * It is **time-consuming and expensive** to manually label examples
#     * It is desirable to find ways of automatically or semi-automatically producing more training data
#     * We have already seen one example of this, bootstrapping
# * Downside of bootstrapping: **semantic drift** 
#     * due to the iterative nature of finding good entity pairs and patterns
# * Alternative: distant supervision

# * We still have a set of entity pairs $\mathcal{E}$, their relation types $\mathcal{R}$ and a set of sentences $\mathcal{X}$ as an input
#     * but we do **not require pre-defined patterns**
# * Instead, entity pairs and relations are obtained from a **knowledge resource**, e.g. the [Wikidata knowledge base](https://www.wikidata.org), Yago or Freebase
# * Those are used to automatically label all sentences with relations
# * Afterwards: supervised learning

# ![image.png](attachment:image.png)

# In[ ]:


def distantly_supervised_labelling(kb_entpairs, unlab_sents, unlab_entpairs):
    """
    Label instances using distant supervision assumption
    Args:
        kb_entpairs: entity pairs for a specific relation
        unlab_sents: unlabelled sentences with entity pairs anonymised
        unlab_entpairs: entity pairs which were anonymised in unlab_sents

    Returns: pos_train_sents, pos_train_enpairs, neg_train_sents, neg_train_entpairs

    """
    train_sents, train_entpairs, train_labels = [], [], []
    for i, unlab_entpair in enumerate(unlab_entpairs):
        # if the entity pair is a KB tuple, it is a positive example for that relation
        if unlab_entpair in kb_entpairs:  
            train_entpairs.append(unlab_entpair)
            train_sents.append(unlab_sents[i])
            train_labels.append("method used for task")
        else: # else, it is a negative example for that relation
            train_entpairs.append(unlab_entpair)
            train_sents.append(unlab_sents[i])
            train_labels.append("NONE")

    return train_sents, train_entpairs, train_labels


# In[ ]:


def distantly_supervised_extraction(kb_entpairs, unlab_sents, unlab_entpairs, test_sents, test_entpairs):
    # training_data <- Find training sentences with entity pairs
    train_sents, train_entpairs, train_labels = distantly_supervised_labelling(kb_entpairs, unlab_sents, unlab_entpairs)
    
    print("Distantly supervised labelling results in", train_labels.count("NONE"), 
          "negative training examples and", train_labels.count("method used for task"), "positive training examples")
    
    # training works the same as for supervised RE
    return supervised_extraction(train_sents, train_entpairs, train_labels, test_sents, test_entpairs)


# In[ ]:


kb_entpairs, unlab_sents, unlab_entpairs = ie.readDataForDistantSupervision()
#print(len(kb_entpairs), "'KB' entity pairs for relation 'method used for task' :", kb_entpairs[0:5])
#print(len(unlab_entpairs), 'all entity pairs')
testing_preds = distantly_supervised_extraction(kb_entpairs, unlab_sents, unlab_entpairs, testing_patterns, testing_entpairs)
list(zip(testing_preds, testing_patterns, testing_entpairs))[:10]


# * Here, results are the same as for supervised relation extraction, because the distant supervision heuristic identified the same positive and negative training examples as in the manually labelled dataset
# * In practice, the distant supervision heuristic typically leads to noisy training data

# * **Overlapping relations**
#     * For instance, 'prof-at' entails 'employee-of' and there are some overlapping entity pairs between the relations 'employee-of' and 'student-at'

# * **Ambiguous entities**
#     * e.g. 'EM' has many possible meanings, only one of which is 'Expectation Maximisation', see [the Wikipedia disambiguation page for the acronym](https://en.wikipedia.org/wiki/EM).

# * **Mention vs. type annotations**
#     * Not every sentence a positive entity pair appears in actually expresses that relation
#     * e.g. 'lives-in', 'works-in', etc.

# ## Universal Schema
# * For pattern-based and bootstrapping, we looked for simplified paths between entity pairs $\mathcal{E}$ expressing a relation $\mathcal{R}$ defined beforehand
#     * This **restricts the relation extraction problem to known relation types** $\mathcal{R}$
#     * To overcome that limitation, we could have defined new relation types for certain simplified relation types on the spot
#     * Here: more principled solution

# ## Universal Schema
# * Goal: overcome limitation of having to pre-define relations, within the supervised learning paradigm
# * This is possible by viewing relation paths **as relations themselves**
# * Simplified paths between entity pairs and relations defined in knowledge base are **no longer considered separately**
#     * instead they are **modelled in the same space**
# 

# The space of entity pairs and relations is defined by a matrix:
# 
# |  | demonstrates XXXXX for XXXXXX | XXXXX is capable of XXXXXX | an XXXXX model is employed for XXXXX | XXXXX decreases the XXXXX | method is used for task |
# | ------ | ----------- |
# | 'text mining', 'building domain ontology' | 1 |  |  |  | 1 |
# | 'ensemble classifier', 'detection of construction materials' |  |  | 1 |  | 1 |
# | 'data mining', 'characterization of wireless systems performance'|  | 1 |  |  | ? |
# | 'frequency domain', 'computational cost' |  |  |  | 1 | ? |

# * 'method is used for task' is a relation defined by a KB schema
# * The other relations are surface pattern relations
# * Where an entity pair and a KB relation or surface pattern relation co-occur, this is signified by a '1'
# * For some of the entities and surface pairs, a label for 'method used for task' is available, whereas for others, it is not (signified by the '?')
# * We can use the same data as for supervised RE, as merely the data representation and model are different

# ## Model: Neural Matrix Factorisation for Recommender Systems
# 
# <img src="dl-applications-figures/neural_mf.png" width=800/> 
# 
# Source: https://arxiv.org/abs/1707.07435

# * Neural Matrix Factorisation model for Recommender Systems is adapted for relation extraction
#    * Users -> relations
#    * Items -> entity pairs

# * **Prediction task**: learn to fill in the empty cells in the matrix
#     * i.e. turn the '?'s into 0/1 predictions for predicting the 'method for task' relation
#     * estimate, for a relation $\mathcal{r}$ such as 'method is used for task' and an unseen entity pair such as $\mathcal{e}$, e.g. ('frequency domain', 'computational cost'), what the probability $\mathcal{p(y_{r,e} = 1)}$ is.
# * **Training objective**: 
#     * learning to distinguish between entity pairs and relations which co-occur in our training data (positive instances) and entity pairs and relations which are not known to co-occur (negative instances, i.e. the empty cells)
#     * logistic loss, or ranking objective

# * **Features representations**:
#     * embeddings for entity pairs and relations, see [word representation](chapters/dl-representations.ipynb) and [RNN slides](chapters/rnn_slides_ucph.ipynb)
# * **Model**:
#     * dot product between entity and relation embeddings

# ![image.png](attachment:image.png)

# * **Training instance**: consists of a surface pattern or KB relation $\mathcal{r_{pos}}$ and an entity pair  $\mathcal{e_{pos}}$ the relation co-occurs with, as well as a relation $\mathcal{r_{neg}}$ and a entity pair $\mathcal{e_{neg}}$ that do not co-occur in the training data
# * **Positive relations and entity pairs**: directly taken from the annotated data
# * **Negative entity pairs and relations**: *sampled randomly* from data points which are represented by the empty cell in the matrix above

# In[ ]:


# data reading
training_sents, training_entpairs, training_labels = ie.readLabelledData()

# split positive and negative training data
pos_train_ids, neg_train_ids = ie.split_labels_pos_neg(training_labels + training_labels)

training_toks_pos = [t.split(" ") for i, t in enumerate(training_sents + training_labels) if i in pos_train_ids]
training_toks_neg = [t.split(" ") for i, t in enumerate(training_sents + training_labels) if i in neg_train_ids]

training_ent_toks_pos = [" || ".join(t).split(" ") for i, t in enumerate(training_entpairs + training_entpairs) if i in pos_train_ids]
training_ent_toks_neg = [" || ".join(t).split(" ") for i, t in enumerate(training_entpairs + training_entpairs) if i in neg_train_ids]
testing_ent_toks = [" || ".join(t).split(" ") for t in testing_entpairs]

# print length statistics
lens_rel = [len(s) for s in training_toks_pos + training_toks_neg]
lens_ents = [len(s) for s in training_ent_toks_pos + training_ent_toks_neg + testing_ent_toks]
print("Max relation length:", max(lens_rel))
print("Max entity pair length:", max(lens_ents))


# In[ ]:


# vectorise data (assign IDs to words)
count_rels, dictionary_rels, reverse_dictionary_rels = ie.build_dataset(
        [token for senttoks in training_toks_pos + training_toks_neg for token in senttoks])

count_ents, dictionary_ents, reverse_dictionary_ents = ie.build_dataset(
        [token for senttoks in training_ent_toks_pos + training_ent_toks_neg for token in senttoks])

print(reverse_dictionary_rels)


# In[ ]:


# transform sentences to IDs, pad vectors for each sentence so they have same length
rels_train_pos = [ie.transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in training_toks_pos]
rels_train_neg = [ie.transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in training_toks_neg]
ents_train_pos = [ie.transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in training_ent_toks_pos]
ents_train_neg = [ie.transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in training_ent_toks_neg]

print(rels_train_pos[0], "\n", rels_train_pos[1])


# In[ ]:


# Negatively sample some entity pairs for training. Here we have some manually labelled neg ones, so we can sample from them.
ents_train_neg_samp = [random.choice(ents_train_neg) for _ in rels_train_neg]
    
ents_test_pos = [ie.transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in testing_ent_toks]
# Sample those test entity pairs from the training ones as for those we have neg annotations
ents_test_neg_samp = [random.choice(ents_train_neg) for _ in ents_test_pos]  

vocab_size_rels = len(dictionary_rels)
vocab_size_ents = len(dictionary_ents) 

# for testing, we want to check if each unlabelled instance expresses the given relation "method for task"
rels_test_pos = [ie.transform_dict(dictionary_rels, training_toks_pos[-1], max(lens_rel)) for _ in testing_patterns]
rels_test_neg_samp = [random.choice(rels_train_neg) for _ in rels_test_pos]


# In[ ]:


data = ie.vectorise_data(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)

rels_train_pos, rels_train_neg, ents_train_pos, ents_train_neg_samp, rels_test_pos, rels_test_neg_samp, \
    ents_test_pos, ents_test_neg_samp, vocab_size_rels, vocab_size_ents, max_lens_rel, max_lens_ents, \
    dictionary_rels_rev, dictionary_ents_rev = data
  
# setting hyper-parameters
batch_size = 4
repr_dim = 30 # dimensionality of relation and entity pair vectors
learning_rate = 0.001
max_epochs = 31


# In[ ]:


def create_model_f_reader(max_rel_seq_length, max_cand_seq_length, repr_dim, vocab_size_rels, vocab_size_cands):
    """
    Create a Model F Universal Schema reader (Tensorflow graph).
    Args:
        max_rel_seq_length: maximum sentence sequence length
        max_cand_seq_length: maximum candidate sequence length
        repr_dim: dimensionality of vectors
        vocab_size_rels: size of relation vocabulary
        vocab_size_cands: size of candidate vocabulary
    Returns:
        dotprod_pos: dot product between positive entity pairs and relations
        dotprod_neg: dot product between negative entity pairs and relations
        diff_dotprod: difference in dot product of positive and negative instances, used for BPR loss (optional)
        [relations_pos, relations_neg, ents_pos, ents_neg]: placeholders, fed in during training for each batch
    """


# In[ ]:


# Placeholders (empty Tensorflow variables) for positive and negative relations and entity pairs
# In each training epoch, for each batch, those will be set through mini batching

relations_pos = tf.placeholder(tf.int32, [None, max_lens_rel], name='relations_pos')  # [batch_size, max_rel_seq_len]
relations_neg = tf.placeholder(tf.int32, [None, max_lens_rel], name='relations_neg')  # [batch_size, max_rel_seq_len]

ents_pos = tf.placeholder(tf.int32, [None, max_lens_ents], name="ents_pos") # [batch_size, max_ent_seq_len]
ents_neg = tf.placeholder(tf.int32, [None, max_lens_ents], name="ents_neg") # [batch_size, max_ent_seq_len]


# In[ ]:


# Creating latent representations of relations and entity pairs
# latent feature representation of all relations, which are initialised randomly
relation_embeddings = tf.Variable(tf.random_uniform([vocab_size_rels, repr_dim], -0.1, 0.1, dtype=tf.float32),
                                   name='rel_emb', trainable=True)

# latent feature representation of all entity pairs, which are initialised randomly
ent_embeddings = tf.Variable(tf.random_uniform([vocab_size_ents, repr_dim], -0.1, 0.1, dtype=tf.float32),
                                      name='cand_emb', trainable=True)

# look up latent feature representation for relations and entities in current batch
rel_encodings_pos = tf.nn.embedding_lookup(relation_embeddings, relations_pos)
rel_encodings_neg = tf.nn.embedding_lookup(relation_embeddings, relations_neg)

ent_encodings_pos = tf.nn.embedding_lookup(ent_embeddings, ents_pos)
ent_encodings_neg = tf.nn.embedding_lookup(ent_embeddings, ents_neg)


# In[ ]:


# our feature representation here is a vector for each word in a relation or entity 
# because our training data is so small
# we therefore take the sum of those vectors to get a representation of each relation or entity pair
rel_encodings_pos = tf.reduce_sum(rel_encodings_pos, 1)  # [batch_size, num_rel_toks, repr_dim]
rel_encodings_neg = tf.reduce_sum(rel_encodings_neg, 1)  # [batch_size, num_rel_toks, repr_dim]

ent_encodings_pos = tf.reduce_sum(ent_encodings_pos, 1)  # [batch_size, num_ent_toks, repr_dim]
ent_encodings_neg = tf.reduce_sum(ent_encodings_neg, 1)  # [batch_size, num_ent_toks, repr_dim]


# In[ ]:


# measuring compatibility between positive entity pairs and relations
# used for ranking test data
dotprod_pos = tf.reduce_sum(tf.multiply(ent_encodings_pos, rel_encodings_pos), 1)

# measuring compatibility between negative entity pairs and relations
dotprod_neg = tf.reduce_sum(tf.multiply(ent_encodings_neg, rel_encodings_neg), 1)

# difference in dot product of positive and negative instances
# used for BPR loss (ranking loss)
diff_dotprod = tf.reduce_sum(tf.multiply(ent_encodings_pos, rel_encodings_pos) - tf.multiply(ent_encodings_neg, rel_encodings_neg), 1)


# To train this model, we define a loss, which tries to maximise the distance between the positive and negative instances. One possibility of this is the logistic loss.
# 
# $\mathcal{\sum -  \log(v_{e_{pos}} * a_{r_{pos}})} + {\sum \log(v_{e_{neg}} * a_{r_{neg}})}$

# Now that we have read in the data, vectorised it and created the universal schema relation extraction model, let's start training

# In[ ]:


# create the model / Tensorflow computation graph
dotprod_pos, dotprod_neg, diff_dotprod, placeholders = ie.create_model_f_reader(max_lens_rel, max_lens_ents, repr_dim, vocab_size_rels,
                          vocab_size_ents)

# logistic loss
loss = tf.reduce_sum(tf.nn.softplus(-dotprod_pos)+tf.nn.softplus(dotprod_neg))

# alternative: BPR loss
#loss = tf.reduce_sum(tf.nn.softplus(diff_dotprod))


# In[ ]:


data = [np.asarray(rels_train_pos), np.asarray(rels_train_neg), np.asarray(ents_train_pos), np.asarray(ents_train_neg_samp)]
data_test = [np.asarray(rels_test_pos), np.asarray(rels_test_neg_samp), np.asarray(ents_test_pos), np.asarray(ents_test_neg_samp)]

# define an optimiser. Here, we use the Adam optimiser
optimizer = tf.train.AdamOptimizer(learning_rate)
    
# training with mini-batches
batcher = tfutil.BatchBucketSampler(data, batch_size)
batcher_test = tfutil.BatchBucketSampler(data_test, 1, test=True)


# In[ ]:


with tf.Session() as sess:
    trainer = tfutil.Trainer(optimizer, max_epochs)
    trainer(batcher=batcher, placeholders=placeholders, loss=loss, session=sess)

    # we obtain test scores
    test_scores = trainer.test(batcher=batcher_test, placeholders=placeholders, model=tf.nn.sigmoid(dotprod_pos), session=sess)


# In[ ]:


# show predictions
ents_test = [ie.reverse_dict_lookup(dictionary_ents_rev, e) for e in ents_test_pos]
rels_test = [ie.reverse_dict_lookup(dictionary_rels_rev, r) for r in rels_test_pos]
testresults = sorted(zip(test_scores, ents_test, rels_test), key=lambda t: t[0], reverse=True)  # sort for decreasing score

print("\nTest predictions by decreasing probability:")
for score, tup, rel in testresults[:10]:
    print('%f\t%s\tREL\t%s' % (score, " ".join(tup), " ".join(rel)))


# Test prediction probabilities are obtained by scoring each test instances with:
# 
# $\mathcal{ \sigma  ( v_{e} \cdot a_{r} )}$
# 
# * Note that as input for the latent feature representation, we discarded words that only appeared twice
#     * Hence, for those words we did not learn a representation, denoted here by 'UNK'
# * This is also typically done for other feature representations, as if we only see a feature once, it is difficult to learn weights for it

# **Thought Exercises**: 
# * The scores shown here are for the relation 'method used for task'. However, we could also use our model to score the compatibility of entity pairs with other relations, e.g. 'demonstrates XXXXX for XXXXXX'. How could this be done here?
# * How could we get around the problem of unseen words, as described above?
# * What other possible problems can you see with the above formulation of universal schema relation extraction?
# * What possible problems can you see with using latent word representations?
# 
# Enter answer: https://tinyurl.com/yytql7wh

# ## Summary
# 
# Various relation extraction techniques:
# * Pattern-based extraction
# * Bootstrapping
# * Supervised
# * Distantly supervised extraction
# * Universal schema
# 
# Features often a mix of 
# * Syntax-based (relation path)
# * Representation learning based (word/sentence embedding)

# ## Background Material
# 
# * Jurafky, Dan and Martin, James H. (2016). Speech and Language Processing, Chapter 18 (Information Extraction): https://web.stanford.edu/~jurafsky/slp3/18.pdf
# 
# * Riedel, Sebastian and Yao, Limin and McCallum, Andrew and Marlin, Benjamin M. (2013). Relation extraction with Matrix Factorization and Universal Schemas. Proceedings of NAACL.  http://www.aclweb.org/anthology/N13-1008

# ## Further Reading
# 
# * Quan Wang, Zhendong Mao, Bin Wang, and Li Guo (2017). Knowledge Graph Embedding: A Survey of Approaches and Applications. https://persagen.com/files/misc/Wang2017Knowledge.pdf
#     * Have a look at this for more scoring functions for universal schema relation extraction
# * Shantanu Kumar (2017). A Survey of Deep Learning Methods for Relation Extraction. https://arxiv.org/pdf/1705.03645.pdf
# * Alex Ratner, Stephen Bach, Paroma Varma, Chris Ré (2018). Weak Supervision: The New Programming Paradigm for Machine Learning. https://dawn.cs.stanford.edu/2017/07/16/weak-supervision/
#     * Have a look at this for details on weak supervision and pointers to methods for learning with limited labelled data
# * Awesome relation extraction, curated list of resources on relation extraction. https://github.com/roomylee/awesome-relation-extraction
