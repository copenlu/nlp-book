#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# %cd .. \nimport sys\nsys.path.append("../statnlpbook/")\n#import statnlpbook.util as util\n#import statnlpbook.ie as ie\nimport ie,tfutil\n#import matplotlib\n')


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

# # Relation Extraction 
# Relation extraction (RE) is the task of extracting semantic relations between arguments. Arguments can either be general concepts such as "a company" (ORG), "a person" (PER); or instances of such concepts (e.g. "Microsoft", "Bill Gates"), which are called proper names or named entitites (NEs). An example for a semantic relation would be "founder-of(PER, ORG)". Relation extraction therefore often builds on the task of named entity recognition.
# 
# Relation extraction is relevant for many high-level NLP tasks, such as
# 
# * for question answering, where users ask questions such as "Who founded Microsoft?",
# * for information retrieval, which often relies on large collections of structured information as background data, and 
# * for text and data mining, where larger patterns in relations between concepts are discovered, e.g. temporal patterns about startups
# 
# 
# ## Relation Extraction as Structured Prediction
# We can formalise relation extraction as an instance of [structured prediction](/template/statnlpbook/02_methods/00_structuredprediction) where the input space $\mathcal{X}$ are pairs of arguments $\mathcal{E}$ and supporting texts $\mathcal{S}$ those arguments appear in. The output space $\mathcal{Y}$ is a set of relation labels such as $\Ys=\{ \text{founder-of},\text{employee-at},\text{professor-at},\text{NONE}\}$. The goal is to define a model \\(s_{\params}(\x,y)\\) that assigns high *scores* to the label $\mathcal{y}$ that fits the arguments and supporting text $\mathcal{x}$, and lower scores otherwise. The model will be parametrized by \\(\params\\), and these parameters we will learn from some training set \\(\train\\) of $\mathcal{x,y}$ pairs. When we need to classify input  instances $\mathcal{x}$ consisting again of pairs of arguments and supporting texts, we have to solve the maximization problem $\argmax_y s_{\params}(\x,y)$. Note that this frames relation extraction as a multi-class classification problem (Exercise: how could RE be formalised to predict multiple labels for each input instance and how would the example below have to be adapted for that?)
# 
# 
# ## Relation Extraction Example
# Before we take a closer look at relation extraction methods, let us consider a concrete example. The concrete task we are considering here is to extract "method used for task" relations from sentences in computer science publications. As mentioned above, the first step would normally be to detection named entities, i.e. to determine tose pairs of arguments $\mathcal{E}$. For simplicity, our training data already contains those annotations.
# 
# 
# ## Pattern-Based Extraction
# The simplest relation extraction model defines a set of textual patterns for each relation and then assigns labels to entity pairs whose sentences match that pattern. The training data consists of entity pairs $\mathcal{E}$, patterns $A$ and labels $Y$.

# In[3]:


def readLabelledPatternData(filepath="../data/ie/ie_bootstrap_patterns.txt"):
    f = open(filepath, "r")
    patterns = []
    entpairs = []
    for l in f:
        label, pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        patterns.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return patterns, entpairs

training_patterns, training_entpairs = readLabelledPatternData()
print("Training patterns and entity pairs for relation 'method used for task'")
[(tr_a, tr_e) for (tr_a, tr_e) in zip(training_patterns[:5], training_entpairs[:5])]


# The patterns are currently sentences where the entity pairs where blanked with the placeholder 'XXXXX'.
# Note that for the training data, we also have labels. However, because we only have positive instances 
# and only for one relation ('method used for task'), we do not differentiate between them. 
# We read test data in the same way, i.e.

# In[4]:


def readPatternData(filepath="../data/ie/ie_patterns.txt"):
    f = open(filepath, "r")
    patterns = []
    entpairs = []
    for l in f:
        pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        patterns.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return patterns, entpairs

testing_patterns, testing_entpairs = readPatternData()
print("Testing patterns and entity pairs")
[(tr_a, tr_e) for (tr_a, tr_e) in zip(testing_patterns[0:5], testing_entpairs[:5])]


# For the testing data, we do not know the relations for the instances. We build a scoring model to determine which of the testing instances are examples for the relation 'method used for task' and which ones are not. (Thought exercise: how do we even know or can we ensure that even any of the instances here are examples of the relation in question?)
# 
# A pattern scoring model \\(s_{\params}(\x,y)\\) only has one parameter and assignes scores to each relation label \\(y\\) proportional to the matches with the set of textual patterns. The final label assigned to each instance is then the one with the highest score.
# Here, our pattern scoring model is even simpler since we only have patterns for one relation. Hence the final label assigned to each instance is 'method used for task' if there is a match with a pattern, and 'NONE' if there is no match.
# 
# Let's have a closer look at how pattern matching works now. Recall that the original patterns in the training data are sentences where the entity pairs are blanked with 'XXXXX'.
# 
# We could use those patterns to find new sentences. However, we are not likely to find many since the patterns are very specific. Hence, we need to generalise those patterns to less specific ones. A simple way is to define the sequence of words between each entity pair as a pattern, like so:

# In[5]:


def sentenceToShortPath(sent):
    """
    Returns the path between two arguments in a sentence, where the arguments have been masked
    Args:
        sent: the sentence
    Returns:
        the path between to arguments
    """
    sent_toks = sent.split(" ")
    indeces = [i for i, ltr in enumerate(sent_toks) if ltr == "XXXXX"]
    pattern = " ".join(sent_toks[indeces[0]+1:indeces[1]])
    return pattern

print(training_patterns[0])
sentenceToShortPath(training_patterns[0])


# There are many different alternatives to this. (Thought exercise: what are better ways of generalising patterns?)
# 
# After the sentences shortening / pattern generalisation is defined, we can then apply those patterns to testing instances to classify them into 'method used for task' and 'NONE'. In practice, the code below returns only the instances which contain a 'method used for task' pattern. 

# In[6]:


def patternExtraction(training_sentences, testing_sentences):
    """
    Given a set of patterns for a relation, searches for those patterns in other sentences
    Args:
        sent: training sentences with arguments masked, testing sentences with arguments masked
    Returns:
        the testing sentences which the training patterns appeared in
    """
    # convert training and testing sentences to short paths to obtain patterns
    training_patterns = set([sentenceToShortPath(test_sent) for test_sent in training_sentences])
    testing_patterns = [sentenceToShortPath(test_sent) for test_sent in testing_sentences]
    # look for training patterns in testing patterns
    testing_extractions = []
    for i, testing_pattern in enumerate(testing_patterns):
        if testing_pattern in training_patterns: # look for exact matches of patterns
            testing_extractions.append(testing_sentences[i])
    return testing_extractions

patternExtraction(training_patterns[:500], testing_patterns[:500])


# (Exercise: introduce patterns for other relations here and amend the scoring function in the Python code. Note that it is also possible to have 'NONE' patterns for 'no relation' between the entities.
# 
# One of the shortcomings of this pattern-based approach is that the set of patterns has to be defined manually and the model does not learn new patterns. We will next look at an approach which addresses those two shortcomings.

# 
# 
# ## Bootstrapping
# 
# Bootstrapping relation extraction models take the same input as pattern-based approaches, i.e. a set of entity pairs and patterns. The overall idea is to extract more patterns and entity pairs iteratively. For this, we need two helper methods: one method that generalises from entity pairs to extract more patterns and entity pairs, and another one that generalises from patterns to extract more patterns and entity pairs.
# 
# <!--Bootstrapping relation extraction models still take as input a set of entity pairs and patterns, same as pattern-based relation extraction approaches, but they aim at discovering new patterns.
# Algo:
# - Input: set of relation types \\(\Ys\\), set of seed entity pairs \\(\Es\\), set of seed patterns for each relation (\Ps\\), set of sentences \\(\Xs\\)
# - For each iteration
#     - Patterns P*
#     - Entity pairs E*
#     - For each sentence:
#         - if it contains a seed entity pair e:
#             - add the path between the entity pairs to P* as a new pattern
#         - if it contains a seed pattern p:
#             - identify an entity pair in the sentence and add it to E*
#     - P <- P + generalise(P*)
#     - E <- E + generalise(E*)
# We can examine the output of the model at each iteration-->
# 

# In[7]:


def searchForPatternsAndEntpairsByPatterns(training_patterns, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_pattern in enumerate(testing_patterns):
        if testing_pattern in training_patterns: # if there is an exact match of a pattern
            testing_extractions.append(testing_sentences[i])
            appearing_testing_patterns.append(testing_pattern)
            appearing_testing_entpairs.append(testing_entpairs[i])
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs


def searchForPatternsAndEntpairsByEntpairs(training_entpairs, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_entpair in enumerate(testing_entpairs):
        if testing_entpair in training_entpairs: # if there is an exact match of an entity pair
            testing_extractions.append(testing_sentences[i])
            appearing_testing_entpairs.append(testing_entpair)
            appearing_testing_patterns.append(testing_patterns[i])
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs


# Those two helper functions are then applied iteratively:

# In[8]:


def bootstrappingExtraction(train_sents, train_entpairs, test_sents, test_entpairs, num_iter):
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
    train_patterns = set([sentenceToShortPath(s) for s in train_sents])
    train_patterns.discard("in") # too general, remove this
    test_patterns = [sentenceToShortPath(s) for s in test_sents]
    test_extracts = []

    # iteratively get more patterns and entity pairs
    for i in range(1, num_iter):
        print("Number extractions at iteration", str(i), ":", str(len(test_extracts)))
        print("Number patterns at iteration", str(i), ":", str(len(train_patterns)))
        print("Number entpairs at iteration", str(i), ":", str(len(train_entpairs)))
        # get more patterns and entity pairs
        test_extracts_p, ext_test_patterns_p, ext_test_entpairs_p = searchForPatternsAndEntpairsByPatterns(train_patterns, test_patterns, test_entpairs, test_sents)
        test_extracts_e, ext_test_patterns_e, ext_test_entpairs_e = searchForPatternsAndEntpairsByEntpairs(train_entpairs, test_patterns, test_entpairs, test_sents)
        # add them to the existing entity pairs for the next iteration
        train_patterns.update(ext_test_patterns_p)
        train_patterns.update(ext_test_patterns_e)
        train_entpairs.extend(ext_test_entpairs_p)
        train_entpairs.extend(ext_test_entpairs_e)
        test_extracts.extend(test_extracts_p)
        test_extracts.extend(test_extracts_e)

    return test_extracts

test_extracts = bootstrappingExtraction(training_patterns, training_entpairs, testing_patterns, testing_entpairs, num_iter=6)


# One of the things that is noticable is that with each iteration, the number of extractions we find increases, but they are less correct.

# In[9]:


print(test_extracts[0:3])
print(test_extracts[-4:-1])


# One of the reasons is that the semantics of the pattern shifts, so here we try to find new patterns for 'method used for task', but because the instances share a similar context with other relations, the patterns and entity pairs iteratively move away from the 'method used in task' relation. Another example in a different domain are the 'student-at' and 'lecturere-at' relations, that have many overlapping contexts.
# One way of improving this is with confidence values for each entity pair and pattern. For example, we might want to avoid patterns which are too general and penalise them.

# In[10]:


from collections import Counter
te_cnt = Counter()
for te in test_extracts:
    te_cnt[sentenceToShortPath(te)] += 1
print(te_cnt)


# Above, we see that the 'in' pattern was found, which maches many contexts that are not 'method used for task'. (Exercise: implement a confidence weighting for patterns.)
# 
# 
# ## Supervised Relation Extraction
# A different way of assigning a relation label to new instances is to follow the supervised learning paradigm, which we have already seen for other structured prediction tasks. For supervised relation extraction, the scoring model \\(s_{\params}(\x,y)\\) is estimated automatically based on training sentences $\mathcal{X}$ and their labels $\mathcal{Y}$.
# For the model, we can use range of different classifiers, e.g. a logistic regression model or an SVM. At testing time, the predict label for each testing instance is the highest-scoring one, i.e. $$ \y^* = \argmax_{\y\in\Ys} s(\x,\y) $$
# 
# First, we read in the training data, consisting again of patterns, entity pairs and labels. This time, the given labels for the training instances are 'method used for task' or 'NONE', i.e. we have positive and negative training data.

# In[11]:


def readLabelledData(filepath="../data/ie/ie_training_data.txt"):
    f = open(filepath, "r")
    sents = []
    entpairs = []
    labels = []
    for l in f:
        label, sent, entpair = l.strip().replace("    ", "\t").split("\t")
        labels.append(label)
        sents.append(sent)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return sents, entpairs, labels

training_sents, training_entpairs, training_labels = readLabelledData()
print("Manually labelled data set consists of", training_labels.count("NONE"), 
          "negative training examples and", training_labels.count("method used for task"), "positive training examples")
[(tr_s, tr_e, tr_l) for (tr_s, tr_e, tr_l) in zip(training_sents[:5], training_entpairs[:5], training_labels[:5])]


# Next, we define how to transform training and testing data to features. 
# Features for the model are typically extracted from the shortest dependency path between two entities. Basic features are n-gram features, or they can be based on the syntactic structure of the input, i.e. the dependency path ([parsing](statnlpbook/chapters/parsing))
# Note that here we assume again that entity pairs are part of the input, i.e. we assume the named entity recognition problem to be solved as part of the preprocessing of the data. In reality, named entities have to be recognised first.
# 
# Here, we use sklearn's built-in feature extractor which transforms sentences to n-grams with counts of their appearances.
# 
# 

# In[12]:


from sklearn.feature_extraction.text import CountVectorizer

def featTransform(sents_train, sents_test):
    cv = CountVectorizer()
    cv.fit(sents_train)
    print(cv.get_params())
    features_train = cv.transform(sents_train)
    features_test = cv.transform(sents_test)
    return features_train, features_test, cv


# We define a model, again with sklearn, using one of their built-in classifiers and a prediction function.

# In[18]:


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


# We further define a helper function for debugging that determines the most useful features learned by the model:

# In[13]:


def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


# Supervised relation extraction algorithm:
# 
# <!--Algo:
#  Transform to Python code 
# - Input: set of training sentences \\(\Xs\\) annotated with entity pairs \\(\Es\\) and relation types \\(\Ys\\) 
# - features <- your_favourite_feature_extractor(training_sentences)
# - model <- train_model(features, labels)
# - predictions_test <- model(testing_sentences) -->
# 

# In[14]:


def supervisedExtraction(train_sents, train_entpairs, train_labels, test_sents, test_entpairs):
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
    train_patterns = [sentenceToShortPath(test_sent) for test_sent in train_sents]
    test_patterns = [sentenceToShortPath(test_sent) for test_sent in test_sents]

    # extract features
    features_train, features_test, cv = featTransform(train_patterns, test_patterns)

    # train model
    model = model_train(features_train, train_labels)

    # show most common features
    show_most_informative_features(cv, model)

    # get predictions
    predictions = predict(model, features_test)

    # show the predictions
    for tup in zip(predictions, test_sents, test_entpairs):
        print(tup)

    return predictions

supervisedExtraction(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)


# As we can see, some of the features are common words (i.e. 'stop words', such as 'is') and very broad. Other features are very specific and thus might not appear very often. Typically these problems can be mitigated by using more sophisticated features such as those based on syntax (Exercise: use dependency parsing features or 3-gram features instead of bag of unigram features.) 
# Also, the current model does not take into the entity pairs, only the path between the entity pairs. We will later examine a model that also takes entity pairs into account.
# Finally, the model requires manually annotated training data, which might not always be available. Next, we will look at a method that provides a solution for the latter.

# ## Distant Supervision
# Supervised learning typically requires large amounts of hand-labelled training examples. Since it is time-consuming and expensive to manually label examples, it is desirable to find ways of automatically or semi-automatically producing more training data. We have already seen one example of this, bootstrapping.
# Although bootstrapping can be useful, one of the downsides already discussed above is semantic drift due to the iterative nature of finding good entity pairs and patterns. 
# An alternative approach to this is to distant supervision. Here, we still have a set of entity pairs $\mathcal{E}$, their relation types $\mathcal{R}$ and a set of sentences $\mathcal{X}$ as an input, but we do not require pre-defined patterns. Instead, a large number of such entity pairs and relations are obtained from a knowledge resource, e.g. the [Wikidata knowledge base](https://www.wikidata.org) or tables.
# These entity pairs and relations are then used to automatically label all sentences with relations if there exists an entity pair between which this relation holds according to the knowledge resource. After sentences are labelled in this way, the rest of the algorithm is the same the supervised relation extraction algorithm.
# 

# In[15]:


def readDataForDistantSupervision(filepath="../data/ie/ie_training_data.txt"):
    f = open(filepath, "r")
    unlab_sents = []
    unlab_entpairs = []
    kb_entpairs = []
    for l in f:
        label, sent, entpair = l.strip().replace("    ", "\t").split("\t")
        entpair = entpair.strip("['").strip("']").split("', '")
        # Define the positively labelled entity pairs as the KB ones, which are all for the same relation. 
        # Normally these would come from an actual KB.
        if label != "NONE": 
            kb_entpairs.append(entpair)
        unlab_sents.append(sent)
        unlab_entpairs.append(entpair)
    return kb_entpairs, unlab_sents, unlab_entpairs

def distantlySupervisedLabelling(kb_entpairs, unlab_sents, unlab_entpairs):
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
        if unlab_entpair in kb_entpairs:  # if the entity pair is a KB tuple, it is a positive example for that relation
            train_entpairs.append(unlab_entpair)
            train_sents.append(unlab_sents[i])
            train_labels.append("method used for task")
        else: # else, it is a negative example for that relation
            train_entpairs.append(unlab_entpair)
            train_sents.append(unlab_sents[i])
            train_labels.append("NONE")

    return train_sents, train_entpairs, train_labels

def distantlySupervisedExtraction(kb_entpairs, unlab_sents, unlab_entpairs, test_sents, test_entpairs):
    # training_data <- Find training sentences with entity pairs
    train_sents, train_entpairs, train_labels = distantlySupervisedLabelling(kb_entpairs, unlab_sents, unlab_entpairs)
    
    print("Distantly supervised labelling results in", train_labels.count("NONE"), 
          "negative training examples and", train_labels.count("method used for task"), "positive training examples")
    
    # training works the same as for supervised RE
    supervisedExtraction(train_sents, train_entpairs, train_labels, test_sents, test_entpairs) 
    
kb_entpairs, unlab_sents, unlab_entpairs = readDataForDistantSupervision()
print(len(kb_entpairs), "'KB' entity pairs for relation 'method used for task' :", kb_entpairs[0:5])
print(len(unlab_entpairs), 'all entity pairs')
distantlySupervisedExtraction(kb_entpairs, unlab_sents, unlab_entpairs, testing_patterns, testing_entpairs)


# The results we get here are the same as for supervised relation extraction. This is because the distant supervision heuristic identified the same positive and negative training examples as in the manually labelled dataset. In practice, the distant supervision heuristic typically leads to noisy training data due to several reasons.
# 
# The first one is overlapping relations. For instance, 'employee-of' entails 'lecturer-at' and there are some overlapping entity pairs between the relations 'employee-of' and 'student-at'.
# 
# The next problem is ambiguous entities, e.g. 'EM' has many possible meanings, only one of which is 'Expectation Maximisation', see [the Wikipedia disambiguation page for the acronym](https://en.wikipedia.org/wiki/EM).
# 
# Next, not every sentence an entity pair that is a positive example for a relation appears in actually contains that relation, e.g. compare the sentence from [the Wikipedia EM definition](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) 'Expectation–maximization algorithm, an algorithm for finding maximum likelihood estimates of parameters in probabilistic models' with 'In this section we introduce EM and probabilistic models'. The first one is a true mention of 'method used for task', whereas the second one is not.
# 
# 
# ## Universal Schema
# Recall that for the pattern-based and bootstrapping approaches earlier, we were looking for simplified paths between entity pairs $\mathcal{E}$ expressing a certain relation $\mathcal{R}$ which we defined beforehand. This restricts the relation extraction problem to known relation types $\mathcal{R}$. In order to overcome that limitation, we could have defined new relations on the spot and added them to $\mathcal{R}$ by introducing new relation types for certain simplified paths between entity pairs.
# 
# The goal of universal schemas is to overcome the limitation of having to pre-define relations, but within the supervised learning paradigm. This is possible by thinking of paths between entity pairs as relation expressions themselves. Simplified paths between entity pairs and relation labels are no longer considered separately, but instead the paths between entity pairs and relations is modelled in the same space. The space of entity pairs and relations is defined by a matrix:
# 
# |  | demonstrates XXXXX for XXXXXX | XXXXX is capable of XXXXXX | an XXXXX model is employed for XXXXX | XXXXX decreases the XXXXX | method is used for task |
# | ------ | ----------- |
# | 'text mining', 'building domain ontology' | 1 |  |  |  | 1 |
# | 'ensemble classifier', 'detection of construction materials' |  |  | 1 |  | 1 |
# | 'data mining', 'characterization of wireless systems performance'|  | 1 |  |  | ? |
# | 'frequency domain', 'computational cost' |  |  |  | 1 | ? |
# 
# Here, 'method is used for task' is a relation defined by a KB schema, whereas the other relations are surface pattern relations generated by blanking entity pairs in sentences. Where an entity pair and a KB relation or surface pattern relation co-occur, this is signified by a '1'. For some of the entities and surface pairs, a label for 'method used for task' is available, whereas for others, it is not (signified by the '?').
# The task is to turn the '?'s into 0/1 predictions for the 'method for task' relation.
# Note that this is the same data and task as for relation extraction extraction with supervised learning, merely the data representation is different. 
# 
# In order to solve this prediction task, we will learn to fill in the empty cells in the matrix. This is achieved by learning to distinguish between entity pairs and relations which co-occur in our training data and entity pairs and relations which are not known to co-occur (the empty cells).
# Each training instance consists of a surface pattern or KB relation $\mathcal{r_{pos}}$ and an entity pair  $\mathcal{e_{pos}}$ the relation co-occurs with, as well as a relation $\mathcal{r_{neg}}$ and a entity pair $\mathcal{e_{neg}}$ that do not co-occur in the training data. The positive relations and entity pairs are directly taken from the annotated data. The negative entity pairs and relations are sampled randomly from data points which are represented by the empty cell in the matrix above. The goal is to estimate, for a relation $\mathcal{r}$ such as 'method is used for task' and an unseen entity pair such as $\mathcal{e}$, e.g. ('frequency domain', 'computational cost'), what the probability $\mathcal{p(y_{r,e} = 1)}$ is.
# 
# First, we read in the annotated data and sample negative entity pairs and relations.

# In[16]:


import random

# data reading
training_sents, training_entpairs, training_labels = readLabelledData()

# data converting
def vectorise_data(training_sents, training_entpairs, training_kb_rels, testing_sents, testing_entpairs):

    pos_train_ids, neg_train_ids = ie.split_labels_pos_neg(training_kb_rels + training_kb_rels)

    training_toks_pos = [t.split(" ") for i, t in enumerate(training_sents + training_kb_rels) if i in pos_train_ids]
    training_toks_neg = [t.split(" ") for i, t in enumerate(training_sents + training_kb_rels) if i in neg_train_ids]

    training_ent_toks_pos = [" || ".join(t).split(" ") for i, t in enumerate(training_entpairs + training_entpairs) if i in pos_train_ids]
    training_ent_toks_neg = [" || ".join(t).split(" ") for i, t in enumerate(training_entpairs + training_entpairs) if i in neg_train_ids]
    testing_ent_toks = [" || ".join(t).split(" ") for t in testing_entpairs]

    lens_rel = [len(s) for s in training_toks_pos + training_toks_neg]
    lens_ents = [len(s) for s in training_ent_toks_pos + training_ent_toks_neg + testing_ent_toks]
    print("Max relation length:", max(lens_rel))
    print("Max entity pair length:", max(lens_ents))

    count_rels, dictionary_rels, reverse_dictionary_rels = ie.build_dataset(
        [token for senttoks in training_toks_pos + training_toks_neg for token in senttoks])

    count_ents, dictionary_ents, reverse_dictionary_ents = ie.build_dataset(
        [token for senttoks in training_ent_toks_pos + training_ent_toks_neg for token in senttoks])

    rels_train_pos = [ie.transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in training_toks_pos]
    rels_train_neg = [ie.transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in training_toks_neg]
    ents_train_pos = [ie.transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in training_ent_toks_pos]
    ents_train_neg = [ie.transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in training_ent_toks_neg]

    # Negatively sample some entity pairs for training. Here we have some manually labelled neg ones, so we can sample from them.
    ents_train_neg_samp = [random.choice(ents_train_neg) for _ in rels_train_neg]
    
    ents_test_pos = [ie.transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in testing_ent_toks]
    # Sample those test entity pairs from the training ones as for those we have neg annotations
    ents_test_neg_samp = [random.choice(ents_train_neg) for _ in ents_test_pos]  

    vocab_size_rels = len(dictionary_rels)
    vocab_size_ents = len(dictionary_ents) 

    # for testing, we want to check if each unlabelled instance expresses the given relation "method for task"
    rels_test_pos = [ie.transform_dict(dictionary_rels, training_toks_pos[-1], max(lens_rel)) for _ in testing_sents]
    rels_test_neg_samp = [random.choice(rels_train_neg) for _ in rels_test_pos]

    return rels_train_pos, rels_train_neg, ents_train_pos, ents_train_neg_samp, rels_test_pos, rels_test_neg_samp, \
           ents_test_pos, ents_test_neg_samp, vocab_size_rels, vocab_size_ents, max(lens_rel), max(lens_ents), \
           reverse_dictionary_rels, reverse_dictionary_ents
        
data = vectorise_data(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)


# We model entity pairs and relations through latent feature representations, $\mathcal{v_e}$ and $\mathcal{a_r}$, respectively. Each entity pair and relation thus corresponds to a vector, and we can measure the compatibility between them by taking their dot product, i.e. $\mathcal{v_e * a_r}$.
# 
# 

# In[17]:


def create_model_f_reader(max_lens_rel, max_lens_ents, repr_dim, vocab_size_rels, vocab_size_ents):
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
    # Placeholders (empty Tensorflow variables) for positive and negative relations and entity pairs
    # In each training epoch, for each batch, those will be set through mini batching

    relations_pos = tf.placeholder(tf.int32, [None, max_lens_rel],
                                   name='relations_pos')  # [batch_size, max_rel_seq_len]
    relations_neg = tf.placeholder(tf.int32, [None, max_lens_rel],
                                   name='relations_neg')  # [batch_size, max_rel_seq_len]

    ents_pos = tf.placeholder(tf.int32, [None, max_lens_ents], name="ents_pos")  # [batch_size, max_ent_seq_len]
    ents_neg = tf.placeholder(tf.int32, [None, max_lens_ents], name="ents_neg")  # [batch_size, max_ent_seq_len]

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

    # our feature representation here is a vector for each word in a relation or entity
    # because our training data is so small
    # we therefore take the sum of those vectors to get a representation of each relation or entity pair
    rel_encodings_pos = tf.reduce_sum(rel_encodings_pos, 1)  # [batch_size, num_rel_toks, repr_dim]
    rel_encodings_neg = tf.reduce_sum(rel_encodings_neg, 1)  # [batch_size, num_rel_toks, repr_dim]

    ent_encodings_pos = tf.reduce_sum(ent_encodings_pos, 1)  # [batch_size, num_ent_toks, repr_dim]
    ent_encodings_neg = tf.reduce_sum(ent_encodings_neg, 1)  # [batch_size, num_ent_toks, repr_dim]

    # measuring compatibility between positive entity pairs and relations
    # used for ranking test data
    dotprod_pos = tf.reduce_sum(tf.multiply(ent_encodings_pos, rel_encodings_pos), 1)

    # measuring compatibility between negative entity pairs and relations
    dotprod_neg = tf.reduce_sum(tf.multiply(ent_encodings_neg, rel_encodings_neg), 1)

    # difference in dot product of positive and negative instances
    # used for BPR loss (ranking loss)
    diff_dotprod = tf.reduce_sum(
        tf.multiply(ent_encodings_pos, rel_encodings_pos) - tf.multiply(ent_encodings_neg, rel_encodings_neg), 1)

    return dotprod_pos, dotprod_neg, diff_dotprod, [relations_pos, relations_neg, ents_pos, ents_neg]


# To train this model, we define a loss, which tries to maximise the distance between the positive and negative instances. One possibility of this is the logistic loss.
# 
# $\mathcal{\sum -  log(v_{e_{pos}} * a_{r_{pos}})} + {\sum log(v_{e_{neg}} * a_{r_{neg}}))}$

# In[19]:


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
np.random.seed(1337)
tf.set_random_seed(1337)

def universalSchemaExtraction(data):
    rels_train_pos, rels_train_neg, ents_train_pos, ents_train_neg_samp, rels_test_pos, rels_test_neg_samp, \
    ents_test_pos, ents_test_neg_samp, vocab_size_rels, vocab_size_ents, max_lens_rel, max_lens_ents, \
    dictionary_rels_rev, dictionary_ents_rev = data

    batch_size = 4
    repr_dim = 30
    learning_rate = 0.001
    max_epochs = 31

    dotprod_pos, dotprod_neg, diff_dotprod, placeholders = create_model_f_reader(max_lens_rel, max_lens_ents, repr_dim, vocab_size_rels,
                          vocab_size_ents)

    # logistic loss
    loss = tf.reduce_sum(tf.nn.softplus(-dotprod_pos)+tf.nn.softplus(dotprod_neg))

    # alternative: BPR loss
    #loss = tf.reduce_sum(tf.nn.softplus(diff_dotprod))

    data = [np.asarray(rels_train_pos), np.asarray(rels_train_neg), np.asarray(ents_train_pos), np.asarray(ents_train_neg_samp)]
    data_test = [np.asarray(rels_test_pos), np.asarray(rels_test_neg_samp), np.asarray(ents_test_pos), np.asarray(ents_test_neg_samp)]

    # we use the Adam optimiser
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # training with mini-batches
    batcher = tfutil.BatchBucketSampler(data, batch_size)
    batcher_test = tfutil.BatchBucketSampler(data_test, 1, test=True)

    with tf.Session() as sess:
        trainer = tfutil.Trainer(optimizer, max_epochs)
        trainer(batcher=batcher, placeholders=placeholders, loss=loss, session=sess)

        # we obtain test scores
        test_scores = trainer.test(batcher=batcher_test, placeholders=placeholders, model=tf.nn.sigmoid(dotprod_pos), session=sess)

    # show predictions
    ents_test = [ie.reverse_dict_lookup(dictionary_ents_rev, e) for e in ents_test_pos]
    rels_test = [ie.reverse_dict_lookup(dictionary_rels_rev, r) for r in rels_test_pos]
    testresults = sorted(zip(test_scores, ents_test, rels_test), key=lambda t: t[0], reverse=True)  # sort for decreasing score

    print("\nTest predictions by decreasing probability:")
    for score, tup, rel in testresults:
        print('%f\t%s\tREL\t%s' % (score, " ".join(tup), " ".join(rel)))
        
        
universalSchemaExtraction(data)


# Test predictions probabilities are obtained by scoring each test instances with:
# 
# $\mathcal{ \sigma  ( v_{e} * a_{r} )}$
# 
# Note that as input for the latent feature representation, we discarded words that only appeared twice. Hence, for those words we did not learn a representation, denoted here by 'UNK'. This is also typically done for other feature representations, as if we only see a feature once, it is difficult to learn weights for it.
# 
# Exercises: 
# The scores shown here are for the relation 'method used for task'. However, we could also use our model to score the compatibility of entity pairs with other relations, e.g. 'demonstrates XXXXX for XXXXXX'. How could this be done here?
# How could we get around the problem of unseen words, as described above?

# 
# 
# ## Background
# Jurafky, Dan and Martin, James H. (2016). Speech and Language Processing, Chapter 21 (Information Extraction): https://web.stanford.edu/~jurafsky/slp3/21.pdf
# 
# Riedel, Sebastian and Yao, Limin and McCallum, Andrew and Marlin, Benjamin M. (2013). Extraction with Matrix Factorization and Universal Schemas. Proceedings of NAACL.  http://www.aclweb.org/anthology/N13-1008
