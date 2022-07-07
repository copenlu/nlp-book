#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Welcome to this interactive book on Statistical Natural Language Processing (NLP). NLP is a field that lies in the intersection of Computer Science, Artificial Intelligence (AI) and Linguistics with the goal to enable computers to solve tasks that require natural language _understanding_ and/or _generation_. Such tasks are omnipresent in most of our day-to-day life: think of [Machine Translation](https://www.bing.com/translator/), Automatic [Question Answering](https://www.youtube.com/watch?v=WFR3lOm_xhE) or even basic [Search](https://www.google.co.uk). All these tasks require the computer to process language in one way or another. But even if you ignore these practical applications, many people consider language to be at the heart of human intelligence, and this makes NLP (and its more linguistically motivated cousin, [Computational Linguistics](http://en.wikipedia.org/wiki/Computational_linguistics)), important for its role in AI alone.
# 
# ### Statistical NLP
# NLP is a vast field with beginnings dating back to at least the 1960s, and it is difficult to give a full account of every aspect of NLP. Hence, this book focuses on a sub-field of NLP termed Statistical NLP (SNLP). In SNLP computers aren't directly programmed to process language; instead, they _learn_ how language should be processed based on the _statistics_ of a corpus of natural language. For example, a statistical machine translation system's behaviour is affected by the statistics of a _parallel_ corpus where each document in one language is paired with its translation in another. This approach has been dominating NLP research for almost two decades now, and has seen widespread in industry too. Notice that while Statistics and Machine Learning are, in general, quite different fields, for the purposes of this book we will mostly identify Statistical NLP with Machine Learning-based NLP.
# 
# ### NDAK18000U Course Information
# We will use materials from this interactive book throughout the course. Note that this book was originally developed for a [15 ECTS course at UCL](https://github.com/uclmr/stat-nlp-book), so we will not be covering all topics of the book and will cover some in less depth. For completeness and context, you can still access all book materials below. 
# Materials are due to minor changes. Materials covered in each week are listed below and will be linked once they are close to finalised. The course schedule is tentative and subject to minor changes.
# The official course description can be found [here](https://kurser.ku.dk/course/ndak18000u/2021-2022).
# 
#   * Week 36 (6-10 Sept)
#       * Lecture (Tuesday): Course Logistics ([slides](chapters/course_logistics.ipynb)), Introduction to NLP ([slides](chapters/intro_short.ipynb)), Tokenisation & Sentence Splitting ([notes](chapters/tokenization.ipynb), [slides](chapters/tokenization_slides.ipynb), [exercises](exercises/tokenization.ipynb)), Text Classification ([slides](chapters/doc_classify_slides_short.ipynb))
#       * Lab (10.09 & 13.09): Jupyter notebook setup, introduction to [Colab](https://colab.research.google.com/). Introduction to [PyTorch](https://pytorch.org/tutorials/). Project group arrangements. Questions about the course project. ([lab](labs/notebooks_2020_2021/lab_1.ipynb))
#   * Week 37 (13-17 Sept)
#       * Reading (before lecture): [Jurafsky & Martin Chapter 7 up to and including 7.4](https://web.stanford.edu/~jurafsky/slp3/7.pdf)
#       * Lecture (Tuesday): Introduction to Representation Learning ([slides](chapters/dl-representations_simple.ipynb)), Language Modelling (partially) ([slides](chapters/language_models_slides.ipynb))
#       * Lab (17.09 & 20.09): Recurrent Neural Networks and word representations. Project help. ([lab](labs/notebooks_2020_2021/lab_2.ipynb))
#   * Week 38 (20-24 Sept)
#       * Reading (before lecture): [Jurafsky & Martin Chapter 9, up to and including 9.6](https://web.stanford.edu/~jurafsky/slp3/9.pdf)
#       * Lecture (Tuesday): Language Modelling (rest) ([slides](chapters/language_models_slides.ipynb)), Recurrent Neural Networks ([slides](chapters/rnn_slides_ucph.ipynb)), Contextualised Word Representations ([slides](chapters/dl-representations_contextual.ipynb))
#       * Lab (24.09 & 27.09): Language Models with [Transformers](https://huggingface.co/course/chapter1) and RNNs. Project help. ([lab](labs/notebooks_2020_2021/lab_3.ipynb))
#   * Week 39 (27 Sept-1 Oct)
#       * Reading (before lecture): [*Attention? Attention!* Blog post by Lilian Weng](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html); [Belinkov and Glass, 2020. *Analysis Methods in Neural Language Processing: A Survey*](https://aclanthology.org/Q19-1004.pdf)
#       * Lecture (Tuesday): Attention ([slides](chapters/attention_slides2.ipynb)), Interpretability ([slides](chapters/interpretability_slides.ipynb))
#       * Lab (01.10 & 04.10): Error analysis and explainability. Project help. ([lab](labs/notebooks_2020_2021/lab_4.ipynb))
#   * Week 40 (4-8 Oct)
#       * Reading (before lecture): [Jurafsky & Martin Chapter 8, up to and including 8.6](https://web.stanford.edu/~jurafsky/slp3/8.pdf); [9.2.2 in Chapter 9](https://web.stanford.edu/~jurafsky/slp3/9.pdf); [18.1 in Chapter 18](https://web.stanford.edu/~jurafsky/slp3/18.pdf)
#       * Lecture (Tuesday): Sequence Labelling ([slides](chapters/sequence_labeling_slides.ipynb), [notes](chapters/sequence_labeling.ipynb))
#       * Lab (08.10 & 11.10): Sequence labelling. Beam search. Project help. ([lab](labs/notebooks_2020_2021/lab_5.ipynb))
#   * Week 41 (11-15 Oct)
#       * Reading (before lecture): [Cardie, 1997. *Empirical Methods in Information Extraction*, up to (not including) the section "Learning Extraction Patterns"](https://ojs.aaai.org//index.php/aimagazine/article/view/1322); [*Question Answering*. Blog post by Vered Shwartz](http://veredshwartz.blogspot.com/2016/11/question-answering.html)
#       * Lecture (Tuesday): Information Extraction ([slides](chapters/information_extraction_slides.ipynb)), Question Answering ([slides](chapters/question_answering_slides.ipynb))
#       * Lab (15.10 & 25.10): In-depth look at Transformers and Multilingual QA. Project help. ([lab](labs/notebooks_2020_2021/lab_6.ipynb))
#   * Week 43 (25-29 Oct)
#       * Reading (before lecture): [Jurafsky & Martin Chapter 10, up to and including 10.8.2](https://web.stanford.edu/~jurafsky/slp3/10.pdf), [recorded lecture 8 part 2; recorded lecture 5 part 9](https://absalon.instructure.com/courses/52205/external_tools/14563)
#       * Lecture (Tuesday): Machine Translation ([slides](chapters/nmt_slides_active.ipynb)), Cross-lingual Transfer Learning ([slides](chapters/xling_transfer_learning_slides.ipynb))
#       * Lab (29.10 & 01.11): Project help.
#   * Week 44 (1-5 Nov)
#       * Reading (before lecture): [Jurafsky & Martin Chapter 14, *except* 14.5](https://web.stanford.edu/~jurafsky/slp3/14.pdf); [de Marneffe et al., 2021. *Universal Dependencies*, up to and including 2.3.2](https://direct.mit.edu/coli/article/47/2/255/98516/Universal-Dependencies)
#       * Lecture (Tuesday): Dependency Parsing ([slides](chapters/dependency_parsing_slides_active.ipynb))
#       * Lab (05.11): Project help.    
# 
# 
# ### Structure of this Book
# We think that to understand and apply SNLP in practice one needs knowledge of the following:
# 
#   * Tasks (e.g. Machine Translation, Syntactic Parsing)
#   * Methods & Frameworks (e.g. Discriminative Training, Linear Chain models, Representation Learning)
#   * Implementations (e.g. NLP data structures, efficient dynamic programming)
#    
# The book is somewhat structured around the task dimension; that is, we will explore different methods, frameworks and their implementations, usually in the context of specific NLP applications.  
# 
# On a higher level the book is divided into *themes* that roughly correspond to learning paradigms within SNLP, and which follow a somewhat chronological order: we will start with generative learning, then discuss discriminative learning, then cover forms of weaker supervision to conclude with representation and deep learning. As an overarching theme we will use *structured prediction*, a formulation of machine learning that accounts for the fact that machine learning outputs are often not just classes, but structured objects such as sequences, trees or general graphs. This is a fitting approach, seeing as NLP tasks often require prediction of such structures.
# 
# ### Table Of Contents
# * Course Logistics: [slides](chapters/course_logistics.ipynb)
# * Introduction to NLP: [slides1](chapters/introduction.ipynb), [slides2](chapters/intro_short.ipynb)
# * Structured Prediction: [notes](chapters/structured_prediction.ipynb), [slides](chapters/structured_prediction_slides.ipynb), [exercises](exercises/structured_prediction.ipynb)
# * Tokenisation and Sentence Splitting: [notes](chapters/tokenization.ipynb), [slides](chapters/tokenization_slides.ipynb), [exercises](exercises/tokenization.ipynb)
# * Generative Learning:
#     * Language Models (MLE, smoothing): [notes](chapters/language_models.ipynb), [slides](chapters/language_models_slides.ipynb), [exercises](exercises/language_models.ipynb)
#         * Maximum Likelihood Estimation: [notes](chapters/mle.ipynb), [slides](chapters/mle_slides.ipynb)
#     * Machine Translation (EM algorithm, beam-search, encoder-decoder models): [notes](chapters/word_mt.ipynb), [slides1](chapters/word_mt_slides.ipynb), [slides2](chapters/neural_mt_slides.ipynb) [exercises](exercises/mt.ipynb)
#     * Constituent Parsing (PCFG, dynamic programming): [notes](chapters/parsing.ipynb), [slides](chapters/parsing_slides.ipynb), [exercises](exercises/parsing.ipynb)
#     * Dependency Parsing (transition based parsing): [notes](chapters/transition-based_dependency_parsing.ipynb), [slides](chapters/transition_slides.ipynb)
# * Discriminative Learning:
#     * Text Classification (logistic regression): [notes](chapters/doc_classify.ipynb), [slides1](chapters/doc_classify_slides.ipynb), [slides2](chapters/doc_classify_slides_short.ipynb)
#     * Sequence Labelling (linear chain models): [notes](chapters/sequence_labeling.ipynb), [slides](chapters/sequence_labeling_slides.ipynb)
#     * Sequence Labelling (CRF): [slides](chapters/sequence_labeling_crf_slides.ipynb)
# * Weak Supervision:
#     * Relation Extraction (distant supervision, semi-supervised learning) [notes](chapters/relation_extraction.ipynb), [slides](https://www.dropbox.com/s/xqq1nwgw1i0gowr/relation-extraction.pdf?dl=0), [interactive-slides](chapters/relation_extraction_slides.ipynb)
# * Representation and Deep Learning
#     * Overview and Multi-layer Perceptrons [slides](chapters/dl.ipynb)
#     * Word Representations [slides](chapters/dl-representations_simple.ipynb)
#     * Contextualised Word Representations [slides](chapters/dl-representations_contextual.ipynb)
#     * Recurrent Neural Networks [slides1](chapters/rnn_slides.ipynb), [slides2](chapters/rnn_slides_ucph.ipynb)
#     * Attention ([slides](chapters/attention_slides.ipynb)
#     * Transfer Learning [slides](chapters/transfer_learning_slides.ipynb)
#     * Textual Entailment (RNNs) [slides](chapters/dl_applications.ipynb)
#     * Interpretability ([slides](chapters/interpretability_slides.ipynb)
# 
# #### Methods
# We have a few dedicated method chapters:
# 
# * Structured Prediction: [notes](chapters/structured_prediction.ipynb)
# * Maximum Likelihood Estimation: [notes](chapters/mle.ipynb)
# * EM-Algorithm: [notes](chapters/em.ipynb)
# 
# ### Interaction
# 
# The best way to learn language processing with computers is 
# to process language with computers. For this reason this book features interactive 
# code blocks that we use to show NLP in practice, and that you can use 
# to test and investigate methods and language. We use the [Python language](https://www.python.org/) throughout this book because it offers a large number of relevant libraries and it is easy to learn.
# 
# ### Installation
# To install the book locally and use it interactively follow the installation instruction on [GitHub](https://github.com/uclmr/stat-nlp-book).
# 
# 
# Setup tutorials:
# * [Azure tutorial](tutorials/azure_tutorial.ipynb)
# 

# In[ ]:




