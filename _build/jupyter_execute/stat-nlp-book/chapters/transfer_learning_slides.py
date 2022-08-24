#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n<style>\n.rendered_html td {\n    font-size: xx-large;\n    text-align: left; !important\n}\n.rendered_html th {\n    font-size: xx-large;\n    text-align: left; !important\n}\n</style>\n')


# In[4]:


get_ipython().run_line_magic('load_ext', 'tikzmagic')


# # Recent Topics in NLP

# # Outline
# 
# - Natural language inference (15 min.)
# - Transfer learning (15 min.)
# - Multi-task benchmarks (15 min.)
# - Break (10 min.)
# - Interpretability ([slides](interpretability_slides.ipynb))
# 

# # Recognizing Textual Entailment = Natural Language Inference
# 
# Determining the logical relationship between two sentences.
# 
# - Classification task
# - Requires commonsense and world knowledge
# - Requires general natural language understanding
# - Requires fine-grained reasoning

# ### Recognizing Textual Entailment (RTE)
# 
# [Dagan et al., 2005](http://u.cs.biu.ac.il/~nlp/downloads/publications/RTEChallenge.pdf)
# 
# - Text (premise) T
# - Hypothesis H
# 
# T entails H if, typically, a human reading T would infer that H is most likely true.

# > **T:** “Google files for its long awaited IPO.”
# > **H:** “Google goes public.”

# Positive ($\Rightarrow$, entails)

# > **T:** “Regan attended a ceremony in Washington to commemorate the landings in Normandy.”
# > **H:** “Washington is located in Normandy.”

# Negative ($\not\Rightarrow$, does not entail)

# ### Stanford Natural Language Inference (SNLI) corpus
# 
# [Bowman et al., 2015](https://www.aclweb.org/anthology/D15-1075.pdf): crowdsourced NLI using image captions.
# 
# 570K sentence pairs, two orders of magnitude larger than other NLI resources (1K-10K examples).
# 
# **T**: A wedding party taking pictures
# - **H:** There is a funeral					: **<span class=red>Contradiction</span>** ($\Rightarrow\neg$)
# - **H:** They are outside					    : **<span class=blue>Neutral</span>** (?)
# - **H:** Someone got married				    : **<span class=green>Entailment</span>** ($\Rightarrow$)
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/Wedding_photographer_at_work.jpg" width=1500/> 

# ### Multi-NLI (MNLI)
# 
# [Williams et al., 2018](https://www.aclweb.org/anthology/N18-1101.pdf): more diverse domains.
# 
# > **T:** “The legislation was widely hailed as a model for the country.”
# > **H:** “Many people thought the legislation was a model for the country.”

# Entailment ($\Rightarrow$)

# > **T:** “The program has helped victims in 90 court cases, and 150 legal counseling sessions have been held there.”
# > **H:** “Victims from 90 grand jury court cases were helped by the program.”

# Neutral (?)

# > **T:** “As a result, Chris Schneider, executive director of Central California Legal Services, is building a lawsuit against Alpaugh Irrigation.”
# > **H:** “Central California Legal Services’ executive director decided not to pursue a lawsuit against Alpaugh Irrigation.”

# Contradiction ($\Rightarrow\neg$)

# ## RTE/NLI state of the art until 2015
# 
# [Lai and Hockenmaier, 2014](https://www.aclweb.org/anthology/S14-2055.pdf),
# [Jimenez et al., 2014](https://www.aclweb.org/anthology/S14-2131.pdf),
# [Zhao et al., 2014](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6889713),
# [Beltagy et al., 2016](https://www.aclweb.org/anthology/J16-4007.pdf) and others:
# engineered pipelines.
# 
# - Various external resources
# - Specialized subcomponents
# - Extensive use of **features**:
#   - Negation detection, word overlap, part-of-speech tags, dependency parses, alignment, symbolic meaning representation
#   
# <img src="https://d3i71xaburhd42.cloudfront.net/fca1e631b8f93036065311eb92727c509423475a/9-Figure1-1.png" width=150%/>

# > **T:** “The program has helped victims in 90 court cases, and 150 legal counseling sessions have been held there.”
# > **H:** “Victims from 90 grand jury court cases were helped by the program.”

# Neutral (?)

# > **T:** “As a result, Chris Schneider, executive director of Central California Legal Services, is building a lawsuit against Alpaugh Irrigation.”
# > **H:** “Central California Legal Services’ executive director decided not to pursue a lawsuit against Alpaugh Irrigation.”

# Contradiction ($\Rightarrow\neg$)

# ## RTE/NLI state of the art until 2015
# 
# [Lai and Hockenmaier, 2014](https://www.aclweb.org/anthology/S14-2055.pdf),
# [Jimenez et al., 2014](https://www.aclweb.org/anthology/S14-2131.pdf),
# [Zhao et al., 2014](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6889713),
# [Beltagy et al., 2016](https://www.aclweb.org/anthology/J16-4007.pdf) and others:
# engineered pipelines.
# 
# - Various external resources
# - Specialized subcomponents
# - Extensive use of **features**:
#   - Negation detection, word overlap, part-of-speech tags, dependency parses, alignment, symbolic meaning representation
#   
# <img src="https://d3i71xaburhd42.cloudfront.net/fca1e631b8f93036065311eb92727c509423475a/9-Figure1-1.png" width=150%/>

# ### Neural networks for NLI
# 
# Large-scale NLI corpora: NNs are feasible to train.

# ### Independent sentence encoding
# 
# [Bowman et al, 2015](https://www.aclweb.org/anthology/D15-1075.pdf): same LSTM encodes premise and hypothesis.
# 
# <img src="dl-applications-figures/rte.svg" width=1500/> 

# Last output vector as sentence representation.
# 
# <img src="dl-applications-figures/rte_encoding.svg" width=1500/>

# MLP to classify as entailment/neutral/contradiction.
# 
# <img src="dl-applications-figures/mlp.svg" width=1400/> 

# ## Results
# 
# <table style="font-size: 28px; border-style: solid;">
# <thead>
# <tr>
# <th>Model</th>
# <th>SNLI Test Score</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>LSTM</td>
# <td>77.6</td>
# </tr>
# </tbody>
# </table>

# ### Conditional encoding
# 
# The way we read the hypothesis could be influenced by our understanding of the premise.
# 
# <img src="dl-applications-figures/conditional.svg" width=1500/> 

# <img src="dl-applications-figures/conditional_encoding.svg" width=1500/> 

# ## Results
# 
# <table style="font-size: 28px; border-style: solid;">
# <thead>
# <tr>
# <th>Model</th>
# <th>SNLI Test Score</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>LSTM</td>
# <td>77.6</td>
# </tr>
# <tr>
# <td>Conditional endcoding</td>
# <td>81.4</td>
# </tr>
# </tbody>
# </table>

# > You can’t cram the meaning of a whole
# %&!\$# sentence into a single \$&!#* vector!
# >
# > -- <cite>Raymond J. Mooney</cite>
# 

# ### Attention
# 
# <img src="dl-applications-figures/attention.svg" width=1500/> 

# <img src="dl-applications-figures/attention_encoding.svg" width=1500/> 

# <img  src="./dl-applications-figures/camel.png"/>

# #### Contextual understanding
# 
# <img src="./dl-applications-figures/pink.png"/>

# ## Results
# 
# <table style="font-size: 28px; border-style: solid;">
# <thead>
# <tr>
# <th>Model</th>
# <th>SNLI Test Score</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>LSTM</td>
# <td>77.6</td>
# </tr>
# <tr>
# <td>Conditional encoding</td>
# <td>81.4</td>
# </tr>
# <tr>
# <td>Attention</td>
# <td>82.3</td>
# </tr>
# </tbody>
# </table>

# #### Fuzzy attention
# 
# <img  src="./dl-applications-figures/mimes.png"/>

# ### Word-by-word attention
# 
# <img src="dl-applications-figures/word_attention.svg" width=1500/> 

# <img src="dl-applications-figures/word_attention_encoding.svg" width=1500/> 

# #### Reordering
# 
# <img src="./dl-applications-figures/reordering.png" width=60%/>

# #### Synonyms
# 
# <img  src="./dl-applications-figures/trashcan.png" width=90%/>

# #### Hypernyms
# 
# <img src="./dl-applications-figures/kids.png" width=80%/>

# #### Lexical inference
# 
# <img src="./dl-applications-figures/snow.png"/>

# ## Results
# 
# <table style="font-size: 28px; border-style: solid;">
# <thead>
# <tr>
# <th>Model</th>
# <th>SNLI Test Score</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>LSTM</td>
# <td>77.6</td>
# </tr>
# <tr>
# <td>Conditional encoding</td>
# <td>81.4</td>
# </tr>
# <tr>
# <td>Attention</td>
# <td>82.3</td>
# </tr>
# <tr>
# <td>Word-by-word attention</td>
# <td><strong>83.5</strong></td>
# </tr>
# </tbody>
# </table>
# 

# ## Composition
# 
# [Bowman et al., 2016](https://www.aclweb.org/anthology/P16-1139.pdf):
# compositional vector representation based on syntactic structure.
# 
# <img src="https://d3i71xaburhd42.cloudfront.net/36c097a225a95735271960e2b63a2cb9e98bff83/1-Figure1-1.png" width=80%/>

# ## NLI artefacts
# 
# SNLI and MNLI are **crowdsourced**.
# 
# [Gururangan et al., 2018](https://www.aclweb.org/anthology/N18-2017.pdf): hypothesis phrasing alone gives out the class.
# 
# <img src="https://d3i71xaburhd42.cloudfront.net/2997b26ffb8c291ce478bd8a6e47979d5a55c466/2-Table1-1.png" width=150%/>

# ## Lexical entailment
# 
# [Glockner et al., 2018](https://www.aclweb.org/anthology/P18-2103.pdf): very **simple** examples that are hard for models.
# 
# <img src="https://persagen.com/files/misc/arxiv1805.02266-table1.png" width=80%>
# 

# # Transfer learning
# 
# <img src="https://ruder.io/content/images/2019/08/transfer_learning_taxonomy.png" width=45%>

# ## Multi-task learning
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/161ffb54a3fdf0715b198bb57bd22f910242eb49/19-Figure1.2-1.png" width=45%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Caruana, 1997](https://www.cs.cornell.edu/~caruana/mlj97.pdf))
# </div>

# ## Multi-task learning
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/161ffb54a3fdf0715b198bb57bd22f910242eb49/44-Figure2.3-1.png" width=60%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Caruana, 1997](https://www.cs.cornell.edu/~caruana/mlj97.pdf))
# </div>

# ## NLP from scratch
# 
# Architecture for POS tagging, chunking, NER, SRL as sequence labeling.
# 
# Key insight: it's better to share **embeddings**.
# 
# <table>
# <tr>
# <td><img src="https://blog.acolyer.org/wp-content/uploads/2016/07/nlu-from-scratch-fig-2.png" width=100%/></td>
# <td><img src="https://files.speakerdeck.com/presentations/56645956135848559b3003875a350fde/slide_48.jpg" width=100%/></td>
# </tr>
# </table>
# 
# <div style="text-align: right;">
#     (from [Collobert et al., 2011](https://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf))
# </div>

# # Pre-trained embeddings
# 
# General-purpose representations trained on large datasets (usually unsupervised):
# 
# - [word2vec](https://arxiv.org/pdf/1301.3781.pdf)
# - [GloVe](https://www.aclweb.org/anthology/D14-1162.pdf)
# - [ELMo](https://www.aclweb.org/anthology/N18-1202.pdf)
# - [BERT](https://www.aclweb.org/anthology/N19-1423.pdf)
# 
# are all forms of transfer learning.
# 
# <center>
#     <img src="mt_figures/bert_gpt_elmo.png" width=100%>
# </center>
# 

# ## MTL: NLI and transition-based parsing
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/36c097a225a95735271960e2b63a2cb9e98bff83/2-Figure2-1.png" width=150%/>
# </center>
# 
# <div style="text-align: right;">
#     (from [Bowman et al., 2016](https://www.aclweb.org/anthology/P16-1139.pdf))
# </div>
# 

# ## MTL with overlapping labels
# 
# Sentiment analysis, stance detection, fake news detection and NLI
# with a Label Embedding Layer.
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/64c5f7055b2e6982b6b95e069b22230d13a134bb/4-Figure1-1.png" width=150%/>
# </center>
# 
# <div style="text-align: right;">
#     (from [Augenstein et al., 2018](https://www.aclweb.org/anthology/N18-1172.pdf))
# </div>
# 

# ## Many-task MTL
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/ade0c116120b54b57a91da51235108b75c28375a/1-Figure1-1.png" width=50%/>
# </center>
# 
# <div style="text-align: right;">
#     (from [Hashimoto et al, 2017](https://www.aclweb.org/anthology/D17-1206.pdf))
# </div>

# ## When is MTL beneficial?
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/1b02204b210f822dabf8d68b7e3ea7ac14ee1268/4-Figure1-1.png" width=50%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Bingel and Søgaard, 2017](https://www.aclweb.org/anthology/E17-2026.pdf))
# </div>

# ## MTL parsing
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/9c852275f3dc176f1fbed0741201f352ec49adc1/5-Figure3-1.png" width=90%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Hershcovich and Arviv, 2019](https://www.aclweb.org/anthology/K19-2002.pdf);
#     also 
#     [Hershcovich et al., 2018](https://www.aclweb.org/anthology/P18-1035.pdf),
#     [Arviv et al., 2020](https://arxiv.org/pdf/2010.05710.pdf))
# </div>

# ## GLUE benchmark
# 
# [Wang et al., 2019](https://openreview.net/pdf?id=rJ4km2R5t7): collection of sentence- and sentence-pair-classification tasks.
# 
# <img src="https://d3i71xaburhd42.cloudfront.net/6aa371f872f49eb69a6ad185c74194d95c01257f/6-Table2-1.png"/>

# ### GLUE: Winograd NLI (WNLI)
# 
# World knowledge and logical reasoning, presented as NLI.
# 
# Positive:
# 
# > “I put the cake away in the refrigerator. It has a lot of butter in it.”
# 
# <center>
# $\Rightarrow$
# </center>
# 
# > “The cake has a lot of butter in it.”

# Negative:
# 
# > “The large ball crashed right through the table because it was made of styrofoam.”
# 
# <center>
# $\not\Rightarrow$
# </center>
# 
# > “The large ball was made of styrofoam.”

# ## GLUE benchmark
# 
# https://gluebenchmark.com/leaderboard
# 
# <center>
#     <img src="https://creatext.ai/static/img/blog/glue_performance_1.png" width=150%/>
# </center>
# 
# <div style="text-align: right;">
#     (from [Wang et al., 2019](https://papers.nips.cc/paper/8589-superglue-a-stickier-benchmark-for-general-purpose-language-understanding-systems.pdf))
# </div>
# 

# ### SuperGLUE
# 
# Harder NLI, for a meaningful comparison.
# 
# https://super.gluebenchmark.com/leaderboard
# 
# > “Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.”
# 
# <center>
# $\not\Rightarrow$
# </center>
# 
# > “Christopher Reeve had an accident.”
# 
# 
# <div style="text-align: right;">
#     (from [Wang et al., 2019](https://papers.nips.cc/paper/8589-superglue-a-stickier-benchmark-for-general-purpose-language-understanding-systems.pdf))
# </div>

# ### XTREME
# 
# Collection of multilingual multi-task benchmarks.
# 
# * Including TyDiQA-GoldP!
# 
# <center>
#     <img src="https://1.bp.blogspot.com/-5J6e2txWChk/XpSc_BaYFnI/AAAAAAAAFss/QCLROHrEutAN3GvOyfRzK8J7DA9yLY5GACLcBGAsYHQ/s640/XTREME%2BStill%2Bart_04%2Broboto.png" width="70%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Hu et al., 2020](https://arxiv.org/pdf/2003.11080.pdf))
# </div>

# ### XTREME
# 
# <center>
#     <img src="https://1.bp.blogspot.com/-yzWzRs2bK7Y/Xn1Amrk3aaI/AAAAAAAAFkM/ClIN7fAeuLgulexicZhotwPXqTLNqGUPQCLcBGAsYHQ/s640/image1.gif" width="70%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Hu et al., 2020](https://arxiv.org/pdf/2003.11080.pdf))
# </div>

# ### XTREME
# 
# Strategies:
# * Multilingual encoder (mBERT, XLM, **XLM-R**, MMTE)
# * Translate-train
# * Translate-test
# * In-language
# * Multi-task

# ### XTREME
# 
# https://sites.research.google/xtreme
# 
# <center>
#     <img src="dl-applications-figures/xtreme.png" width="70%">
# </center>
# 
# <div style="text-align: right;">
#     (from [Hu et al., 2020](https://arxiv.org/pdf/2003.11080.pdf))
# </div>

# # Further reading
# 
# - [Ruder, 2017. An Overview of Multi-Task Learning in Deep Neural Networks](https://ruder.io/multi-task/)
# - [Ruder, 2019. The State of Transfer Learning in NLP](https://ruder.io/state-of-transfer-learning-in-nlp/)
# - [Liu et al., 2019. Linguistic Knowledge and Transferability of Contextual Representations](https://www.aclweb.org/anthology/N19-1112.pdf)
