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


# # Information Extraction

# ## Schedule
# 
# * Motivation: information extraction (10 min.)
# * Reminder: named entity recognition (5 min.)
# * Background: relation extraction (5 min.)
# * Methods: relation extraction (20 min.)
# * Relation extraction via reading comprehension (5 min.)
# * Break (10 min.)
# * Question answering ([slides](question_answering_slides.ipynb))

# ![trading](https://images.squarespace-cdn.com/content/v1/5a459ea9692ebe7dcf9ca9ff/1515028331771-FGZ2K9P0VRAB1Q80OYSR/image-asset.jpeg?format=750w)

# ![news](../img/news.png)

# ### Information Extraction: Motivation
# 
# * The amount of available information is vast and still growing quickly
# * Text contains a lot of information
# * Only some of information is relevant for each use case
# * How can we automatically make sense of information?

# ### Subtasks of Information Extraction
# 

# **Relation** Extraction:
# 
# 1. **Named Entity Recognition (NER)**: text $\rightarrow$ (entity spans, entity types)
# > <font color="green">Kraft</font>, owner of <font color="green">Milka</font>, purchased <font color="green">Cadbury Dairy Milk</font> and is now gearing up for a roll-out of its new brand.
# 
# 2. **Coreference Resolution**: text $\rightarrow$ clusters of spans referring to the same entities
# > <font color="blue">Motor Vehicles International Corp.</font> announced a major management shake-up.... <font color="blue">MVI</font> said the chief executive officer has resigned.... <font color="blue">The Big 10 auto maker</font> is attempting to regain market share.... <font color="blue">It</font> will announce significant losses for the fourth quarter...
# 
# 4. **Relation Classification (RC)**: (text, entity spans, possible relations) $\rightarrow$ relation instances
# 
# ```
# Milka is owned by Kraft
# Cadbury Dairy Milk is owned by Kraft
# Kraft is not owned by Milka
# ...
# ```

# * **Temporal** Information Extraction:
#     * Recognise and/or normalise temporal expressions, e.g., "tomorrow morning at 8" -> "2016-11-26 08:00:00"

# * **Event** Extraction:
#     * Recognise events, typically consisting of entities and relations between them at a point in time and place, e.g., an election

# ##  Relation Extraction
# 
# It would be useful to automatically build a database of this form
# 
# <table style="font-size: x-large; border-style: solid;">
# <tr><th style="text-align: left; border-style: solid;">Brand</th><th style="text-align: left; border-style: solid;">Parent</th></tr>
# <tr><td style="text-align: left; border-style: solid;">KitKat</td><td style="text-align: left; border-style: solid;">Nestle</td></tr>
# <tr><td style="text-align: left; border-style: solid;">Lipton</td><td style="text-align: left; border-style: solid;">Unilever</td></tr>
# <tr><td style="text-align: left; border-style: solid;">...</td><td style="text-align: left; border-style: solid;">...</td></tr>
# </table>

# or this graph:
# ![graph](https://geekologie.com/2012/04/25/parent-companies-large.jpg)

# These are all instances of the "[owned by](https://www.wikidata.org/wiki/Property:P127)" relation.
# Can also be expressed as:
# 
# ```
# owned_by(KitKat, Nestle)
# owned_by(Lipton, Unilever)
# ```

# The web contains a lot of textual evidence for this relation:
# 
# > <font color="blue">Dechra Pharmaceuticals</font>, which has just made its second acquisition, had previously purchased <font color="green">Genitrix</font>.
# 
# > <font color="blue">Trinity Mirror plc</font>, the largest British newspaper, purchased <font color="green">Local World</font>, its rival.
# 
# > <font color="blue">Kraft</font>, owner of <font color="green">Milka</font>, purchased <font color="green">Cadbury Dairy Milk</font> and is now gearing up for a roll-out of its new brand.
# 

# ... and for many other relations.
# 
# ```
# born_in(Barack Obama, Hawaii)
# educated_at(Albert Einstein, University of Zürich)
# occupation(Steve Jobs, businessman)
# spouse(Angela Merkel, Joachim Sauer)
# ...
# ```

# ### Relation Extraction: Examples
# 
# ReVerb ([Fader et al., 2011](https://www.aclweb.org/anthology/D11-1142.pdf)) demo:
# 
# * [who is owned by Nestle?](https://openie.allenai.org/search?arg1=&rel=owned+by&arg2=Nestle&corpora=)
# * [whom did Google purchase?](https://openie.allenai.org/search?arg1=Google&rel=purchased&arg2=&corpora=)
# * [who invented email?](https://openie.allenai.org/search?arg1=who&rel=invented&arg2=email&corpora=)

# ## Reminder: named entity recognition (NER)
# 
# | |
# |-|
# | \[Barack Obama\]<sub>PER</sub> was born in \[Hawaii\]<sub>LOC</sub> |
# 
# | |
# |-|
# | \[Isabelle Augenstein\]<sub>PER</sub> is an associate professor at the \[University of Copenhagen\]<sub>ORG</sub> |

# ### NER as sequence labeling with IOB encoding
# 
# Label tokens as beginning (B), inside (I), or outside (O) a **named entity:**
# 
# | | | | | | |
# |-|-|-|-|-|-|
# | Barack | Obama | was |  born | in | Hawaii |
# | B-PER | I-PER | O |  O | O | B-LOC |
# 
# 
# ||||||||||
# |-|-|-|-|-|-|-|-|-|
# | Isabelle | Augenstein | is | an | associate | professor | at | the | University | of | Copenhagen |
# | B-PER | I-PER   | O | O | O  | O | O  | O | B-ORG | I-ORG | I-ORG         |
# 

# ### Relation Extraction
# 
# Task of extracting **semantic relations between arguments**
# * Arguments are entities
#     * Entity types may be "a company" (ORG), "a person" (PER), "a location" (LOC)
#     * Entities are instances of these types (e.g., "Microsoft", "Bill Gates")
# * Builds on named entity recognition

# ### Relation Extraction: Example
#    
# Step 1: IOB sequence labelling for NER
# 
# | Isabelle | Augenstein | is | an | associate | professor | at | the | University | of | Copenhagen |
# |-|-|-|-|-|-|-|-|-|
# | B-PER | I-PER   | O | O | O  | O | O  | O | B-ORG | I-ORG | I-ORG         |
# 

# Step 2: NE decoding
#   
#   * Isabelle Augenstein: PER  
#   * University of Copenhagen: ORG

# Step 3: Relation extraction
#   
#   
# | Relation   | Entity 1    |Entity 2    |
# |---------|-----------|-----------|
# | associate professor at  | Isabelle Augenstein | University of Copenhagen |

# ### Knowledge Bases
# 
# ![image.png](attachment:image.png)
# 

# ### Manually Created Knowledge Bases
# 
# <center>
#   <table>
#   <tr>
#   <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Wikidata-logo-en.svg/500px-Wikidata-logo-en.svg.png" width="400"></td>
#   <td><img src="https://yago-knowledge.org/assets/images/logo.png" width="400"></td>
#   </tr><tr>
#   <td><img src="https://content.alegion.com/hubfs/datasets/wordnet.png" width="400"></td>
#   <td><img src="https://cst.ku.dk/projekter/dannet/billeder/dannet.png" width="400"></td>
#   </tr>
#   </table>
# </center>

# ### Relation Extraction for Automated Knowledge Base Construction
# 
# ![image.png](attachment:image.png)

# ### Biomedical Processes
# 
# ![image.png](attachment:image.png)

# ### Drug-Gene Interactions
# 
# ![image.png](attachment:image.png)

# ## Relation Extraction as Classification
# * Input space $\mathcal{X}$: argument pairs $\in\mathcal{E}$ and supporting texts $\in\mathcal{S}$
#   * e.g., ((Isabelle Augenstein, University of Copenhagen), "Isabelle Augenstein is an associate professor at the University of Copenhagen")
# * Output space $\mathcal{Y}$: set of relation labels
#   * e.g., $\Ys=\{ \text{assoc-prof-at}, \text{founder-of},\text{employee-at},\text{professor-at},\text{NONE}\}$

# * **Goal**: train model \\(s_{\params}(\x,y)\\) that assigns high *scores* to a correct label $\mathcal{y}$ for $\mathcal{\x}$, low scores otherwise 
# * **Training**: learn parameters \\(\params\\) from training set of $(\mathcal{\x,y})$ pairs
# * **Prediction** of labels for input instances $\mathcal{\x}$: solve maximisation problem $\argmax_y s_{\params}(\x,y)$.

# ## Challenges
# 
# * Diversity: `owned_by` can be expressed as `purchased/acquired/owns/...`
# * Complexity:
# > Wired reports that in a surprising reshuffle at Microsofte, Satya Nadellae has taken over as the managing director of the company.
# * Rare relations: `org:dissolved`
# * Unseen relations: `deciphered`
# * Unseen languages:
# > Der Fluss Amazonas gab seinerseits dem Amazonasbecken sowie mehreren gleichnamigen Verwaltungseinheiten in Brasilien, Venezuela, Kolumbien

# <img src="dl-applications-figures/WS_mapping.png" width="100%"/>
# 
# <div style="text-align: right;">
#     Source: http://ai.stanford.edu/blog/weak-supervision/
# </div>

# ## Relation Extraction Approaches
# * Manually defined linguistic **patterns**: `<active-voice-verb>` followed by `<target-np>=<direct object>`, e.g. `purchased Genitrix`
# * Patterns **learned** from manually-annotated data (supervised)
# * **Bootstrapping**: iterative pattern acquisition from unlabeled data (weak supervision)
# * Use **knowledge bases** to automatically annotate data (distant supervision)

# And of course,
# * Neural approaches: encode and classify

# Few-shot or zero-shot relation extraction:
# * **Universal Schema**, **Open Information Extraction**, **Relation Extraction via Reading Comprehension**...

# ## Labeled Data
# 
# [TACRED](https://nlp.stanford.edu/projects/tacred/) is a popular Relation Extraction dataset.
# 
# [Top performance](https://paperswithcode.com/sota/relation-extraction-on-tacred) is around 75% F1.

# ## Example: Learning Patterns
# * Extract `used_for` (method used for task) relations from sentences in computer science publications
# * Our training data contains pairs of arguments $\mathcal{E}$ for this relation
# 
# Example publications:
# * https://www.sciencedirect.com/science/article/pii/S1474034613000475
# * https://www.sciencedirect.com/science/article/pii/S1474034613000773
# * https://www.sciencedirect.com/science/article/pii/S1474034615000166
# * https://arxiv.org/abs/2104.08481

# * Learn a set of textual patterns for each relation (in this case, just one relation, `used_for`)
# * Assign labels to entity pairs whose sentences match a pattern
#     * Labels: relation types or `NONE`

# ### Closer look at pattern matching
# 
# Sentence:
# 
# > Demonstrates text mining and clustering techniques for building domain ontology.
# 
# Labeled pair:
# 
# `used_for(clustering techniques, building domain ontology)`
# 
# $\downarrow$
# 
# Pattern:
# 
# ```Demonstrates text mining and [...] for [...].```

# This pattern is too specific.
# 
# We need to **generalise** patterns:
# 
# ```Demonstrates ... and [...] for [...].```

# ## Bootstrapping
# 
# * Input: a set of entity pairs
# * Overall idea: extract patterns and entity pairs **iteratively**
# * One of the first algorithms: [DIPRE (Sergey Brin, 1999)](http://ilpubs.stanford.edu:8090/421/1/1999-65.pdf)
# * Two helper methods: 
#     * *use entity pairs* to find/generate (more) patterns
#     * *apply patterns* to find entity pairs
# 

# With each iteration, the number of pattern, entity pairs and extractions increases.
# However, they are less correct: **semantic drift**
# 
# ```[...] for [...].```
# 
# The patterns should not be too general...
# 
# > the cost may still be too high for customers

# ## Supervised Relation Extraction
# * Scoring model \\(s_{\params}(\x,y)\\) is estimated based on training sentences $\mathcal{X}$ and their labels $\mathcal{Y}$
# * At testing time, predict highest-scoring label for each testing instance: $$ \argmax_{\y\in\Ys} s_{\params}(\x,\y) $$
# * Requires both positive (`used_for`) and negative (`NONE`) training examples

# ### Feature extraction
# Represent training and testing data as feature vectors.
# 
# Typical features:
# * Patterns!
# * Shortest dependency path between two entities (see [parsing slides](dependency_parsing_slides.ipynb))
# * Bag-of-words/n-grams (as in examples coming up; using `sklearn`'s built-in feature extractor)
# 
# <center>
#     <img src="parsing_figures/dep4re.png" width="60%">
# </center>

# and of course,
# * word embeddings and neural representations

# ## Distant Supervision
# * Supervised learning typically requires large amounts of hand-labelled training examples
# * It is **time-consuming and expensive** to manually label examples
#     * It is desirable to find ways of automatically or semi-automatically producing more training data
#     * We have already seen one example of this, bootstrapping
# * Downside of bootstrapping: **semantic drift** 
#     * due to the iterative nature of finding good entity pairs and patterns
# * Alternative: distant supervision

# * We still have a set of entity pairs $\mathcal{E}$, their relation types $\mathcal{Y}$ and a set of sentences $\mathcal{X}$ as an input
#     * but we do **not require pre-defined patterns**
# * Instead, entity pairs and relations are obtained from a **knowledge resource**, e.g. the [Wikidata knowledge base](https://www.wikidata.org), Yago or Freebase
# * Those are used to automatically label all sentences with relations
# * Afterwards: supervised learning

# ![image.png](attachment:image.png)

# ### Limitations of distant supervision
# 
# * Overlapping relations
# * Ambiguous entities
# * Mention vs. type annotations
# 
# For example, this relation holds:
# 
# `lives_in(Margrethe II of Denmark,  Amalienborg)`
# 
# but it would be wrong to attribute it to the sentence
# 
# > Margrethe was born 16 April 1940 at Amalienborg

# ## Universal Schema
# * Goal: overcome limitation of pre-defined relations
# * Viewing patterns **as relations themselves**
# 

# The space of entity pairs and relations is defined by a matrix:
# 
# |  | demonstrates [...] for [...] | [...] is capable of [...] | an [...] model is employed for [...] | [...] decreases the [...] | `used_for` |
# | ------ | ----------- |
# | 'text mining', 'building domain ontology' | 1 |  |  |  | 1 |
# | 'ensemble classifier', 'detection of construction materials' |  |  | 1 |  | 1 |
# | 'data mining', 'characterization of wireless systems performance'|  | 1 |  |  | ? |
# | 'frequency domain', 'computational cost' |  |  |  | 1 | ? |

# * `used_for` is a pre-defined relation, others are patterns
# * Co-occurrence is signified by a '1'
# * We would like to fill in the '?' cells

# Training data:
# * **Positive relations and entity pairs** from the annotated data
# * **Negative entity pairs and relations** *sampled randomly* from empty cell in the matrix

# ## Model: Neural Matrix Factorisation for Recommender Systems
# 
# <center>
#     <img src="dl-applications-figures/neural_mf.png" width=800/> 
# </center>
# 
# <div style="text-align: right;">
#     (from [Zhang et al., 2017](https://arxiv.org/abs/1707.07435))
# </div>

# ![image.png](attachment:image.png)

# ## Summary so far
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

# ## Relation extraction via reading comprehension
# 
# &nbsp;
# 
# <center>
#     <a href="slides/zeroshot-relation-extraction-via-reading-comprehension-conll-2017.pdf">
#     <img src="https://d3i71xaburhd42.cloudfront.net/fa025e5d117929361bcf798437957762eb5bb6d4/4-Figure2-1.png" width="100%">
#     </a>
# </center>
# 
# <div style="text-align: right;">
#     (from [Levy et al., 2017](https://www.aclweb.org/anthology/K17-1034.pdf); [slides](https://levyomer.files.wordpress.com/2017/08/zeroshot-relation-extraction-via-reading-comprehension-conll-2017.pptx))
# </div>

# ## Background Material
# 
# * Cardie, 1997. Empirical Methods in Information Extraction: https://ojs.aaai.org//index.php/aimagazine/article/view/1322
# * Jurafky, Dan and Martin, James H. (2016). Speech and Language Processing, Chapter 18 (Information Extraction): https://web.stanford.edu/~jurafsky/slp3/18.pdf
# * Riedel, Sebastian and Yao, Limin and McCallum, Andrew and Marlin, Benjamin M. (2013). Relation extraction with Matrix Factorization and Universal Schemas. Proceedings of NAACL.  http://www.aclweb.org/anthology/N13-1008

# ## Further Reading
# 
# * Abdou, M., Sas, C., Aralikatte, R., Augenstein, I., & Søgaard, A. (2019). X-WikiRE: A large, multilingual resource for relation extraction as machine comprehension. Proceedings of ACL. https://aclanthology.org/D19-6130/
# * Shantanu Kumar (2017). A Survey of Deep Learning Methods for Relation Extraction. https://arxiv.org/pdf/1705.03645.pdf
# * Alex Ratner, Stephen Bach, Paroma Varma, Chris Ré (2018). Weak Supervision: The New Programming Paradigm for Machine Learning. https://dawn.cs.stanford.edu/2017/07/16/weak-supervision/
# * Rosenman, Shachar, Alon Jacovi, and Yoav Goldberg (2020). Exposing Shallow Heuristics of Relation Extraction Models with Challenge Data. Proceedings of EMNLP. https://arxiv.org/pdf/2010.03656.pdf
# * Awesome relation extraction, curated list of resources on relation extraction. https://github.com/roomylee/awesome-relation-extraction
