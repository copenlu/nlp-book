#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', '<script>\n  function code_toggle() {\n    if (code_shown){\n      $(\'div.input\').hide(\'500\');\n      $(\'#toggleButton\').val(\'Show Code\')\n    } else {\n      $(\'div.input\').show(\'500\');\n      $(\'#toggleButton\').val(\'Hide Code\')\n    }\n    code_shown = !code_shown\n  }\n\n  $( document ).ready(function(){\n    code_shown=false;\n    $(\'div.input\').hide()\n  });\n</script>\n<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>\n<style>\n.rendered_html td {\n    font-size: xx-large;\n    text-align: left; !important\n}\n.rendered_html th {\n    font-size: xx-large;\n    text-align: left; !important\n}\n</style>\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\nimport sys\nsys.path.append("..")\nfrom statnlpbook.util import execute_notebook\nimport statnlpbook.parsing as parsing\nfrom statnlpbook.transition import *\nfrom statnlpbook.dep import *\nimport pandas as pd\nfrom io import StringIO\nfrom IPython.display import display, HTML\n\nexecute_notebook(\'transition-based_dependency_parsing.ipynb\')\n')


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


# # Parsing

# ## Schedule
# 
# + Parsing motivation
# 
# + Background: parsing (10 min.)
# 
# + Exercise: multi-word expressions (10 min.)
# 
# + Background: Universal Dependencies (5 min.)
# 
# + Background: transition-based parsing (10 min.)
# 
# + Break (10 min.)
# 
# + Example: transition-based parsing (5 min.)
# 
# + Motivation: natural language understanding (5 min.)
# 
# + Background: learning to parse (10 min.)
# 
# + Math: dependency parsing evaluation (5 min.)
# 
# + Examples: dependency parsers (5 min.)
# 
# + Background: semantic parsing (15 min.)
# 

# ## Motivation: information extraction
# 
# > <font color="blue">Dechra Pharmaceuticals</font>, which has just made its second acquisition, had previously purchased <font color="green">Genitrix</font>.
# 
# > <font color="blue">Trinity Mirror plc</font>, the largest British newspaper, purchased <font color="green">Local World</font>, its rival.
# 
# > <font color="blue">Kraft</font>, owner of <font color="blue">Milka</font>, purchased <font color="green">Cadbury Dairy Milk</font> and is now gearing up for a roll-out of its new brand.
# 

# Check out [UDPipe](https://lindat.mff.cuni.cz/services/udpipe/run.php?model=english-ewt-ud-2.6-200830&data=Kraft,%20owner%20of%20Milka,%20purchased%20Cadbury%20Dairy%20Milk%20and%20is%20now%20gearing%20up%20for%20a%20roll-out%20of%20its%20new%20brand.) and [Stanza](http://stanza.run/).

# ## Motivation: question answering by reading comprehension
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/05dd7254b632376973f3a1b4d39485da17814df5/6-Figure4-1.png" width=100%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Rajpurkar et al., 2016](https://aclanthology.org/D16-1264))
# </div>

# ## Motivation: question answering from knowledge bases
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/faee0c81a1170402b149500f1b91c51ccaf24027/2-Figure1-1.png" width=50%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Reddy et al., 2017](https://aclanthology.org/D17-1009/))
# </div>

# Parsing is is the process of **constructing these graphs**:
# 
# * very important for downstream applications
# * researched in academia and [industry](https://ai.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html)

# How is this done?

# ## Syntactic Dependencies
# 
# * **Lexical Elements**: words
# * **Syntactic Relations**: subject, direct object, nominal modifier, etc. 
# 
# Task: determine the syntactic relations between words

# ### Grammatical Relations
# > <font color="blue">Kraft</font>, owner of <font color="blue">Milka</font>, purchased <font color="green">Cadbury Dairy Milk</font> and is now gearing up for a roll-out of its new brand.
# 
# * *Subject* of **purchased**: Kraft
# * *Object* of **purchased**: Cadbury

# In[4]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Kraft	Kraft	NOUN	NN	_	7	nsubj	_	_
2	,	,	PUNCT	,	_	1	punct	_	_
3	owner	owner	NOUN	NN	_	1	appos	_	_
4	of	of	ADP	IN	_	5	case	_	_
5	Milka	Milka	PROPN	NNP	_	3	nmod	_	_
6	,	,	PUNCT	,	_	7	punct	_	_
7	purchased	purchase	VERB	VBD	_	0	root	_	_
8	Cadbury	Cadbury	PROPN	NNP	_	7	dobj	_	_
9	Dairy	Dairy	PROPN	NNP	_	8	flat	_	_
10	Milk	milk	PROPN	NNP	_	8	flat	_	_
"""
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"1200px")


# ## Anatomy of a Dependency Tree
# 
# * Nodes (vertices):
#     * Words of the sentence (+ punctuation tokens)
#     * a ROOT node
# * Arcs (edges):
#     * Directed from syntactic **head** to **dependent**
#     * Each **non-ROOT** token has **exactly one head**
#         * the word that controls its syntactic function, or
#         * the word "it depends on"
# * ROOT **has no head**
# 

# In[54]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Kraft	Kraft	NOUN	NN	_	7	nsubj	_	_
2	,	,	PUNCT	,	_	1	punct	_	_
3	owner	owner	NOUN	NN	_	1	appos	_	_
4	of	of	ADP	IN	_	5	case	_	_
5	Milka	Milka	PROPN	NNP	_	3	nmod	_	_
6	,	,	PUNCT	,	_	7	punct	_	_
7	purchased	purchase	VERB	VBD	_	0	root	_	_
8	Cadbury	Cadbury	PROPN	NNP	_	7	dobj	_	_
9	Dairy	Dairy	PROPN	NNP	_	8	flat	_	_
10	Milk	milk	PROPN	NNP	_	8	flat	_	_
"""
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"1200px")


# ### Example
# 
# (in [CoNLL-U Format](https://universaldependencies.org/format.html))

# In[55]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Alice	_	_	_	_	2	nsubj	_	_
2	saw	_	_	_	_	0	root	_	_
3	Bob	_	_	_	_	2	dobj	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"900px")


# ### https://ucph.padlet.org/dh/mw

# ### Need for Universal Syntax
# 
# ### https://cl.lingfil.uu.se/~nivre/docs/NivreCLIN2020.pdf

# ### Universal Syntax
# 
# English and Danish are similar, while others are more distant:
# ![similarities](https://www.mitpressjournals.org/na101/home/literatum/publisher/mit/journals/content/coli/2019/coli.2019.45.issue-2/coli_a_00351/20190614/images/large/00351f03c.jpeg)
# 
# <div style="text-align: right;">
#     Left: clustering based on syntactic dependencies; right: genetic tree
#     (from <a href="https://www.mitpressjournals.org/doi/full/10.1162/coli_a_00351">Bjerva et al., 2019</a>)
# </div>

# ### Danish Example

# In[56]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Alice	Alice	NOUN	_	_	2	nsubj	_	_
2	så	se	VERB	_	_	0	root	_	_
3	Bob	Bob	PROPN	_	_	2	obj	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"900px")


# ### Korean Example

# In[57]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	앨리스는	앨리스+는	NOUN	_	_	3	nsubj	_	_
2	밥을	밥+을	NOUN	_	_	3	obj	_	_
3	보았다	보+았+다	VERB	_	_	0	root	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"900px")


# ### Longer English Example

# In[58]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Kraft	Kraft	NOUN	NN	_	7	nsubj	_	_
2	,	,	PUNCT	,	_	1	punct	_	_
3	owner	owner	NOUN	NN	_	1	appos	_	_
4	of	of	ADP	IN	_	5	case	_	_
5	Milka	Milka	PROPN	NNP	_	3	nmod	_	_
6	,	,	PUNCT	,	_	7	punct	_	_
7	purchased	purchase	VERB	VBD	_	0	root	_	_
8	Cadbury	Cadbury	PROPN	NNP	_	7	dobj	_	_
9	Dairy	Dairy	PROPN	NNP	_	8	flat	_	_
10	Milk	milk	PROPN	NNP	_	8	flat	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"1200px")


# ### Universal Dependencies 
# 
# * Annotation framework featuring [37 syntactic relations](https://universaldependencies.org/u/dep/all.html)
# * [Treebanks](http://universaldependencies.org/) in over 90 languages
# * Large project with over 200 contributors
# * Linguistically universal [annotation guidelines](https://universaldependencies.org/guidelines.html)

# ### UD Dependency Relations
# 
# <table border="1">
#   <tr style="background-color:cornflowerblue; font-size: x-large; text-align: left;">
#       <td style="text-align: left;"> </td>
#       <td style="text-align: left;"> Nominals </td>
#       <td style="text-align: left;"> Clauses </td>
#       <td style="text-align: left;"> Modifier words </td>
#       <td style="text-align: left;"> Function Words </td>
#   </tr>
#   <tr style="font-size: x-large; text-align: left;">
#       <td style="background-color:darkseagreen">
# 	Core arguments
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/nsubj.html" title="u-dep nsubj">nsubj</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/obj.html" title="u-dep obj">obj</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/iobj.html" title="u-dep iobj">iobj</a>
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/csubj.html" title="u-dep csubj">csubj</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/ccomp.html" title="u-dep ccomp">ccomp</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/xcomp.html" title="u-dep xcomp">xcomp</a>
#       </td>
# 	  <td style="text-align: left;"></td><td style="text-align: left;"></td>
#   </tr>
#   <tr style="font-size: x-large; text-align: left;">
#       <td style="background-color:darkseagreen;">
# 	Non-core dependents
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/obl.html" title="u-dep obl">obl</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/vocative.html" title="u-dep vocative">vocative</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/expl.html" title="u-dep expl">expl</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/dislocated.html" title="u-dep dislocated">dislocated</a>
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/advcl.html" title="u-dep advcl">advcl</a>
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/advmod.html" title="u-dep advmod">advmod</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/discourse.html" title="u-dep discourse">discourse</a>
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/aux_.html" title="u-dep aux">aux</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/cop.html" title="u-dep cop">cop</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/mark.html" title="u-dep mark">mark</a>
#       </td>
#   </tr>
#   <tr style="font-size: x-large; text-align: left;">
#       <td style="background-color:darkseagreen">
# 	Nominal dependents
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/nmod.html" title="u-dep nmod">nmod</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/appos.html" title="u-dep appos">appos</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/nummod.html" title="u-dep nummod">nummod</a>
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/acl.html" title="u-dep acl">acl</a>
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/amod.html" title="u-dep amod">amod</a>
#       </td>
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/det.html" title="u-dep det">det</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/clf.html" title="u-dep clf">clf</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/case.html" title="u-dep case">case</a>
#       </td>
#   </tr style="font-size: x-large; text-align: left;">
#   <tr style="background-color:cornflowerblue; font-size: x-large; text-align: left;">	
#       <td style="text-align: left;"> Coordination </td>
#       <td style="text-align: left;"> MWE </td>
#       <td style="text-align: left;"> Loose </td>
#       <td style="text-align: left;"> Special </td>
#       <td style="text-align: left;"> Other </td>
#   </tr>
#   <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;">
# 	    <a href="https://universaldependencies.org/u/dep/conj.html" title="u-dep conj">conj</a><br>
# 	    <a href="https://universaldependencies.org/u/dep/cc.html" title="u-dep cc">cc</a>
#       </td>
#       <td style="text-align: left;">
# 	  <a href="https://universaldependencies.org/u/dep/fixed.html" title="u-dep fixed">fixed</a><br>
# 	  <a href="https://universaldependencies.org/u/dep/flat.html" title="u-dep flat">flat</a><br>
# 	  <a href="https://universaldependencies.org/u/dep/compound.html" title="u-dep compound">compound</a>
#     </td>
#     <td style="text-align: left;">
# 	  <a href="https://universaldependencies.org/u/dep/list.html" title="u-dep list">list</a><br>
# 	  <a href="https://universaldependencies.org/u/dep/parataxis.html" title="u-dep parataxis">parataxis</a>
#     </td>
#     <td style="text-align: left;">
# 	  <a href="https://universaldependencies.org/u/dep/orphan.html" title="u-dep orphan">orphan</a><br>
# 	  <a href="https://universaldependencies.org/u/dep/goeswith.html" title="u-dep goeswith">goeswith</a><br>
# 	  <a href="https://universaldependencies.org/u/dep/reparandum.html" title="u-dep reparandum">reparandum</a>
#     </td>
#     <td style="text-align: left;">
# 	  <a href="https://universaldependencies.org/u/dep/punct.html" title="u-dep punct">punct</a><br>
# 	  <a href="https://universaldependencies.org/u/dep/root.html" title="u-dep root">root</a><br>
# 	  <a href="https://universaldependencies.org/u/dep/dep.html" title="u-dep dep">dep</a>
#     </td>
#   </tr>
# </table>

# ## Universal POS Tags (UPOS)
# 
# As opposed to language-specific POS tags (XPOS).
# 
# <table class="typeindex">
#   <thead>
#     <tr style="font-size: x-large; text-align: left;">
#       <th>Open class words</th>
#       <th>Closed class words</th>
#       <th>Other</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/ADJ.html" class="doclink doclabel" title="u-pos ADJ">ADJ</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/ADP.html" class="doclink doclabel" title="u-pos ADP">ADP</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/PUNCT.html" class="doclink doclabel" title="u-pos PUNCT">PUNCT</a></td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/ADV.html" class="doclink doclabel" title="u-pos ADV">ADV</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/AUX_.html" class="doclink doclabel" title="u-pos AUX">AUX</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/SYM.html" class="doclink doclabel" title="u-pos SYM">SYM</a></td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/INTJ.html" class="doclink doclabel" title="u-pos INTJ">INTJ</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/CCONJ.html" class="doclink doclabel" title="u-pos CCONJ">CCONJ</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/X.html" class="doclink doclabel" title="u-pos X">X</a></td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/NOUN.html" class="doclink doclabel" title="u-pos NOUN">NOUN</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/DET.html" class="doclink doclabel" title="u-pos DET">DET</a></td>
#       <td style="text-align: left;">&nbsp;</td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/PROPN.html" class="doclink doclabel" title="u-pos PROPN">PROPN</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/NUM.html" class="doclink doclabel" title="u-pos NUM">NUM</a></td>
#       <td style="text-align: left;">&nbsp;</td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/VERB.html" class="doclink doclabel" title="u-pos VERB">VERB</a></td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/PART.html" class="doclink doclabel" title="u-pos PART">PART</a></td>
#       <td style="text-align: left;">&nbsp;</td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;">&nbsp;</td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/PRON.html" class="doclink doclabel" title="u-pos PRON">PRON</a></td>
#       <td style="text-align: left;">&nbsp;</td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;">&nbsp;</td>
#       <td style="text-align: left;"><a href="https://universaldependencies.org/u/pos/SCONJ.html" class="doclink doclabel" title="u-pos SCONJ">SCONJ</a></td>
#       <td style="text-align: left;">&nbsp;</td>
#     </tr>
#   </tbody>
# </table>

# ## Dependency Parsing
# 
# * Predict **head** and **relation** for each word.
# * Structured prediction, just like POS tagging.
# * Or is it?

# In[59]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Alice	_	_	_	_	2	nsubj	_	_
2	saw	_	_	_	_	0	root	_	_
3	Bob	_	_	_	_	2	dobj	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))


# ## Dependency Parsing Approaches
# 
# * Graph-based: score all possible parts (e.g. word pairs), find best combination (e.g. maximum spanning tree)
# * Transition-based: incrementally build the tree, one arc at a time, by applying a sequence of actions

# ## Transition-Based Parsing
# 
# * Learn to perform the right action / transition in a bottom-up left-right parser
# * Train classifiers $p(y|\x)$ where $y$ is an action, and $\x$ is the **parser state**
# * Many possible transition systems; shown here: **arc-standard** ([Nivre, 2004](https://www.aclweb.org/anthology/W04-0308))

# ## Configuration (Parser State)
# 
# Consists of a buffer, stack and set of arcs created so far.

# ### Buffer
# of tokens waiting for processing

# In[26]:


render_transitions_displacy(transitions[0:1], tokenized_sentence)


# ### Stack
# of tokens currently being processed

# In[27]:


render_transitions_displacy(transitions[2:3],tokenized_sentence)


# ### Parse (set of arcs)
# tree built so far

# In[60]:


render_transitions_displacy(transitions[6:7], tokenized_sentence)


# We use the following 
# ### Actions

# ### Shift
# 
# Push the word at the top of the buffer to the stack. 
# 
# $$
# (S, i|B, A)\rightarrow(S|i, B, A)
# $$

# In[29]:


render_transitions_displacy(transitions[0:2], tokenized_sentence)


# ### rightArc-[label]
# 
# Add labeled arc from secondmost top node of stack \\(i\\) to top of the stack \\(j\\). Pop the top of the stack.
# 
# $$
# (S|i|j, B, A) \rightarrow (S|i, B, A\cup\{(i,j,l)\})
# $$
# 

# In[61]:


render_transitions_displacy(transitions[4:7], tokenized_sentence)


# ### leftArc-[label] 
# 
# Add labeled arc from top of stack, \\(j\\), to secondmost top node of stack, \\(i\\). Reduce the secondmost top node of the stack.
# 
# $$
# (S|i|j, B, A) \rightarrow (S|j, B, A\cup\{(j,i,l)\})
# $$
# 

# In[62]:


render_transitions_displacy(transitions[2:4], tokenized_sentence)


# ## Full Example

# In[63]:


render_transitions_displacy(transitions[:], tokenized_sentence)


# <center>
#     <img src="parsing_figures/tb_example.png" width=100%/>
# </center>

# ### Summary: Configuration
# 
# **Configuration**:
# - Stack \\(S\\): a last-in, first-out memory to keep track of words to process later
# - Buffer \\(B\\): words not processed so far
# - Arcs \\(A\\): the dependency edges predicted so far
# 
# We further define two special configurations:
# - initial: buffer is initialised to the words in the sentence, stack contains root, and arcs are empty
# - terminal: buffer is empty, stack contains only root

# ### Summary: Actions
# 
# - shift: Push the word at the top of the buffer to the stack \\((S, i|B, A)\rightarrow(S|i, B, A)\\)
# - rightArc-label: Add labeled arc from secondmost top node of stack \\(i\\) to top of the stack \\(j\\). Pop the top of the stack.
# - leftArc-label: Add labeled arc from top of stack, \\(j\\), to secondmost top node of stack, \\(i\\). Reduce the secondmost top node of the stack.

# ## Syntactic Ambiguity

# In[51]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	I	_	_	_	_	2	nsubj	_	_
2	saw	_	_	_	_	0	root	_	_
3	the	_	_	_	_	4	det	_	_
4	star	_	_	_	_	2	dobj	_	_
5	with	_	_	_	_	7	case	_	_
6	the	_	_	_	_	7	det	_	_
7	telescope	_	_	_	_	2	obl	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"900px")


# <center>
#     <img src="parsing_figures/telescope1.jpeg" width=30%/>
# </center>

# ## Syntactic Ambiguity

# In[52]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	I	_	_	_	_	2	nsubj	_	_
2	saw	_	_	_	_	0	root	_	_
3	the	_	_	_	_	4	det	_	_
4	star	_	_	_	_	2	dobj	_	_
5	with	_	_	_	_	7	case	_	_
6	the	_	_	_	_	7	det	_	_
7	telescope	_	_	_	_	4	nmod	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"900px")


# <center>
#     <img src="parsing_figures/telescope2.jpg" width=30%/>
# </center>

# ## Learning a Transition-Based Parser
# 
# * Decompose parse tree into a sequence of **actions**
# * Learn to score individual actions
#     * Structured prediction problem!
#     * Sequence labeling? Sequence-to-sequence?
# 
# <center>
#     <img src="parsing_figures/tb1.png" width=60%/>
# </center>

# How to decide what action to take? 
# 
# * Learn a discriminative classifier $p(y | \x)$ where 
#    * $\x$ is a representation of buffer, stack and parse
#    * $y$ is the action to choose
# * Current state-of-the-art systems use neural networks as classifiers (Bi-LSTMs, Transformers, BERT)
# * Use **greedy search** or **beam search** to find the highest scoring sequence of steps

# <center>
#     <img src="parsing_figures/tb2.png" width=30%/>
# </center>

# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/8292a74aba4eab2ca864b457c17b02634fef4ddd/5-Figure7-1.png" width=30%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/K18-2010.pdf">Hershcovich et al., 2018</a>)
# </div>

# ### Oracle
# 
# How do we get training data for the classifier?
# 
# * Training data: whole trees labelled as correct
# * We need to design an **oracle**
#     * function that, given a sentence and its dependency tree, recovers the sequence of actions used to construct it
#     * can also be thought of reverse engineering a tree into a sequence of actions
# * An oracle does this for every possible parse tree
# * Oracle can also be thought of as human demonstrator teaching the parser

# ## Dependency Parsing Evaluation
# 
# * Unlabeled Attachment Score (**UAS**): % of words with correct head
# * Labeled Attachment Score (**LAS**): % of words with correct head and label
# 
# Always 0 $\leq$ LAS $\leq$ UAS $\leq$ 100%.

# ### Example: LAS and UAS
# 
# <center>
#     <img src="parsing_figures/as.png" width=80%/>
# </center>
# 
# <center>
#     $\mathrm{UAS}=\frac{8}{12}=67\%$
# </center>
# 
# <center>
#     $\mathrm{LAS}=\frac{7}{12}=58\%$
# </center>

# ### State-of-the-Art in Dependency Parsing
# 
# * [CoNLL 2018 Shared Task](https://universaldependencies.org/conll18/results-las.html)
# * [IWPT 2020 Shared Task](http://pauillac.inria.fr/~seddah/coarse_IWPT_SharedTask_unofficial_results.html)
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/d524efd5fe910c0f03c67cd3ba5335d95a5ee4fa/5-Figure1-1.png" width=60%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://universaldependencies.org/conll18/proceedings/pdf/K18-2005.pdf">Che et al., 2018</a>)
# </div>

# ### NN Parsers
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/a14045a751f5d8ed387c8630a86a3a2861b90643/4-Figure2-1.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/D14-1082.pdf">Chen and Manning, 2014</a>)
# </div>

# ### Stack LSTMs
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/396b7932beac62a72288eaea047981cc9a21379a/4-Figure2-1.png" width=80%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/P15-1033.pdf">Dyer et al., 2015</a>)
# </div>

# ### Transition-Based Neural Networks
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/6d671e0e26d239bd6a0b8f67d5fc49a76d733f29/4-Figure3-1.png" width=100%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://arxiv.org/pdf/1703.04474.pdf">Kong et al., 2017</a>)
# </div>

# ### mBERT for zero-shot cross-lingual parsing
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/31c872514c28a172f7f0221c8596aa5bfcdb9e98/1-Figure1-1.png" width=30%/>
# </center>
# 
# <div style="text-align: right;">
#     (from <a href="https://www.aclweb.org/anthology/D19-1279.pdf">Kondratyuk and Straka, 2019</a>)
# </div>

# ## Beyond Dependency Parsing: Meaning Representations
# 
# ### https://danielhers.github.io/dikubits_20200218.pdf

# ## Summary
# 
# * **Dependency parsing** predicts word-to-word dependencies
# * Simple annotations in many languages, thanks to **UD**
# * Fast parsing, e.g. **transition-based**
# * Sufficient for most **down-stream applications**
# * More sophisticated **meaning representations** are more informative but harder to parse

# ## Background Material
# 
# * Arc-standard transition-based parsing system ([Nivre, 2004](https://www.aclweb.org/anthology/W04-0308))
# * [EACL 2014 tutorial](http://stp.lingfil.uu.se/~nivre/eacl14.html)
# * Jurafsky & Martin, [Speech and Language Processing (Third Edition)](https://web.stanford.edu/~jurafsky/slp3/15.pdf): Chapter 15, Dependency Parsing.
