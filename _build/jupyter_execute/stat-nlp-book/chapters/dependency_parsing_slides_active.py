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

# ## Reading material survey
# 
# ### https://ucph.page.link/w44
# 
# https://docs.google.com/forms/d/1Su1uANwJ8HM5MKOqXcJKZtJrgVISmR82CxDAHHaV2as/edit#responses

# ## Schedule
# 
# + Parsing motivation
# 
# + Exercise: dependency syntax
# 
# + Break
# 
# + Parsing algorithms
# 
# + Exercise: transition-based parsing
# 
# + Summary

# ## But first, debt from three weeks ago...
# 
# ### https://ucph.padlet.org/dh/qa

# ## Motivation: information extraction
# 
# > <font color="blue">Dechra Pharmaceuticals</font>, which has just made its second acquisition, had previously purchased <font color="green">Genitrix</font>.
# 
# > <font color="blue">Trinity Mirror plc</font>, the largest British newspaper, purchased <font color="green">Local World</font>, its rival.
# 
# > <font color="blue">Kraft</font>, owner of <font color="blue">Milka</font>, purchased <font color="green">Cadbury Dairy Milk</font> and is now gearing up for a roll-out of its new brand.
# 

# Syntactic dependency trees to the rescue!
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

# ## Motivation: machine translation
# 
# <center>
#     <img src="https://d3i71xaburhd42.cloudfront.net/f4c750cdf8f557eea3a4b76be16e99ec15f0c92b/3-Figure2-1.png" width=100%>
# </center>
# 
# <div style="text-align: right;">
#     (from [Rasooli et al., 2021](https://arxiv.org/abs/2104.08384))
# </div>

# ## Syntactic ambiguity

# In[4]:


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
# display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"2400px")


# <center>
#     <img src="parsing_figures/telescope1.jpeg" width=30%/>
# </center>

# ## Syntactic ambiguity

# In[35]:


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
# display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"2400px")


# <center>
#     <img src="parsing_figures/telescope2.jpg" width=30%/>
# </center>

# ### Need for universal syntactic annotation
# 
# <center>
#     <img src="../img/ud.png" width=60%>
# </center>
# 
# <div style="text-align: right;">
#     (from [de Lhoneux, 2019](https://cl.lingfil.uu.se/~miryam/assets/pdf/thesis.pdf))
# </div>

# ### Universal Dependencies
# 
# * Annotation framework featuring [37 syntactic relations](https://universaldependencies.org/u/dep/all.html)
# * [Treebanks](http://universaldependencies.org/) in over 100 languages
# * Large project with [over 200 contributors](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3687)
# * Linguistically universal [annotation guidelines](https://universaldependencies.org/guidelines.html)

# ### UD dependency relations
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

# ## Universal POS tags (UPOS)
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

# ### Danish example

# In[36]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Alice	Alice	NOUN	_	_	2	nsubj	_	_
2	så	se	VERB	_	_	0	root	_	_
3	Bob	Bob	PROPN	_	_	2	obj	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"1400px")


# ### Korean example

# In[37]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	앨리스는	앨리스+는	NOUN	_	_	3	nsubj	_	_
2	밥을	밥+을	NOUN	_	_	3	obj	_	_
3	보았다	보+았+다	VERB	_	_	0	root	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))
arcs, tokens = to_displacy_graph(*load_arcs_tokens(conllu))
render_displacy(arcs, tokens,"1400px")


# ## Dependency syntax exercise
# 
# ### https://ucph.page.link/dep

# ## Dependency parsing
# 
# * Predict **head** and **relation** for each word.
# * Classification? Sequence tagging? Sequence-to-sequence? Span selection? Or something else?

# In[38]:


conllu = """
# ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
1	Alice	_	_	_	_	2	nsubj	_	_
2	saw	_	_	_	_	0	root	_	_
3	Bob	_	_	_	_	2	dobj	_	_
"""
display(HTML(pd.read_csv(StringIO(conllu), sep="\t").to_html(index=False)))


# ## Dependency parsing approaches
# 
# * Graph-based: score all possible parts (e.g. word pairs), find best combination (e.g. maximum spanning tree)
# * Transition-based: incrementally build the tree, one arc at a time, by applying a sequence of actions

# ## Transition-based parsing: configuration
# 
# Consists of a buffer, stack and set of arcs created so far.

# ### Buffer
# of tokens waiting for processing

# In[39]:


render_transitions_displacy(transitions[0:1], tokenized_sentence)


# ### Stack
# of tokens currently being processed

# In[40]:


render_transitions_displacy(transitions[2:3],tokenized_sentence)


# ### Parse tree (set of arcs)
# tree built so far

# In[41]:


render_transitions_displacy(transitions[6:7], tokenized_sentence)


# ### Configuration
# 
# - Stack \\(S\\): a last-in, first-out memory to keep track of words to process later
# - Buffer \\(B\\): words not processed yet
# - Arcs \\(A\\): the dependency arcs created so far

# What are the possible actions? Depends which system we are using!

# ### arc-standard
# 
# - SHIFT: move the buffer top to the stack.
# - RIGHT-ARC: create arc from second stack item to stack top. Pop stack top.
# - LEFT-ARC: create arc from stack top to second stack item. Pop second stack item.

# Two special configurations:
# - initial: buffer contains the words, stack contains root, and arcs are empty
# - terminal: buffer is empty, stack contains only root

# ### arc-standard: example
# 
# <center>
#     <img src="parsing_figures/tb_example.png" width=100%/>
# </center>

# ## Alternative transition systems
# + arc-standard ([Nivre, 2003](https://aclanthology.org/W03-3017/))
# + arc-eager ([Nivre, 2004](https://www.aclweb.org/anthology/W04-0308))
# + arc-hybrid ([Kuhlmann et al., 2011](https://aclanthology.org/P11-1068/))

# ## arc-hybrid
# 
# ### https://danielhers.github.io/archybrid.pdf
# 
# https://app.quizalize.com/dash/R3JvdXA6YTUzMGNkZjItYTRiYS00NGM2LTk3ZGEtZDc4YjlkMjkyODg4/activity/QWN0aXZpdHk6MmU3NzUxYjQtMjljNy00ZTI2LWFiOTMtNjM2ZWUxZWNjMGI2/overview

# ## arc-hybrid vs arc-standard
# <table class="typeindex">
#   <thead>
#     <tr style="font-size: x-large; text-align: left;">
#       <th></th>
#       <th>arc-standard</th>
#       <th>arc-hybrid</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;">LEFT-ARC</td>
#       <td style="text-align: left;">create arc from stack top to second stack item. Pop second stack item.</td>
#       <td style="text-align: left;">create arc from buffer top to stack top. Pop stack top.</td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;">initial configuration</td>
#       <td style="text-align: left;">stack contains root, buffer contains words</td>
#       <td style="text-align: left;">stack is empty, buffer contains words and root</td>
#     </tr>
#     <tr style="font-size: x-large; text-align: left;">
#       <td style="text-align: left;">terminal configuration</td>
#       <td style="text-align: left;">stack contains root, buffer is empty</td>
#       <td style="text-align: left;">stack is empty, buffer contains root</td>
#     </tr>
#   </tbody>
# </table>

# ## Learning a transition-based parser
# 
# * Decompose parse tree into a sequence of **actions**
# * Learn to score individual actions
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

# ## Dependency parsing evaluation
# 
# * Unlabeled Attachment Score (**UAS**): % of words with correct head
# * Labeled Attachment Score (**LAS**): % of words with correct head and label
# 
# Always 0 $\leq$ LAS $\leq$ UAS $\leq$ 100%.

# ### Example: LAS and UAS
# 
# <center>
#     <img src="parsing_figures/as.png" width=70%/>
# </center>
# 
# <center>
#     $\mathrm{UAS}=\frac{8}{12}=67\%$
# </center>
# 
# <center>
#     $\mathrm{LAS}=\frac{7}{12}=58\%$
# </center>

# ## Summary
# 
# * **Dependency parsing** predicts word-to-word dependencies
# * Treebanks in many languages, thanks to **UD**
# * Fast and accurate parsing, e.g. **transition-based**

# ## Further reading
# 
# * [EACL 2014 tutorial](http://stp.lingfil.uu.se/~nivre/eacl14.html)
# * [Beyond dependency parsing: meaning representations](https://danielhers.github.io/mr.pdf)

# ## Evaluation
# 
# ### https://ucph.page.link/eval
# 
# https://docs.google.com/forms/d/1uZtVR5vQaTRVwoUvEBrBixjzsHEGdimEpk0reBgYsKI/edit#responses
