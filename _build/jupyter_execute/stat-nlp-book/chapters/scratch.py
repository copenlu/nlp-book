#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path.append("..")
from graphviz import Digraph
sys.path.append("..")
import statnlpbook.util as util 


# In[2]:


from graphviz import Digraph

dot = Digraph(comment='The Round Table')

dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')

dot.edges(['AB', 'AL'])
# dot.edge('B', 'L', constraint='true')

dot


# In[3]:


from bokeh.plotting import figure, output_notebook, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 1]

# output to static HTML file
output_notebook()

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend="Temp.", line_width=2)

# show the results
show(p)


# In[4]:


[(1,2)] * 5


# In[5]:


"(Bush|||Russia)".split("|||")


# In[6]:


import re
re.split("\|\|\||\(|\)", "(Bush|||Russia)")


# In[7]:


re.split("z=$", "(Bush|||Russia)")


# In[8]:


import tikzmagic


# In[9]:


get_ipython().run_cell_magic('tikz', '', '\\tikzset{every node/.style={font=\\sffamily,white}}\n\n\\node[fill=red] at (0,0) (a) {Can I make this $\\sum$ longer?};\n\\node[fill=blue] at (3,1) (b) {That};\n\\node[fill=blue] at (0,-1) (c) {This};\n\\draw[->] (a) edge (b) (a) edge (c);\n')


# In[10]:


get_ipython().run_cell_magic('tikz', '', '\\input{../fig/test.tex}\n')


# In[11]:


import pandas as pd
pd.DataFrame([('Ab',1),('C',2),('asdasd',2)], columns=['Name', 'Age'])


# In[12]:


get_ipython().run_cell_magic('latex', '', '\\usepackage{algorithm}\n\\begin{algorithmic}\n\\If {$i\\geq maxval$}\n    \\State $i\\gets 0$\n\\Else\n    \\If {$i+k\\leq maxval$}\n        \\State $i\\gets i+k$\n    \\EndIf\n\\EndIf\n\\end{algorithmic}\n')


# $\operatorname{sgn}$

# ![img](../fig/fig.png)
# 
#  Right     Left     Center     Default
# -------     ------ ----------   -------
#      12     12        12            12
#     123     123       123          123
#       1     1          1             1

# In[13]:


# util.Table([["\sum"]])
from IPython.core.display import display, HTML
display(HTML('<h1>Hello, world!</h1>'))


# In[14]:


import statnlpbook.word_mt as word_mt

word_mt.Alignment("NULL 音楽 が 好き".split(" "),
                  "I like music".split(" "),
                  [(0,0),(1,2),(3,1)])

# "NULL 音楽 が 好き".split(" ")


# In[15]:


import statnlpbook.transition as transition
   
arcs = [
    { 'dir': 'right', 'end': 1, 'label': 'vmod', 'start': 0 }
]
words = [
    { 'tag': 'UH', 'text': 'Hello' },
    { 'tag': 'NNP', 'text': 'Blah.' },    
     
]
transition.DependencyTree(arcs, words)


# In[16]:


transition.DependencyTree(arcs, words)


# In[18]:


import statnlpbook.draw as draw 

draw.edit_svg("blah")

