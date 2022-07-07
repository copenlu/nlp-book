#!/usr/bin/env python
# coding: utf-8

# # Welcome
# 
# <table>
#       <tr>
#     <td style="text-align:center">Pepa Atanasova</td>
#      <td style="text-align:center">Dustin Wright</td>
#      <td style="text-align:center">Karolina Stanczak</td>
#   </tr>
#     <tr>
#   <td><img src="../../img/pepa_featured.png" width="250px" /></td>
#   <td><img src="../../img/dustin_featured.jpg" width="210px" /></td>
#   <td><img src="../../img/karolina_featured.png" width="250px" /></td>
#     </tr>
# </table>

# ## Lab Schedule
#     
# - Lab Sessions:
#     - Group 1: Mondays, 10-12 in øv - 1-0-26, Universitetsparken 1, DIKU, Weeks 37-41 + 43-44
#     - Group 2: Fridays, 10-12 in øv - 1-0-26, Universitetsparken 1, DIKU, Weeks 36-41 + 43-44
# 
# We have assigned you to one of two lab session groups based on your answers to the [Getting to Know You survey](https://absalon.instructure.com/courses/52205/quizzes/62668). Please note that this is a preliminary assignment. 

# ## Lab Agenda
# 
# - Introduction to useful tools for the course (Jupyter, Google Colab)
# - Setting up the course environment
# - Tokenization 
# - Introduction to PyTorch
# - Questions about the course project / group work on the  project

# ### Corona Guidelines (subject to changes)
# 
# - **Do not attend** the in-person lectures and lab sessions if you have COVID-19 symptoms
# - **Report** symptoms to the course teachers as well as university immediately
# - Wash and/or **disinfect** your hands before entering the auditorium
# - Keep one metre's **distance** from other students
# - Keep two metre's distance from members of the teaching team when they are presenting
# - Leave the classroom in an **orderly fashion**
# 
# 
# - Full COVID-19 guidelines by SCIENCE: https://kunet.ku.dk/faculty-and-department/science/COVID-19/Pages/default.aspx
# - Poster with summary of guidelines from SCIENCE: https://kunet.ku.dk/faculty-and-department/science/COVID-19/Documents/app%201%20poster%20generel%20guidelines%20200821.pptx

# ## Introduction to Jupyter 
# 
# Jupyter is an open-source web app that combines visualizations, narrative text, mathematical equations, and other rich media in a single document. 
# 
# * Creating new notebook - from the menu File -> New Notebook. Once created, you can rename the notebook by clicking on its name and edit its content by adding (plus button), deleting or editing cells.

# In[1]:


print("Hello, World!"); # this is a code block


# This is a _Markdown block_ where you can write text. 

# You can get help about a method with __(<kbd>Shift</kbd> + <kbd>Tab</kbd>)__
#  
# __(<kbd>Shift</kbd> + <kbd>Enter</kbd>)__ executes the text/code blocks
# 
# While the a code cell is being executed, you'll see a star on the right side of the cell.

# In[2]:


get_ipython().system('python --version # you can also write shell commands in code blocks')


# In[3]:


get_ipython().system('pip3 install nltk # you can also install new libraries')


# We can also create visualizations and save them.

# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');


# In[5]:


fig.savefig('my_figure.png')


# In[6]:


import time
get_ipython().run_line_magic('time', 'time.sleep(10) # this is some jupyter magic')


# All Jupyter magic commands are described [here](https://ipython.readthedocs.io/en/stable/interactive/magics.html).

# In[7]:


# Magic used in the notebooks : 

# automatically re-load imported modules every time before executing the Python code typed
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# automatically include matplotlib plots in the frontend of the notebook and save them with the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Kernels
# Behind every notebook runs a **kernel**. When you run a code cell, that code is executed within the kernel. The kernel is build with a specific Python version. Any output is returned back to the cell to be displayed. The kernel’s state persists over time and between cells — it pertains to the document as a whole and not individual cells.

# In[8]:


import nltk
nltk.download('punkt')

text = "Time flies like an arrow."


# In[9]:


nltk.tokenize.word_tokenize(text)


# ### Checkpoints
# 
# When you create a notebook, a checkpoint file is also saved in a hidden directory called .ipynb_checkpoints. Every time you manually save the notebook (__(<kbd>command</kbd> + <kbd>S</kbd>)__), the checkpoint file updates. Jupyter autosaves your work on occasion, which only updates the .ipynb file but not the checkpoint. You can revert back to the latest checkpoint using File -> Revert to Checkpoint.

# ### WARNING
# 
# Code in Jupyter notebooks can be executed in a non-sequential order. Cells can get deleted.
# Notebooks are dangerous unless you run each cell exactly once and sequentially!
# 
# To restart the state of the notebook you can select:
# **"Kernel -> Restart & Run All"**
# 
# This is especially good to do before sharing your notebook with someone else.

# References:
# - https://www.dataquest.io/blog/jupyter-notebook-tutorial/
# - https://nbviewer.jupyter.org/github/cgpotts/cs224u/blob/master/tutorial_jupyter_notebooks.ipynb

# ---
# ## Introduction to Colab https://colab.research.google.com/

# Colab allows to run notebooks on the Google Cloud with free access to GPUs and TPUs. You can run the same commands reviewed above in Colab as well.
# 
# The notebooks can be shared with other people and you can leave comments and control permissions on it.
# 
# To run the notebook on GPU/TPU you have to select from the menu Runtime->Change Runtime type, which will be None (CPU) by default.
# 
# ### Collaboration options:
# - Share button in the upper right corner.
# - File->Make a Copy creates a copy of the notebook in Drive.
# - File->Save saves the File to Drive and pins a version to the checkpoint and you can later restore version from File->Revision history
# - GitHub - you can open notebooks hosted in GitHub, this will open a new editable version of the notebook and any changes won't override the GitHub version. If you want to save the changes to GitHub select File->Make a copy to GitHub.
# 
# 
# 
# ### Using a custom dataset
# **The code cells below have to be run in a Colab environment!**
# #### Uploading files from your local file system
# files.upload returns a dictionary of the files which were uploaded. The dictionary is keyed by the file name and values are the data which were uploaded.

# In[ ]:


# this code cell has to be run in Colab environment
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# ### files.download will invoke a browser download of the file to your local computer.

# In[ ]:


from google.colab import files

with open('example.txt', 'w') as f:
  f.write('some content')

files.download('example.txt')


# ### Mounting Google Drive locally
# You can mount your Google Drive in the notebook and use all of the files available there.

# In[ ]:


from google.colab import drive
drive.mount('/content/drive') # this will trigger permission prompts


# In[ ]:


with open('/content/drive/My Drive/foo.txt', 'w') as f:
  f.write('Hello Google Drive!')
get_ipython().system('ls /content/drive/My\\ Drive/ | wc -l')


# In[ ]:


drive.flush_and_unmount()
print('All changes made in this colab session should now be visible in Drive.')


# In[ ]:


get_ipython().system('pip3 freeze # contains a lot of pre-installed packages')


# References:
# - Colab guides and examples : https://colab.research.google.com/notebooks/intro.ipynb?hl=en
# - Integration with GitHub: https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC
# - External data handling: https://colab.research.google.com/notebooks/io.ipynb#scrollTo=G5IVmR8S9SeF

# -----
# 
# ## Python tutorials
# 
# * Go thoutgh this [Notebook](../exercises/python_intro.ipynb) with elementary Python syntax.
# 
# * [An Informal Introduction to Python](https://docs.python.org/3/tutorial/introduction.html)
# 
# * [Python for Beginners](https://www.python.org/about/gettingstarted/)
# 
# * [Learn Python in 10 minutes](https://www.stavros.io/tutorials/python/)
# 
# * [LOADS of others](https://www.google.co.uk/search?q=python+tutorial)

# ----- 
# 
# ## stat-nlp-book setup
# 
# ### On your personal computer
# 
# Go to https://github.com/copenlu/stat-nlp-book and follow the readme to set up the stat-nlp-book.
# 
# ### On Microsoft Azure
# 
# If you feel adventurous, and want to set up your own Virtual Machine with stat-nlp-book, follow the [Azure tutorial](../tutorials/azure_tutorial.ipynb)

# ## Additional Info
# 
# ### Docker commands
# * Get a list of the currently running containers <br> 
# <code>docker ps -q</code> <br>
# * Run shell commands in your container by first getting the id of the container with above command and then: <br>
# <code>docker exec -it _container-id_ _command_</code> <br>
# e.g. <code>docker exec -it 8c16b8de4771 python --version</code>
# 
# 
# ### Managing your changes
# 
# There are several ways to keep your changes within the official repo organised. Some of them are:
# * Create your own [fork](https://help.github.com/en/articles/fork-a-repo)
# of the repo. The fork can be [synced](https://help.github.com/en/articles/syncing-a-fork?query=f) with the official course repo when new changes are available. Meanwhile, you can also maintain your changes in your forked repo.
# * Another option is to keep your changes only in a local branch (<code>git checkout -b _your-branch-name_</code>) on your computer. Each time there is a change in the course repo, you can pull the repo and merge the changes in your branch (<code>git merge origin/master</code>)

# ----
# 
# ## Tokenisation

# Tokenisation is an important pre-processing step for NLP models. 
# 
# You can tokenise text at different levels - split to sentences, tokens, subwords, etc. 
# 
# There are a lot of corner cases, language-specific and/or domain-specific cases, which have to handled in different ways.
# 

# In[10]:


import re

text_sentences = "The office is open between 10 a.m. and 1 p.m. every day... Please, be respective of the hours."
re.split('(\.|!|\?)', text_sentences)


# Luckily, there are libraries providing tokenization functionalities that handle most of the cases. Let's look two of the most common libraries for tokenisation:
# 
# ### Spacy

# In[11]:


# download the language models, this can be done for other languages as well
get_ipython().system('python -m spacy download en_core_web_sm # You might have to restart the notebook if the file cannot be found')
get_ipython().system('python -m spacy download fr_core_news_sm')


# In[12]:


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(text_sentences)
list(doc.sents)


# ### NLTK

# In[13]:


import nltk

nltk.tokenize.sent_tokenize(text_sentences)


# #### Word-level tokenization

# In[14]:


text = "Mr. O'Neill thinks that the boys' stories about Chile's capital aren't amusing... Good muffins cost $3.88 in New York. Please buy me two of them!!! Thanks.."
text_tweet = "https://t.co/9z2J3P33Uc Hey @NLPer! This is a #NLProc tweet :-D"
noisy_tweet = "UserAnonym123 What's your timezone_!@# !@#$%^&*()_+ 0123456"

print('Common English tokenization')
print(nltk.word_tokenize(text))
print([token.text for token in nlp(text)])

print('\nTweet tokenization')
print(nltk.word_tokenize(text_tweet))
print([token.text for token in nlp(text_tweet)])

print('\nTokenization of a noisy tweet')
print(nltk.word_tokenize(noisy_tweet))
print([token.text for token in nlp(noisy_tweet)])


# Both libraries perform almost similar for tokenising English common text, so it depends which library you'll use for other features. 
# 
# When it comes to tweets, the nltk default tokenizer performs bad, but NLTK also provides the TweetTokenizer that is suited for tweet tokenization.

# In[15]:


tweet_tokenizer = nltk.tokenize.TweetTokenizer()
print(tweet_tokenizer.tokenize(text_tweet))
print(tweet_tokenizer.tokenize(noisy_tweet))


# As you saw, the above tokenizers tokenize negation contractions like "are", "n't", which is per the the Penn Treebank guidelines. Such tokenization can be useful when building sentiment classification or information extraction. 
# 
# Question:
# - How should we split "I bought a 12-ft boat!"? In 1, 2, or 3 tokens?
# - How should we tokenize "It is a 2850m distance flight.", "The maximum speed on the autobahn is 130km/h."? 
# 
# There is again a rule that units are split from numerical values. Let's test the performance of the tokenizers:

# In[16]:


print('Spacy tokenizer')
print([token.text for token in nlp("I bought a 12-ft boat!")])
print([token.text for token in nlp("It is a 2850m distance flight.")])
print([token.text for token in nlp("The maximum speed on the autobahn is 130km/h.")])

print('\nNLTK simple tokenizer')
print([nltk.tokenize.word_tokenize("I bought a 12-ft boat!")])
print([nltk.tokenize.word_tokenize("It is a 2850m distance flight.")])
print([nltk.tokenize.word_tokenize("The maximum speed on the autobahn is 130km/h.")])


# #### Language dependent tokenization
# 
# While some languages have similar rules for tokenization, other languages are quite different.
# In French, words originally composed of more than one lexical unit that nowadays form a single lexical unit and should thus be recognized as a single token, where an apostrophe should be used to split the word in some cases, but not in all. 
# 
# The following sentence "On nous dit qu’aujourd’hui c’est le cas, encore faudra-t-il l’évaluer.", which means "We are told that this is the case today, it still needs to be assessed." has the following correct tokenization:
# 
# 'On', 'nous', 'dit', 'qu’', 'aujourd’hui', 'c’', 'est', 'le', 'cas', ',', 'encore', 'faudra', '-t-il', 'l’', 'évaluer', '.'
# 
# Explanation:
# - words originally composed of more than one lexical unit that nowadays form a single lexical unit and should thus be recognized as a single token like 'aujourd’hui'
# - qu’aujourd’hui (that today) - today is in contracted form (qu’) and has to be separated from the rest of the word
# - c'est (this is) is ce (C') combined with est and has to be split in two words
# - l’évaluer (evaluate it) is two words, where one is in contracted form and has to be separated
# - faudra-t-il (will it take) - consists of will (faudra), -t is used to prevent two vowels from clashing and should not be tokenized

# In[17]:


print([nltk.tokenize.word_tokenize("On nous dit qu’aujourd’hui c’est le cas, encore faudra-t-il l’évaluer.")])
print([token.text for token in nlp("On nous dit qu’aujourd’hui c’est le cas, encore faudra-t-il l’évaluer.")])


# Let's use the language-specific tokenization:

# In[18]:


nlp_fr = spacy.load("fr_core_news_sm")
print([token.text for token in nlp_fr("On nous dit qu’aujourd’hui c’est le cas, encore faudra-t-il l’évaluer.")])
nltk.tokenize.word_tokenize("On nous dit qu’aujourd’hui c’est le cas, encore faudra-t-il l’évaluer.", language='french')


# #### References:
# - Introduction to Spacy and its features: https://spacy.io/usage/spacy-101
# - NLTK tokenization functionalities: https://www.nltk.org/api/nltk.tokenize.html
# - On rules and different languages: http://ceur-ws.org/Vol-2226/paper9.pdf
# - Why do we need language-specific tokenization: https://stackoverflow.com/questions/17314506/why-do-i-need-a-tokenizer-for-each-language

# ---
# ## Introduction to PyTorch https://pytorch.org/

# In[19]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(123)


# ### Dataset and Bag-of-Words

# In[20]:


data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}


# Typically a common way to read in data in PyTorch is to use one of the two following classes: `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`. Since today we work with a tiny dataset, we omit this part.
# 
# Now we represent the data as Bag-of-Words (BoW) which is a simple way of extracting features from text describing the occurrence of words within a document. Intuitevely, similar documents have similar content.

# In[21]:


# Function to map each word in the vocab to an unique integer
# Indexing the Bag of words vector
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)


# In[22]:


VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


# ### Custom Classifier

# Our classifier inherits from the `nn.Module` class which provides an interface to important methods used for constructing and working with our models. 
# 
# Here, we will implement a custom multi-layer feed forward neural network. In our example, we calculate:
# $$ 
# y = final(nonlinear(linear(BoW))
# $$
# where nonlinear denotes a non-linear function (we use [`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)) and the first and final layer are both linear functions implemented with`nn.Linear`. In practice, we create a `nn.Module` containing the definition of our model architecture in the `__init__` function. The `__forward__` function defines how tensors are processed by our model.

# In[23]:


class BoWClassifier(nn.Module):

    def __init__(self, num_labels, vocab_size, num_hidden = 2):
        # Calls the init function of nn.Module. 
        super(BoWClassifier, self).__init__()

        # Define the parameters that you need.
        self.linear = nn.Linear(vocab_size, num_hidden)
        # non-linearity (here it is also a layer!)
        self.nonlinear = nn.ReLU()
        # final affine transformation
        self.final = nn.Linear(num_hidden, num_labels)

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return self.final(self.nonlinear(self.linear(bow_vec)))


# In[24]:


# Functions to create BoW vectors
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


# The `BoWClassifier` (or any other module you will create) stores knowledge of the models's parameters.

# In[25]:


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# The first output below is A, the second is b.
for param in model.parameters():
    print(param)


# We run the model on the test data before we train to compare with the results from a trained model. 

# In[26]:


with torch.no_grad():
    for text, label in test_data:
        bow_vec = make_bow_vector(text, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])


# ### Training

# We set our loss function to cross entropy which combines `nn.LogSoftmax()` and `nn.NLLLoss()` (negative log likelihood) and calculate gradients with stochastic gradient descent.
# 
# Usually we want to pass over the training data several times by setting a respective number of epochs. Since we have a tiny dataset, we will exaggerate with the number of epochs.

# In[27]:


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# In[28]:


for epoch in range(200):
    for instance, label in data:
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        # Step 3. Run our forward pass.
        log_probs = model(bow_vec)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()


# ### Evaluation

# Let's see if our model can now predict more accurately, if a sentence is written in English or Spanish! 
# 
# Indeed, the log probability for Spanish is much higher for the first sentence, while the log probability for English is much higher for the second sentence in the test data!

# In[31]:


with torch.no_grad():
    for instance, label in test_data:
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)


# In[32]:


print(next(model.parameters())[:, word_to_ix["creo"]])


# ### Useful links
# - PyTorch Tutorials:
# https://pytorch.org/tutorials/index.html
# - Introduction to Pytorch notebook from Stanford: https://nbviewer.jupyter.org/github/cgpotts/cs224u/blob/master/tutorial_pytorch.ipynb

# ----
# ## Course Project 

# * **Group project**, can be completed in a group of up to 3 students
#     * Released 1 September, hand-in 5 November 17:00
#        * joint report, contribution of each student should be stated clearly
#        * code as attachment
#        * individual mark for each group member, based on the quality and quantity of their contributions
#        * submission via Digital Exam
#     * Assignment consists of several parts tied to weekly lecture topics
#     * Finding a group: 
#        * deadline for group forming: **13 September 17:00**
