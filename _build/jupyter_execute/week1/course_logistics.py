#!/usr/bin/env python
# coding: utf-8

# # Course Logistics

# # NDAK18000U Overview
# content at [https://www.copenlu.com/nlp-book](https://www.copenlu.com/nlp-book)

# ### NDAK18000U Details
# 
# - **Course Responsible**: [Isabelle Augenstein](http://isabelleaugenstein.github.io/)
# - **Lecturer**: [Daniel Hershcovich](https://danielhers.github.io/)
# - **Teaching Assistants**: [Karolina Stanczak](https://kstanczak.github.io/), [Erik Arakelyan](https://scholar.google.co.uk/citations?user=63BfrxMAAAAJ), [Nadav Borenstein](https://www.linkedin.com/in/nadavbor/), [Ruixiang Cui](https://ruixiangcui.github.io/)

# ### NDAK18000U Schedule
# 
# - Lectures:  
#     -  Tuesdays, 13-15 in Lille UP1 (04-1-22), Universitetsparken 1, Weeks 36-41 + 43-44
#     
# - Lab Sessions:
#     - Group 1: Mondays, 10-12 in the old library (4-0-17), Universitetsparken 1, Weeks 37-41 + 43-44
#     - Group 2: Fridays, 10-12 in the old library (4-0-17), Universitetsparken 1, Weeks 36-41 + 43-44
# 
# We will assign you to one of two lab session groups based on your answers to the [Getting to Know You survey](https://absalon.instructure.com/courses/61339/quizzes/75089) (if you have not filled it in yet, do it asap). You will receive an announcement about this before the first lab session.

# ### NDAK18000U Syllabus (subject to small changes)
# 
#   * Week 36 (5-9 Sept)
#       * Lecture (Tuesday): Course Logistics ([slides](course_logistics.ipynb)), Introduction to NLP ([slides](intro_short.ipynb)), Tokenisation & Sentence Splitting ([notes](tokenization.ipynb), [slides](tokenization_slides.ipynb), [exercises](tokenization.ipynb)), Text Classification ([slides](doc_classify_slides_short.ipynb))
#       * Lab (09.09 & 12.09): Jupyter notebook setup, introduction to [Colab](https://colab.research.google.com/). Introduction to [PyTorch](https://pytorch.org/tutorials/). Project group arrangements. Questions about the course project. ([lab](lab_1.ipynb))
#   * Week 37 (12-16 Sept)
#       * Lecture (Tuesday): Introduction to Representation Learning ([slides](../week2/dl-representations_simple.ipynb)), Language Modelling (partially) ([slides](../week2/language_models_slides.ipynb))
#       * Lab (16.09 & 19.09): Recurrent Neural Networks and word representations. Project help. ([lab](../week2/lab_2.ipynb))

# ### NDAK18000U Syllabus (subject to small changes)
# 
#   * Week 38 (19-23 Sept)
#       * Lecture (Tuesday): Language Modelling (rest) ([slides](../week3/language_models_slides.ipynb)), Recurrent Neural Networks ([slides](../week3/rnn_slides_ucph.ipynb)), Contextualised Word Representations ([slides](../week3/dl-representations_contextual.ipynb))
#       * Lab (23.09 & 28.09): Language Models with [Transformers](https://huggingface.co/course/chapter1) and RNNs. Project help. ([lab](../week3/lab_3.ipynb))
#   * Week 39 (26-30 Sept)
#       * Lecture (Tuesday): Attention ([slides](../week4/attention_slides2.ipynb)), Interpretability ([slides](../week4/interpretability_slides.ipynb))
#       * Lab (30.09 & 03.10): Error analysis and explainability. Project help. ([lab](../week4/lab_4.ipynb))
#   * Week 40 (3-7 Oct)
#       * Lecture (Tuesday): Sequence Labelling ([slides](../week5/sequence_labeling_slides.ipynb), [notes](../week5/sequence_labeling.ipynb))
#       * Lab (07.10 & 10.10): Sequence labelling. Beam search. Project help. ([lab](../week4/lab_5.ipynb))

# ### NDAK18000U Syllabus (subject to small changes)
# 
#   * Week 41 (10-14 Oct)
#       * Lecture (Tuesday): Information Extraction ([slides](../week6/information_extraction_slides.ipynb)), Question Answering ([slides](../week6/question_answering_slides.ipynb))
#       * Lab (14.10 & 24.10): In-depth look at Transformers and Multilingual QA. Project help. ([lab](../week6/lab_6.ipynb))
#   * Week 43 (24-28 Oct)
#       * Lecture (Tuesday): Machine Translation ([slides](../week7/nmt_slides_active.ipynb)), Cross-lingual Transfer Learning ([slides](../week7/xling_transfer_learning_slides.ipynb))
#       * Lab (28.10 & 31.10): Project help.
#   * Week 44 (31 Oct-4 Nov)
#       * Lecture (Tuesday): Dependency Parsing ([slides](../week8/dependency_parsing_slides_active.ipynb))
#       * Lab (04.11): Project help.

# ### Course Requirements
# * Familiarity with machine learning (probability theory, linear algebra, classification)
# * Knowledge of programming (Python)
# * No prior knowledge of natural language processing or linguistics is required
# 
# Relevant machine learning competencies can be obtained through one of the following courses: 
# * NDAK15007U Machine Learning (ML) 
# * NDAK16003U Introduction to Data Science (IDS) 
# * NDAB18000U Data Science
# * NDAB18003U Elements of Machine Learning 
# * Machine Learning, Coursera
# 
# See also the [course description](https://kurser.ku.dk/course/ndak18000u/2022-2023)

# ### About You: previously taken courses related to NLP?
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['Yes, NDAK14004U Web Science (WS)', 'Yes, NDAK15005U Information Retrieval (IR)', 'Yes, other course', 'No']
sizes = [8, 9, 11, 24] 

plt.rcParams['figure.dpi'] = 400

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
# ax.set_title('My title')

plt.show()


# ### About You: previously taken courses in Machine Learning?
# 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['Yes, NDAK15007U Machine Learning (ML)', 'Yes, NDAB18003U Elements of Machine Learning', 'Yes, NDAK15018U Large-Scale Data Analysis (LSDA)', 'Yes, NDAK16003U Introduction to Data Science (IDS)', 'Yes, NDAB18000U Data Science', 'Yes, other course']
sizes = [22, 3, 1, 7, 5, 23] 

plt.rcParams['figure.dpi'] = 400

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
# ax.set_title('My title')

plt.show()


# ### About You: experience with using neural network software libraries?
# 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['PyTorch', 'Tensorflow', 'Keras', 'No', 'Other']
sizes = [20, 18, 17, 4, 13] 

plt.rcParams['figure.dpi'] = 200

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
# ax.set_title('My title')

plt.show()


# ### About You: degree are you enrolled in
# 

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['MSc Computer Science', 'BSc Datalogi', 'BSc Machine Learning og Datavidenskab', 'Other MSc program', 'Other BSc program', 'Other']
sizes = [19, 2, 4, 17, 1, 1]

plt.rcParams['figure.dpi'] = 300

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
# ax.set_title('My title')

plt.show()


# ### About You: what you want to get out of this course
# 

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['getting more thorough introduction to NLP', 
          'learning about an interesting application of ML', 
          'wanting to understand what all the hype is about', 
          'preparing for thesis project', 
          'improving job prospects', 
          'practical skills for processing text',
          'course credits']
sizes = [14, 8, 3, 7, 6, 6, 2] 

plt.rcParams['figure.dpi'] = 400

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
# ax.set_title('My title')

plt.show()


# ### About You: what you want to get out of the lab sessions
# 

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
get_ipython().run_line_magic('matplotlib', 'inline')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['step-by-step tutorials for practical topics related to the course', 'to work on the course project with my group', 'an opportunity to ask questions about the course or assignment', 'other']
sizes = [27, 32, 34, 1] 

plt.rcParams['figure.dpi'] = 400

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
# ax.set_title('My title')

plt.show()


# ### Course Materials
# * We will be using the [stat-nlp-book](../overview.ipynb) project. 
# * Contains **interactive** [jupyter](http://jupyter.org/) notes and slides
#     * View statically [here](https://nbviewer.jupyter.org/github/copenlu/stat-nlp-book/blob/master/overview.ipynb)
#     * Use interactively via install, see [github repo](https://github.com/copenlu/stat-nlp-book) instructions  
# * References to other material are given in context.
# * This is work in progress.
#     * Course materials are adapted from a [course that Isabelle co-taught at UCL](https://github.com/uclmr/stat-nlp-book) (course organiser: [Sebastian Riedel](http://www.riedelcastro.org/))
#     * Use `git pull` regularly for updates
#     * *Watch* for updates
#     * Please contribute by adding issues on github when you see errors
# * For assignment hand-in, announcements, discussion forum, check [Absalon](https://absalon.instructure.com/courses/61339)

# ### Teaching Methods
# * Course combines
#     * Traditional lectures
#     * Hands-on exercises
#     * Group work
# * Occasional small exercises during lectures, so bring your laptop
# * You are expected to read some background material for each lecture
#     * This is such that everyone is on the same page
#     * And so that there is more time for exercises and discussions in lectures
# * The background material will be made available a week before each lecture
#     * No reading material for the first lecture

# ### Lecture Preparation
# 
# * Read Background Material
# * Go through lecture notes, play with code (optional)
# * Do exercises (optional)

# ### Assessment Methods
# 
# * **Group project (50%)**, can be completed in a group of up to 3 students
#     * Released 1 September, **hand-in 4 November 17:00**
#        * joint report, contribution of each student should be stated clearly
#        * code as attachment
#        * individual mark for each group member, based on the quality and quantity of their contributions
#        * submission via Digital Exam
#     * Assignment consists of several parts tied to weekly lecture topics

# ### Assessment Methods
# 
# * **Group project (50%)**, can be completed in a group of up to 3 students
#     * Finding a group: 
#        * Deadline for group forming: **12 September 17:00**
#        * Offer to help you find a group -- fill in the "Getting to know you" quiz by the end of *first lecture day,* **6 September 2022 17:00**.
#        * If you choose this option, you will be informed of your assigned group on **7 September 2022** and can still change groups by asking other students to swap groups (it's your responsibility to arrange this though).
#        * Otherwise, we assume you will find a group by yourself in the first course week, e.g. by coordinating with other students in the lab session

# ### Assessment Methods
# 
# * **Take-home exam (50%)**, to be completed individually
#     * Released 7 November 17:00, **hand-in 9 November 17:00**
#     * Exam is timed: 1.5 hours
#     * Takes place on Absalon
#     * When completed, submit your answers on Absalon
#     * Theoretical exam, covering the whole course curriculum

# ### Late Hand-In
# 
# * Late hand-ins **cannot be accepted**
# * Exceptions can be made in rare cases, e.g. due to illness with doctor's notice
#     * Get in touch with course responsible at least one working day in advance

# ### Plagiarism
# 
# * Don't do it
# * Don't enable it
# * Check [rules and consequences](https://student-ambassador.ku.dk/rights/avoid-plagiarism/) if unclear

# ### Docker
# 
# * The book and tutorials run in a [docker](https://www.docker.com/) container
# * Container comes with all dependencies pre-installed
# * You can install it on your machine or on Azure/AWS machines
# * We provide no support for non-docker installations
# * We recommend you use this container for your assignment
#    * Contains all core software packages for solving the assignment
#    * You may use additional packages if needed

# ### Python
# 
# * Lectures, lab exercises and assignments focus on **Python**
# * Python is a leading language for data science, machine learning etc., with many relevant libraries
# * We expect you to know Python, or be willing to learn it **on your own**
# * Labs and assignments focus on development within [jupyter notebooks](http://jupyter.org/)

# ### Lab Sessions
# 
# * Some lab sessions are tutorial-style (to introduce you to practical aspects of the course)
# * Other lab sessions are open-topic. You can use them as an opportunity to:
#    * ask the TAs clarifying questions about the lectures and/or assignment
#    * ask the TAs for informal feedback on your assignment so far
#    * work on your assignment with your group

# ### Discussion Forum
# 
# * Our Absalon page has a [**discussion forum**](https://absalon.instructure.com/courses/61339/discussion_topics).
# * Please post questions there (instead of sending private emails) 
# * We give low priority to **questions already answered** in previous lectures, tutorials and posts, 
#     * and to **pure programming related issues**
# * We expect you to **search online** for answers to your questions before you contact us.
# * You are highly encouraged to participate and **help each other** on the forum. 
# * The teaching team will check the discussion forum regularly **within normal working hours**
#     * do not expect answers late in the evenings and on weekends
#     * **start working on your assignments early**
#     * come to the lab sessions and ask questions there

# ### Copenhagen NLP
# 
# * Research Section, UCPH Computer Science Department
# * Faculty members: Isabelle Augenstein (head of section), Daniel Hershcovich, Desmond Elliott, Anders Søgaard
# * Official webpage: https://di.ku.dk/english/research/nlp/
# * List of group members: http://copenlu.github.io ; http://coastalcph.github.io/; https://elliottd.github.io/people.html
# * Twitter: 
#     * @copenlu https://twitter.com/CopeNLU
#     * @coastalcph https://twitter.com/coastalcph
# * Always looking for strong MSc students
# * PhD positions available dependent on funding
