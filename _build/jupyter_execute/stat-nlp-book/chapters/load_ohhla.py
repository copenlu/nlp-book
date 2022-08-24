#!/usr/bin/env python
# coding: utf-8

# # Loading the OHHLA corpus
# This notebook shows a typical example of data loading and preprocessing necessary for NLP. In this case we are loading a corpus downloaded from the Hip-Hop Lyrics webpage [www.ohhla.com](www.ohhla.com). Our primary goal is to provide a dataset loading function for the [language modelling](todo) chapter in this book.
# 
# We provide the corpus in the `data` directory. As this notebook lives in a sub-directory itself, we access it via `../data`. Before preprocessing all files and provide *generic* loaders it is useful to inspect the format of the files based on a specific example file, and work on the loading process in this context. Here we look at `/data/ohhla/train/www.ohhla.com/anonymous/j_live/SPTA/authentc.jlv.txt`.  

# In[1]:


with open('../data/ohhla/train/www.ohhla.com/anonymous/j_live/allabove/satisfy.jlv.txt.html', 'r') as f:
    # we use read().splitlines() instead of readlines() to skip newline characters
    lines = f.read().splitlines()
    
lines


# We first would like to remove everything outside of the `<pre>` tag, and then remove the meta information.

# In[2]:


def find_lyrics(lines):
    filtered = []
    in_pre = False
    for line in lines:
        if '<pre>' in line:
            in_pre = True
            filtered.append(line.replace("<pre>",""))
        elif '</pre>' in line:
            in_pre = False
            filtered.append(line.replace("</pre>",""))
        elif in_pre:
            filtered.append(line)
    return filtered[6:]
    
lyrics = find_lyrics(lines)
lyrics[:10]


# Finally, we would like to convert the list of lines with newline characters to a single string, as this will be easier to process for our language models. We will also mark lyrical "bars" (lines) using a `BAR` tag to still capture the rhythmical structure in the song.

# In[3]:


string = '[BAR]' + '[/BAR][BAR]'.join(lyrics) + '[/BAR]'
string[:500]


# We are now ready to provide a loading function. 

# In[4]:


def load_song(file_name):
    def load_raw(encoding):
        with open(file_name, 'r',encoding=encoding) as f:
            # we use read().splitlines() instead of readlines() to skip newline characters
            lines = f.read().splitlines()   
            # some files are pure txt files for which we don't need to extract the lyrics 
            lyrics = find_lyrics(lines) if file_name.endswith('html') else lines[5:]
            string = '[BAR]' + '[/BAR][BAR]'.join(lyrics) + '[/BAR]'
            return string
    try:
        return load_raw('utf-8')
    except UnicodeDecodeError:
        try:
            return load_raw('cp1252')
        except UnicodeDecodeError:
            print("Could not load " + file_name)
            return ""

        
    
song = load_song('../data/ohhla/train/www.ohhla.com/anonymous/j_live/allabove/satisfy.jlv.txt.html')
song[:500]


# Now we want to load several files from an album directory. 

# In[5]:


from os import listdir
from os.path import isfile, join

def load_album(path):
    # we filter out directories, and files that don't look like song files in OHHLA.
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and 'txt' in f]
    lyrics = [load_song(f) for f in onlyfiles]
    return lyrics

songs = load_album('../data/ohhla/train/www.ohhla.com/anonymous/j_live/SPTA/')
[len(s) for s in songs]


# We will also make it easy to load several albums. Then, for a few artists we provide short cuts to the album directories we care about. 

# In[6]:


def load_albums(album_paths):
    return [song 
            for path in album_paths 
            for song in load_album(path)]

top_dir = '../data/ohhla/train/www.ohhla.com/anonymous/'
j_live = [
    top_dir + '/j_live/allabove/',
    top_dir + '/j_live/bestpart/'
]
len(load_albums(j_live))


# It will be useful to convert a list of documents into a flat list of tokens. Based on the approach showed in the [tokenisation chapter](todo) we can do this as follows:

# In[7]:


import re
token = re.compile("\[BAR\]|\[/BAR\]|[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
def words(docs):
    return [word 
            for doc in docs 
            for word in token.findall(doc)]
song_words = words(songs)
song_words[:20]


# Finally we provide a function that can load all songs within a top-level directory.

# In[8]:


def load_all_songs(path):
    only_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and 'txt' in f]
    only_paths = [join(path, f) for f in listdir(path) if not isfile(join(path, f))]
    lyrics = [load_song(f) for f in only_files]
    sub_songs = [song for sub_path in only_paths for song in load_all_songs(sub_path)]
    return lyrics + sub_songs

len(load_all_songs("../data/ohhla/train/www.ohhla.com/anonymous/j_live/"))

