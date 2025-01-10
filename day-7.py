#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install gensim nltk')


# In[4]:


from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS, stem_text
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import os

# Ensure nltk data is downloaded
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to preprocess text
def preprocess_text(text):
    # Tokenization using gensim's simple_preprocess
    tokens = simple_preprocess(text, deacc=True)  # deacc=True removes punctuations
    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return lemmatized_tokens

# Read the sample text file
file_path = "sample_text.txt"  # Replace with your file path
if not os.path.exists(file_path):
    with open(file_path, "w") as f:
        f.write("This is a sample text file to demonstrate preprocessing. It includes tokenization, stemming, and lemmatization.")

with open(file_path, "r") as file:
    text = file.read()

# Preprocess the text
processed_tokens = preprocess_text(text)

# Output the results
print("Original Text:")
print(text)
print("\nProcessed Tokens:")
print(processed_tokens)


# In[ ]:




