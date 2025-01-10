#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install NLTK (if not already installed)
get_ipython().system('pip install nltk')

# Import required libraries
import nltk

# Download NLTK resources (run once)
nltk.download('punkt')

# Define a sample paragraph
sample_paragraph = """
Python is an amazing programming language! It is widely used in data science, machine learning, and web development.
Tokenization helps break a paragraph into smaller units like sentences and words.
"""

# Tokenize the paragraph into sentences
sentence_tokens = nltk.sent_tokenize(sample_paragraph)

# Tokenize the paragraph into words
word_tokens = nltk.word_tokenize(sample_paragraph)

# Print the results
print("Original Paragraph:")
print(sample_paragraph)
print("\nTokenized Sentences:")
print(sentence_tokens)
print("\nTokenized Words:")
print(word_tokens)


# In[ ]:




