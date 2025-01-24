#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install spacy')



# In[6]:


import spacy

def pos_tagging(sentence):

    # Load the English language model from SpaCy
    nlp = spacy.load("en_core_web_sm")

    # Process the sentence
    doc = nlp(sentence)

    # Print each token with its part-of-speech tag
    print("Token\tPOS Tag")
    print("-" * 20)
    for token in doc:
        print(f"{token.text}\t{token.pos_}")

# Input sentence
sentence = 'NLP is amazing and fun to learn.'

# Perform POS tagging
pos_tagging(sentence)


# In[ ]:




