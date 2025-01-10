#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install nltk spacy')
get_ipython().system('python -m spacy download en_core_web_sm')



# In[4]:


import nltk
from nltk.corpus import stopwords
import spacy

# Download the NLTK stopwords dataset
nltk.download('stopwords')

# Load SpaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text_lower = text.lower()
    print("Text after converting to lowercase:", text_lower)

    # Tokenize the text using SpaCy
    doc = nlp(text_lower)

    # Remove stopwords using NLTK
    nltk_stopwords = set(stopwords.words('english'))
    filtered_tokens = [token.text for token in doc if token.text not in nltk_stopwords]
    print("Text after removing stopwords:", " ".join(filtered_tokens))

    return filtered_tokens

# Example text
text = "NLTK and SpaCy are popular libraries for natural language processing in Python."

# Process the text
processed_text = preprocess_text(text)


# In[ ]:




