#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re

def clean_text(text):

    # Remove special characters using regex and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return cleaned_text

# Test the function
input_text = 'Hello, World! Welcome to NLP 101.'
cleaned_output = clean_text(input_text)
print(f"Original: {input_text}")
print(f"Cleaned: {cleaned_output}")


# In[ ]:




