#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
from bs4 import BeautifulSoup

def fetch_webpage_title(url):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the webpage content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract and return the title of the webpage
        return soup.title.string if soup.title else "No title found"
    except requests.exceptions.RequestException as e:
        return f"Error fetching the URL: {e}"

# Test the function with the given URL
url = 'https://example.com'
title = fetch_webpage_title(url)
print(f"Title of the webpage: {title}")


# In[ ]:




