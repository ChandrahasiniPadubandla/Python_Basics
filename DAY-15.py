#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
def extract_emails(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # Use re.findall to find all matches in the text
    emails = re.findall(email_pattern, text)
    return emails
# Test the function
input_text = 'Contact us at support@example.com and sales@example.org.'
emails = extract_emails(input_text)
print(emails)


# In[ ]:




