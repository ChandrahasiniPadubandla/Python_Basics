#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install wordcloud matplotlib


# In[2]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, output_filename):

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Save the WordCloud as an image
    wordcloud.to_file(output_filename)

    # Optionally display the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Input text
text = 'data science machine learning artificial intelligence'

# Generate and save the WordCloud
output_filename = 'wordcloud.png'
generate_wordcloud(text, output_filename)

print(f"WordCloud saved as {output_filename}")


# In[ ]:




