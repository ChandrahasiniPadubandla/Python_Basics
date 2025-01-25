#!/usr/bin/env python
# coding: utf-8

# In[4]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text, output_file):
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Save the WordCloud to an image file
    wordcloud.to_file(output_file)
    
    # Display the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Input text
text = 'data science machine learning artificial intelligence'

# Output file path
output_file = 'wordcloud.png'

# Generate and save the WordCloud
generate_wordcloud(text, output_file)

print(f"WordCloud saved as {output_file}")


# In[ ]:




