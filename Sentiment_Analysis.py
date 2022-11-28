# importing necessary packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
# train dataset used for our analysis
train = pd.read_csv('https://github.com/upendra2001/Sentiment_Analysis/blob/main/train.csv')
# making a copy of dataset
train_original=train.copy()

train.shape

train_original
test = pd.read_csv('https://github.com/upendra2001/Sentiment_Analysis/blob/main/test.csv')
test_original=test.copy()
test.shape
test_original
# We combine Train and Test datasets for pre-processing stage
combine = train.append(test,ignore_index=True,sort=True)
combine.head()
combine.tail()

#Removing Twitter Handles (@user)
# a user-defined function to remove unwanted text patterns from the tweets.
# It takes two arguments, one is the original string of text and the other is the pattern of text that we want to remove from the string.
# The function returns the same input string but without the given pattern.
# We will use this function to remove the pattern ‘@user’ from all the tweets in our data.

def remove_pattern(text, pattern):
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern, text)

    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i, "", text)

    return text

combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")

combine.head()

# Removing Punctuations, Numbers, and Special Characters
# Punctuations, numbers and special characters do not help much. It is better to remove them from the text just as we removed the twitter handles. 
# Here we will replace everything except characters and hashtags with spaces.

combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

combine.head(10)


combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

combine.head(10)

#Tokenization

tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())
tokenized_tweet.head()

#Stemming

from nltk import PorterStemmer

ps = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()

#Now let’s stitch these tokens back together.

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combine['Tidy_Tweets'] = tokenized_tweet
combine.head()

#Visualization from Tweets
# Wordcloud
# A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes.
# Importing Packages necessary for generating a WordCloud
# from wordcloud import WordCloud,ImageColorGenerator

from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests

# Store all the words from the dataset which are non-racist/sexist
all_words_positive = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==0])

# We can see most of the words are positive or neutral. With happy, smile, and love being the most frequent ones. Hence, most of the frequent words are compatible with the sentiment which is non racist/sexists tweets.

# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_positive)

# Size of the image generated 
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()




# Store all the words from the dataset which are racist/sexist

all_words_negative = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==1])

# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_negative)

# Size of the image generated 
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="gaussian")

plt.axis('off')
plt.show()
