import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
!pip install wordcloud

comments = pd.read_csv(r"C:\Users\huyan\Desktop\Youtube Project\1-Youtube Text Data Analysis/GBcomments.csv",error_bad_lines=False)
TextBlob("It's more accurate to call it the M+ (1000) be...").sentiment.polarity
comments.dropna(inplace=True)
polarity = []
for i in comments["comment_text"]:
        polarity.append(TextBlob(i).sentiment.polarity)

comments["polarity"] = polarity
comments.head(20)

comments_positive = comments[comments['polarity']==1]
comments_positive.shape
comments_positive.head(20)
stopwords=set(STOPWORDS)
total_comments=''.join(comments_positive['comment_text'])
wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')

comments_negative = comments[comments['polarity']==-1]
comments_negative.shape
total_comments=''.join(comments_negative['comment_text'])
wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')

videos = pd.read_csv(r"C:\Users\huyan\Desktop\Youtube Project\1-Youtube Text Data Analysis/USvideos.csv",error_bad_lines=False)
videos.head()
tags_complete =''.join(videos['tags'])
tags_complete
tags = re.sub('[^a-zA-Z]',' ',tags_complete)
tags = re.sub(' +',' ',tags)
wordcloud= WordCloud(width = 1000, height = 500,stopwords=set(STOPWORDS)).generate(tags)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


sns.regplot(data=videos,x='views',y='likes')
plt.title('Regression plot for views and likes')
df_corr = videos[['views','likes','dislikes']]
sns.heatmap(df_corr.corr(),annot=True)
df_corr.corr()
