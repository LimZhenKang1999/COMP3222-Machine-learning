import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\User\Documents\University of Southampton\Year 3\COMP3222 Machine learning\Coursework\df.xlsx')
dft = pd.read_excel(r'C:\Users\User\Documents\University of Southampton\Year 3\COMP3222 Machine learning\Coursework\dft.xlsx')

#trainingnset#######################
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(ngram_range=(1, 1),max_df=0.5, min_df=5, token_pattern=r'[^\s]+', use_idf = True, stop_words='english')
#tfidfmatrix
tfidf_training = v.fit_transform(df.tweetText_cleaned.values.astype('U'))
x_train = tfidf_training.toarray()
y_train = df.encoded_label
tfidf_testing = v.transform(dft.tweetText_cleaned.values.astype('U'))
x_test = tfidf_testing.toarray()
y_test = dft.encoded_label
###############################################

#head()
print(df.head())
##############

#plotting counts of label#
%matplotlib inline
plt.figure();
df['label'].value_counts().plot(kind='bar')
plt.title('Counts of each label')
###############################################

#plot number of language###########
#detect english#
from langdetect import detect
df['language'] = df['tweetText_definedwords'].apply(lambda x: detect(x))
df['language'].value_counts().plot(kind='bar')
###################################################

#heatmap#
import seaborn as sns
my_df = pd.DataFrame(x_train)
sns.heatmap(my_df.corr())
############################

#Label against usserID
my_df = pd.DataFrame(x_train)
plt.figure()
plt.plot(df.userId, df.encoded_label, ".")
#########################################

#wordcloud#
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
%matplotlib inline
wordcloud = WordCloud().generate(' '.join(df['tweetText_cleaned'])
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
##############################