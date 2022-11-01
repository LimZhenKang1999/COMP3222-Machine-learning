import pandas as pd
#df = pd.read_table(r'C:\Users\User\Documents\University of Southampton\Year 3\COMP3222 Machine learning\Coursework\mediaeval-2015-trainingset.txt',encoding= 'utf-8')
#df.to_excel(r'C:\Users\User\Documents\University of Southampton\Year 3\COMP3222 Machine learning\Coursework\mediaeval-2015-trainingset.xlsx', index= None)
df = pd.read_excel(r'C:/Users/User/Documents/University of Southampton/Year 3/SEM1/COMP3222 Machine learning/Coursework/training set utp8.xlsx')
#df = pd.read_excel(r'C:/Users/User/Documents/University of Southampton/Year 3/SEM1/COMP3222 Machine learning/Coursework/test set utp8.xlsx')

#True =1 , False || humour = 0 ###############################################
df['encoded_label'] = df['label'].replace(to_replace=r'real',value='1',regex=True)
df['encoded_label'] = df['encoded_label'].replace(to_replace=r'fake',value='0',regex=True)
df['encoded_label'] = df['encoded_label'].replace(to_replace=r'humor',value='0',regex=True)
##############################################################################

#remove all links and URLs##################################################
df['tweetText_URLless'] = df['tweetText'].replace(to_replace=r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*(http: \/\/))*',value='',regex=True)
df['tweetText_URLless'] = df.tweetText_URLless.str.replace("(http: ).*","")
##############################################################################

#extract the emoji############################################################
import emoji
def extract_emojis(s):
  return ' '.join(c for c in s if c in emoji.UNICODE_EMOJI)
df['emojis'] = df['tweetText_URLless'].apply(extract_emojis)
###############################################################################

# removing non-ascii text###################################################
from string import printable
st = set(printable)
df["tweetText_ascii"] = df["tweetText_URLless"].apply(lambda x: ''.join([" " if  i not in  st else i for i in x]))
###############################################################################

#contractions
# https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python
abbr_dict = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have",
"lookin" : "looking",
"gon'" : "going to",
"huracán" : "hurricane"
}
def lookup_words(text):
    words = text.split()
    new_words = [] 
    for word in words:
        if word.lower() in abbr_dict:
            word = abbr_dict[word.lower()]
        new_words.append(word)
        new_text = " ".join(new_words) 
    return new_text 
df['tweetText_proper'] = df['tweetText_ascii'].apply(lookup_words)
#######################################################################

#remove user tagging ######################################################
df['tweetText_proper'] = df['tweetText_proper'].replace(to_replace=r'@\w*',value='',regex=True)
#########################################################################

#convert to lower string######################
df['tweetText_proper'] = df['tweetText_proper'].str.lower()
##############################################

#definations with punc
df['tweetText_proper'] = df['tweetText_proper'].replace("&lt;3", "love")
df['tweetText_proper'] = df['tweetText_proper'].replace("&lt;", "<")
df['tweetText_proper'] = df['tweetText_proper'].replace("&gt;", ">")
df['tweetText_proper'] = df['tweetText_proper'].replace("&amp", "and")
########################################################################

#remove punctuation
import string   
df['tweeetText_proper'] = df['tweetText_proper'].str.replace(r'[^\w\s]+', '')
################################################################

#nlp 
shortform = {
"rt" : "retweet",
"fo" : "fuck off",
"bcs" : "because",
"bc" : "because",
"b/c" :"because",
"nyc": "new york city",
"ny" :"new york",
"vs." : "versus.",
"pls" : "please",  
"wtf": "what the fuck",
"fuc" : "fuck",
"wth" : "what the hell",
"fave": "favourite",
"ct":"connecticut",
"usa":"united states of america",
"happi" : "happy",
"de":  "delaware",
"ma" : "massachusetts",
"md" : "maryland",
"me" : "maine",
"nc" : "north carolina",
"nh" : "new hampshire",
"nj" : "new jersey",
"n.j." : "new jersey",
"goin" : "going",
"gon'" :"are going to",
"bruh" : "brother",
"rip" : "rest in peace",
"omg" : "oh my god",
"pray 4" : "pray for",
"wat" : "what",
"youl" : "you all",
"lil" : "little",
"bro" : "brother",
"pms" : "premenstrual syndrome",
"chillin" : "chilling",
"rp" : "repost",
"usmc" : "united states marine corps",
"d.c." : "District of Columbia",
"lmao" :"laugh my ass off", 
"smh" : "shake my head",
"lol" : "laugh out loud",
"disast" : "disaster",
"pics" : "picture",
"pic" : "picture",
"sbcdr" : "southern baptist disaster relief",
"lmbo" : "laughing my butt off",
"pix" : "picture",
"tho" : "though",
"zomg" : "oh my god",
"deff" : "definitely",
"bs" : "bullshit",
"gtfo" : "get the fuck off",
"boi" : "boy",
"yrs" : "years",
"nypd" : "new york police department",
"dis" : "this",
"w" : "with",
"u" : "you",
"beatiful" : "beautiful",
"instapic" : "instagram picture",
"instamood" : "instagram mood"
}
def definedwords(text):
    words = text.split()
    newwords = [] 
    for word in words:
        if word.lower() in shortform:
            word = shortform[word.lower()]
        newwords.append(word)
        newtext = " ".join(newwords) 
    return newtext 
df['tweetText_definedwords'] = df['tweeetText_proper'].apply(definedwords)
############################################

#split joined words library
# https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
import wordninja
def ninjasplit(x):
        splitsentence = " ".join(w for w in wordninja.split(x))
        return splitsentence                   
df['tweetText_english'] = df['tweetText_definedwords'].apply(ninjasplit)
##################################################################

#remove stopwords#############
import time
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['tweetText_xstopwords'] = df['tweetText_english'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#########################################

#remove integer
df['tweetText_xstopwords'] = df['tweetText_xstopwords'].str.replace('\d+','')
############################################

#lemmetizing####################################
# https://stackoverflow.com/questions/771918/how-do-i-do-word-stemming-or-lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
# Lemmatizing
df['tweetText_lemmatized'] = df['tweetText_xstopwords'].apply(lambda x: lemmatize_sentence(x))
# #####################################################################

#check for english words and only return
import nltk
def removeNonEnglishWordsFunct(x):
    words = set(nltk.corpus.words.words())
    filteredSentence = " ".join(w for w in nltk.wordpunct_tokenize(x)
                                if w.lower() in words or not w.isalpha())
    return filteredSentence
df['tweetText_onlyenglish'] = df['tweetText_lemmatized'].apply(removeNonEnglishWordsFunct)
###############################

#function to remove 1 character as it does not hold any meaning
df['tweetText_onlyenglish']=df['tweetText_onlyenglish'].str.replace(r'\b\w\b','').str.replace(r'\s+', ' ')
##################################################################

#to join the emoji and text together
df['tweetText_cleaned']= df["tweetText_onlyenglish"] + " " + df['emojis']
##############################################################

#the cleaned training and trained data are saved to save time dsuring algorithm running.
#training##############
df.to_excel(r'C:/Users/User/Documents/University of Southampton/Year 3/SEM1/COMP3222 Machine learning/Coursework/df.xlsx', index=None)

# Testing#########################
#df.to_excel(r'C:/Users/User/Documents/University of Southampton/Year 3/SEM1/COMP3222 Machine learning/Coursework/dft.xlsx', index=None)





