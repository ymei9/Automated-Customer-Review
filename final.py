# !/usr/bin/env python

"""
Team A1 project_final.py
Final Project Submission
Class: BA 820 - A
Title: Amazon Review Automated Response Generator for Electronics Products
Data Source: https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
Data Download: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz
"""

# import libraries and packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
import time

from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import re
import spacy 
from spacy import cli 
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from bs4 import BeautifulSoup
import unicodedata

####################################################################
######################## DATA PREPROCESSING ########################
####################################################################

# Data ingestion and pre-processing
# change file path to your local path before running
path_ym = '/Users/yuxuanmei/Documents/GitHub/BA820_ym/assignments/assignment-01/amazon_reviews_us_Electronics_v1_00.tsv'
path_ss = "C:/Users/subhi/OneDrive/Documents/GitHub/BA820-Fall-2021/amazon_reviews_us_Electronics_v1_00.tsv.gz"
path_cw = "C:/Users/corde/OneDrive/Documents/BA820/TeamProject/amazon_reviews_us_Electronics_v1_00.tsv.gz"
path_cb = "/Users/chinarb/Downloads/amazon_reviews_us_Electronics_v1_00.tsv"
df = pd.read_csv(path_cw, sep='\t', error_bad_lines=False)
df.shape # 3091024, 15
df.head(3).T

# Subset for 2015 review data
pd.to_datetime(df['review_date'])
df = df.loc[df['review_date'] >='2015-01-01']
df.shape # 809813, 15

# explore different columns (***This could be part of the EDA***)
df.marketplace.unique() # only US, safe to drop
len(df.customer_id.unique())/len(df) # 95% represents a good variation, can drop this col
len(df.review_id.unique())/len(df) # no duplicates, safe to drop
len(df.product_id.unique())/len(df) # not sure what we can use it for, keep
df.product_parent # may be useless in NLP
df.product_title # maybe can help identify potential bias in goods ratings (hard), keep for now
df.product_category.unique() # only electronics, safe to drop
df.star_rating.unique() # ratings 1-5
df.helpful_votes.describe() # can't filter based on this feature as the mean is 0.6, safe to drop 
df.total_votes.describe() # 
df.vine.describe() # Amazon's paid review specialist, useful for filter
df.verified_purchase # filter for verfied purchases only
df.columns
# keep columns of interest
df = df[['review_date', 'customer_id', 'product_id', 'verified_purchase', 'vine', 'review_headline',  
'review_body', 'star_rating', 'helpful_votes', 'total_votes']]
df.head(3).T
df.shape # 809813, 10

# check for missing values
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

# check for duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape # 809736, 10

# filter for verified purchases
df['verified_purchase'].replace('Y', 1, inplace = True)
df['verified_purchase'].replace('N', 0, inplace = True)
df['verified_purchase'].describe() # Given that 92.8% of the reviews are verified purchases, we can drop the non-verified purchases
df = df[df['verified_purchase'] == 1]
df.drop(columns = 'verified_purchase', inplace = True)
df.info()
df.shape # 751494, 9

# explore on vine, decide to drop the column
df['vine'].replace('Y', 1, inplace = True)
df['vine'].replace('N', 0, inplace = True)
df.vine.describe() # Given only .0004% are positive, we can drop this variable
df.drop(columns = 'vine', inplace = True)
df.shape #751494, 8

# # set index as review date 
# df.set_index('review_date', inplace=True)

### Due to computation limitations, we will partition the dataset 
### based upon top products by review count

df['product_id'].nunique() # There are 80,841 unique product IDs

top_50 = df.product_id.value_counts()[:50].sum() # 69,140 reviews
top_100 = df.product_id.value_counts()[:100].sum() # 102,425 
top_200 = df.product_id.value_counts()[:200].sum() # 143,194
top_500 = df.product_id.value_counts()[:500].sum() # 214,087
top_1000 = df.product_id.value_counts()[:1000].sum() # 284,131
print(top_50, top_100, top_200, top_500, top_1000)

top_100_list = df.product_id.value_counts()[:100].index
df = df.loc[df.product_id.isin(top_100_list)]
df.shape # 102425, 8 <- this matches top_100 count 

### Explore our reduced data set 

len(df.customer_id.unique())/len(df) # 95% still representing a solid mix of customers leaving reviews, this can be dropped now
df.star_rating.unique() # ratings 1-5
df.star_rating.describe() # heavily weighted towards 4 and 5 star-rated items
df.star_rating.value_counts()
df.helpful_votes.describe() #  
df.total_votes.describe() # 
df.loc[df.total_votes != 0].product_id.value_counts()

df.drop(['helpful_votes','total_votes','customer_id'], axis=1, inplace=True)
df.shape # 102425, 5

####################################################################
######################## TEXT PREPROCESSING ########################
####################################################################

# Build html stripping tag
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+','\n', stripped_text)
    return stripped_text

# Build function to remove accents and convert to standard ascii
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8','ignore')
    return text

# HANDLING CONTRACTIONS
CONTRACTION_MAP = {
                    "ain't": "is not",
                    "aren't": "are not",
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
                    "he'd": "he would",
                    "he'd've": "he would have",
                    "he'll": "he will",
                    "he'll've": "he he will have",
                    "he's": "he is",
                    "how'd": "how did",
                    "how'd'y": "how do you",
                    "how'll": "how will",
                    "how's": "how is",
                    "I'd": "I would",
                    "I'd've": "I would have",
                    "I'll": "I will",
                    "I'll've": "I will have",
                    "I'm": "I am",
                    "I've": "I have",
                    "i'd": "i would",
                    "i'd've": "i would have",
                    "i'll": "i will",
                    "i'll've": "i will have",
                    "i'm": "i am",
                    "i've": "i have",
                    "isn't": "is not",
                    "it'd": "it would",
                    "it'd've": "it would have",
                    "it'll": "it will",
                    "it'll've": "it will have",
                    "it's": "it is",
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
                    "she'd": "she would",
                    "she'd've": "she would have",
                    "she'll": "she will",
                    "she'll've": "she will have",
                    "she's": "she is",
                    "should've": "should have",
                    "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have",
                    "so's": "so as",
                    "that'd": "that would",
                    "that'd've": "that would have",
                    "that's": "that is",
                    "there'd": "there would",
                    "there'd've": "there would have",
                    "there's": "there is",
                    "they'd": "they would",
                    "they'd've": "they would have",
                    "they'll": "they will",
                    "they'll've": "they will have",
                    "they're": "they are",
                    "they've": "they have",
                    "to've": "to have",
                    "wasn't": "was not",
                    "we'd": "we would",
                    "we'd've": "we would have",
                    "we'll": "we will",
                    "we'll've": "we will have",
                    "we're": "we are",
                    "we've": "we have",
                    "weren't": "were not",
                    "what'll": "what will",
                    "what'll've": "what will have",
                    "what're": "what are",
                    "what's": "what is",
                    "what've": "what have",
                    "when's": "when is",
                    "when've": "when have",
                    "where'd": "where did",
                    "where's": "where is",
                    "where've": "where have",
                    "who'll": "who will",
                    "who'll've": "who will have",
                    "who's": "who is",
                    "who've": "who have",
                    "why's": "why is",
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
                    "you'd": "you would",
                    "you'd've": "you would have",
                    "you'll": "you will",
                    "you'll've": "you will have",
                    "you're": "you are",
                    "you've": "you have"
}

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Function to remove special characters
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text 

# Function to lemmatize words
cli.download('en_core_web_lg')
nlp = spacy.load('en_core_web_lg')
def lemmatize_text(text, model='en_core_web_lg', model_download=False):
    if model_download:
        from spacy import cli
        cli.download(model)
        # nlp = spacy.download(model) 
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

# Function to remove stopwords
def remove_stopwords(text, tokenizer=ToktokTokenizer(),is_lower_case=False):
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# CREATE FUNCTION TO CLEAN CORPUS
def review_processor(corpus, html_stripping=True, contraction_expansion=True, accented_char_removal=True,
                    text_lower_case=True, text_lemmatization=True, special_char_removal=True,
                    stopword_removal=True, remove_digits=True):
                    """
                    Function to systematically clean a text corpus.

                    PARAMETERS:
                    - corpus: list or series of strings that must be cleaned
                    - html_stripping: boolean, remove HTML tags
                    - contraction_expansion: boolean, default is True, convert common contractions into formal verbs ## area for improvement in the contraction map source
                    - accented_char_removal: boolean, default is True, removes special accented characters commonly found in non-English words
                    - text_lower_case: boolean, default is True, converts all text to lower case
                    - text_lemmatization: boolean, default is True, converts all text to lexiconographic derivative
                    - special_char_removal: boolean, default is True, removes all punctuation and other special characters
                    - stopword_removal: boolean, default is True, removes all stopwords from the corpus
                    - remove_digits: boolean, default is True, removes all digits from the corpus

                    RETURNS:
                    - normalized_corpus: cleaned corpus
                    """
                    normalized_corpus = []
                    # normalize each document in the corpus
                    for doc in corpus:
                        # strip HTML
                        if html_stripping:
                            doc = strip_html_tags(doc)
                        # remove accented characters
                        if accented_char_removal:
                            doc = remove_accented_chars(doc)
                        # expand contractions
                        if contraction_expansion:
                            doc = expand_contractions(doc)
                        # lowercase the text
                        if text_lower_case:
                            doc = doc.lower()
                        # remove extra newlines
                        doc = re.sub(r'[\r|\n|\r\n]+','',doc)
                        # lemmatize the text
                        if text_lemmatization:
                            doc = lemmatize_text(doc)
                        # remove special characters and/or digits
                        if special_char_removal:
                            # insert spaces between special characters to isolate them
                            special_char_pattern = re.compile(r'([{.(-)!}])')
                            doc = special_char_pattern.sub(" \\1 ", doc)
                            doc = remove_special_characters(doc, remove_digits=remove_digits)
                        # remove extra whitespace
                        doc = re.sub(' +',' ', doc)
                        # remove stopwords
                        if stopword_removal:
                            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
                        
                        normalized_corpus.append(doc)

                    return normalized_corpus

#### Optimization Testing
# from datetime import datetime
# test_list = []
# then = datetime.now()
# test = df.review_body[:25]
# review_processor(corpus=test, html_stripping=True, contraction_expansion=True, accented_char_removal=True,
#                     text_lower_case=True, text_lemmatization=True, special_char_removal=True,
#                     stopword_removal=True, remove_digits=True)
# now = datetime.now()
# diff = now - then
# test_list.append(diff)
# test_list

## Function compilation testing playground

# sample = df.loc[:,'review_body'][:5]; sample
# review_processor(sample)
# sample = strip_html_tags(sample); sample
# sample = remove_accented_chars(sample); sample
# sample = expand_contractions(sample); sample
# sample = sample.lower(); sample
# sample = re.sub(r'[\r|\n|\r\n]+','',sample); sample
# sample = lemmatize_text(sample); sample
# special_char_pattern = re.compile(r'([{.(-)!}])')
# sample = special_char_pattern.sub(" \\1 ", sample); sample
# sample = remove_special_characters(sample, remove_digits=True); sample
# sample = re.sub(' +',' ', sample); sample
# sample = remove_stopwords(sample, is_lower_case=True); sample

# Create our cleaned review body and headline 
df['review_headline'] = review_processor(df['review_headline'])
df['review_body'] = review_processor(df['review_body'])
df.head().T

####################################################################
################### TOKENIZATION AND CLUSTERING ####################
####################################################################

# Create train and test data
df_train, df_test = train_test_split(df, test_size=0.2)
df_train.shape # 81914,4
df_test.shape # 20479,4

# Document Tokenization and Vectorization
nlp = spacy.load('en_core_web_lrg')
docs = list(nlp.pipe(df_train.review_body))
docs_t = list(nlp.pipe(df_test.review_body))

def doc_vects(docs_list):
    v = np.array([doc.vector for doc in docs_list])
    d = pd.DataFrame(v)
    return d

dv_train = doc_vects(docs)
dv_test = doc_vects(docs_t)
dv_train.shape, dv_test.shape

# Clustering
# KMeans
k = list(range(2,15))
ss= []
for i in k:
    kmeans = KMeans(i)
    kmeans.fit(dv_train)
    k_labs= kmeans.predict(dv_train)
    ss.append(kmeans.inertia_)
ss

## plotting the fit for inertia values in 14 clusters
for a,b in zip(k,ss):
    plt.plot(a,b,'r*')
sns.lineplot(k,ss)
plt.title("Inertia plot for KMeans clustering")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

## plotting silhouette coefficients
sns.lineplot(x=k,y=ss)
plt.title("Silhouette Coefficients clusterwise-KMeans")
plt.xlabel("Clusters")
plt.ylabel("Silhouette Coefficients")
plt.show()

## Create a kmeans model and prediction
def kmeans_fit_predict(clusters, doc_vect, data=None):
    """
    Create a kmeans model, append the predicted labels, and return the model.

    PARAMETERS:
    - clusters: number of kmeans clusters 
    - doc_vect: document vector matrix to fit the model and predict labels
    - data: data frame to append predicted labels on training data; must be same length as doc_vect

    RETURNS:
    - km: kmeans model
    """
    km = KMeans(clusters, random_state=42)
    km.fit(doc_vect)
    if data is not None:
        labs = km.predict(doc_vect)
        clust_lab = "Cluster" + str(clusters)
        data[clust_lab] = labs
        print('Dataframe has been updated.')
    return km

km4 = kmeans_fit_predict(4, dv_train, df_train)
km9 = kmeans_fit_predict(9, dv_train, df_train)

def kmeans_predict(k_model, clusters, test_dv, data=None, train_dv=None):
    """
    Function to create a kmeans cluster prediction from a document vector, add to a data frame, and return the labels

    PARAMETERS:
    - k_model: kmeans model
    - clusters: number of k-clusters to run in the model
    - test_dv: document vector matrix that will be used to produce the predicted labels
    - train_dv: optional, default is None; document vector matrix; if specified, the kmeans model will use this matrix to train the model prior to predicting the labels.
                NOTE: If specified, the k_model will be disregarded and the clusters parameter will be used to tune the kmeans model.
    - data: optional, default is None, if specified the predicted labels will be appended to the specified data frame. Must be of same length as d_v

    RETURNS:
    - labels: list of predicted labels from the model
    """
    if train_dv is not None:
        k_model = KMeans(n_clusters=clusters, random_state=42)
        k_model.fit(train_dv)
    labs = k_model.predict(test_dv)
    if data is not None:
        clust_lab = "Cluster" + str(clusters)
        data[clust_lab] = labs
        print('Dataframe has been updated.')
    return labs

km4_predict = kmeans_predict(km4, 4, dv_test, df_test)
km9_predict = kmeans_predict(km9, 9, dv_test, df_test)

# plot the fit
skplt.metrics.plot_silhouette(df,df['cluster'],figsize=(7,7))
plt.title("KMeans-Clustering")
plt.show()

####################################################################
###################### FEATURE ENGINGEERING ########################
####################################################################

# PCA
pca = PCA(0.9)
pcs = pca.fit_transform(dv_train)

# Plotting EVR
plt.title("Explained Variance Ratio by Component-PCA")
plt.plot(range(pca.n_components_), pca.explained_variance_ratio_)
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.show();

# Most of variance explained in first 20 principal components
## taking the top 20 components
pc20 = pcs[:, :20]
pc20.shape

# TSNE for 2D Visualization
## tsne for visualizing the 20 principal components in a lower dimensional space
tsne = TSNE(n_components=2,verbose=1,perplexity=50,n_iter=1000,learning_rate=200)
tsne.fit(pc20)

## take the embeddings
te = tsne.embedding_
te.shape
## cluster column
cluster_arr = df_train[['Cluster9']].reset_index(drop=True)
## converting to dataframe since we have to visualize in a 2-d frame
tdata = pd.DataFrame(te,columns=['e1','e2'])
tdata = tdata.join(cluster_arr)
tdata.rename({'Cluster9':'cluster_number'}, axis=1, inplace=True)
tdata.head(2)

## visulizing data in different clusters for top 20 PC using tsne
sns.scatterplot(x="e1",y="e2",data=tdata,hue='cluster_number',palette="bright")
plt.title("TSNE 2-d visualization of top 20 principal components")
plt.xlabel("Embedding 1")
plt.ylabel("Embedding 2")
plt.legend(loc='upper right')
plt.show()

####################################################################
###################### CLUSTER TEXT ANALYSIS #######################
####################################################################

cluster_range = range(0,9)
for i in cluster_range:
    for text in df_train.loc[df_train['Cluster9'] == i].review_body.sample(100):
        print(text)
        print("============="*10)
    print("Sample from cluster ", str(i))
    time.sleep(30)

#### 9 CLUSTER INTERPRETATION
# Cluster 0: Good price/quality for the price
# Cluster 1: EXCELLENT PRODUCT, GREAT PRODUCT
# Cluster 2: Blank text or different language
# Cluster 3: GOOD, GREAT
# Cluster 4: Tend to be more advice-style reviews, slightly more detail, might needed additional
# Cluster 5: More negative sentiments, problem, poor quality, slow, might need assistance
# Cluster 6: Works really well, great experience, works the way it is supposed to
# Cluster 7: GREAT, IMPRESSED, GOOD, 
# Cluster 8 : LOVE, LOVE, LOVE

####################################################################
####################### PREDICTIVE MODELING ########################
####################################################################

# Set up the target variables
ytrain = df_train.Cluster9
ytest = df_test.Cluster9

# KNN CLASSISFICATION
scores1 = []
krange1 = range(2, 11)

for i in krange1:
    knn = KNeighborsClassifier(i)
    knn.fit(dv_train, ytrain)
    preds = knn.predict(dv_test)
    preds.shape
    kscore = metrics.accuracy_score(ytest, preds)
    scores1.append(kscore)
len(scores1)
## 4 neighbors are optimal
len(krange1)
krange1
plt.title("K vs scores")
plt.xlabel("Number of nearest neighbors")
plt.ylabel("Accuracy")
sns.lineplot(x=krange1, y=scores1)
plt.show()
preds.shape

# RANDOM FOREST
model = RandomForestClassifier(random_state=0,n_jobs=1,max_depth=5,n_estimators=100,oob_score=True)
model.fit(dv_train, ytrain)
prediction = model.predict(dv_test)
acc_score=metrics.accuracy_score(ytest, prediction)
print("Accuracy obtained from random forest is", round(acc_score*100),"%") # 82% accuracy

## SUPPORT VECTOR CLASSIFICATION
svc = SVC(random_state=42)
svc.fit(dv_train, ytrain)
svc_pred = svc.predict(dv_test)
svc_acc = accuracy_score(ytest, svc_pred)
print("Support Vector Model accuracy prediction is :", svc_acc) # 98.4% accuracy!?

### Visualize results via confusion matrix
cm = confusion_matrix(ytest, svc_pred)
sns.heatmap(cm, 
            cmap='Blues')
plt.show()

# GRADIENT BOOSTING CLASSIFICATION
gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, random_state=42)
gbc.fit(dv_train, ytrain)
gbc_pred = gbc.predict(dv_test)
gbc_acc = accuracy_score(ytest, gbc_pred)
print("Gradient Boosting Model prediction accuracy is :", gbc_acc)

# LOGISTIC REGRESSION 
model_logistic=LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(dv_train,ytrain)
prediction_logistic = model.predict(dv_test)
acc_score_logistic=metrics.accuracy_score(ytest, prediction_logistic)
print("Accuracy from logistic regression is",round(acc_score_logistic*100),"%") # 82%

####################################################################
################# RECOMMENDED RESPONSE DICTIONARY ##################
####################################################################

def response_generator(cluster):
    """"
    Returns a string response based on the cluster assignment of the review.

    PARAMETERS:
    - cluster: number, the predicted cluster label

    RETURNS:
    - a string with a recommended response to the review

    """
    RESPONSES = {
        '0':'Thank you so much for leaving a review. We agree that the quality of this product surpasses all other rivals!',
        '1':'Thank you so much for leaving a review. We appreciate the fact that you recognize the greatness of this product!',
        '2':None,
        '3':'Thank you so much for leaving a review. Are there any things that you would recommend to help make this product even better?',
        '4':'Thank you so much for leaving a review. You have provided some really great advice on how to use our product!',
        '5':'Uh-Oh! It sounds like you might be having some bad experiences with our product. Is there anything that we can do to make this better? Please reach out to us.',
        '6':'Thank you so much for leaving review. We are glad to hear the product is working as you expected!',
        '7':'Thank you so much for leaving a review. We are glad we have made an impression on you!',
        '8':'Thank you so much for leaving a review. We are so glad you love the product as much as we do!',       
    }
    
    response = RESPONSES[str(cluster)] 
    return response

####################################################################
##################### BUILD COMPLETE PIPELINE ######################
####################################################################

# First, we'll need to take our body of data through the document cleaner process
## Function: review_processor()

# Next, we'll want to convert corpus into a document vector
def doc_creation(doc_body,model='en_core_web_lg',model_download=False):
    """
    Returns a dataframe of document vectors with the shape of (n, 300) where n is equal to the length of the corpus.

    PARAMETERS:
    - doc_body: string, corpus that needs to be transformed and vectorized
    - model: string, eligible spacy model for language tokenization
    - model_download: boolean, states whether spacy language model needs to be downloaded

    RETURNS:
    - dataframe of shape (n, 300)
    
    NOTE: Installation of spacy is required.  
    """
    if model_download:
        from spacy import cli
        cli.download(model)
    nlp = spacy.load(model)
    docs_list = list(nlp.pipe(doc_body))
    v = np.array([doc.vector for doc in docs_list])
    d = pd.DataFrame(v)
    return d

# Then, we'll run the document vector through the model prediction
def svc_predictor(model, test_doc, train_model=False, train_doc=None):
    """
    Runs a classification model to produce a list of predicted labels for a document vector.

    PARAMETERS:
    - model: support vector classification model
    - test_doc: document vector matrix that will be used to predict the labels
    - train_model: boolean, allows users to fit a new model on a training dataset 
    - train_doc: optional, document vector matrix used to train the model if train_model is set to True.

    RETURNS:
    - list of labels

    NOTE: If train_model is set to True and train_doc is left blank, the test_doc will be used to train the model and to predict the returned labels.
    """
    if train_model:
        if train_doc is None:
            model.fit(test_doc)
        else:
            model.fit(train_doc)
    labels = model.predict(test_doc)
    return labels

# And use the response generator to automate our response
## Function: response_generator()

# Tie it all together

def automated_response_generator(reviews):
    """
    Function that takes a list of strings and returns an automated response based upon the text.

    PARAMETERS:
    - reviews: list of strings representing the reviews to be responded
    
    RETURNS:
    - responses: list, text response to each review
    """
    # Analyze the review

    ## Clean the review data
    reviews = review_processor(reviews)

    ## Tokenize and Vectorize the review data
    vector = doc_creation(reviews)

    ## Predict the cluster label for each review
    labels = svc_predictor(svc, vector)

    ## Determine appropriate response for each 
    responses = [response_generator(i) for i in labels]

    # return responses
    return responses

####################################################################
############################ PLAYGROUND ############################
####################################################################

test = ['I have really enjoyed this product. It has been so useful in helping to revitalize my life!']
test_ = automated_response_generator(test)
test_

# # VISUALIZATION -SVM
# support_vector_indices = svc.support_
# print(support_vector_indices)
# support_vectors = svc.support_vectors_
# support_vectors
# dv_train.columns
# v = np.array([doc.vector for doc in docs])

# # Visualize support vectors
# plt.scatter(v[:,0], v[:,1])
# plt.scatter(support_vectors[:,0], support_vectors[:,1], color='blue')
# plt.title('Linearly separable data with support vectors')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()

# # ROC CURVE SVM and KNN
# fpr, tpr, thresholds = roc_curve(ytest, svc_pred, pos_label=1)
# fpr_knn, tpr_knn, thresh_knn = roc_curve(ytest,preds, pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)
# roc_auc_knn=metrics.auc(fpr_knn,tpr_knn)
# display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
# display_knn = metrics.RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn, roc_auc=roc_auc_knn,estimator_name='example estimator')
# display.plot()
# display_knn.plot()
# plt.xlabel("False positive")
# plt.ylabel("True positive")
# plt.title("ROC CURVE")
# plt.legend(loc="upper left")
# plt.show()