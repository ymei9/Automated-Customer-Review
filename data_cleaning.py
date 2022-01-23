# import libiaries and packages
from umap import UMAP
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import os  # apple mac system
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Data ingestion and pre-processing
# change file path to your local path before running
# path_ss = "C:/Users/subhi/OneDrive/Documents/GitHub/BA820-Fall-2021/amazon_reviews_us_Electronics_v1_00.tsv.gz"
path_ym = '/Users/yuxuanmei/Documents/GitHub/BA820_ym/assignments/assignment-01/amazon_reviews_us_Electronics_v1_00.tsv'
# path_cw = "C:/Users/subhi/OneDrive/Documents/GitHub/BA820-Fall-2021/amazon_reviews_us_Electronics_v1_00.tsv.gz"
path_cw = open(os.path.expanduser(
    "/Users/pennyxiong/Downloads/amazon_reviews_us_Electronics_v1_00.tsv"))
<<<<<<< Updated upstream
df = pd.read_csv(path_ym, sep='\t', error_bad_lines=False)
=======
df = pd.read_csv(path_cw, sep='\t', error_bad_lines=False)

# Data ingestion and pre-processing
# change file path to your local path before running
#path_ym = '/Users/yuxuanmei/Documents/GitHub/BA820_ym/assignments/assignment-01/amazon_reviews_us_Electronics_v1_00.tsv'
path_ss = "C:/Users/subhi/OneDrive/Documents/GitHub/BA820-Fall-2021/amazon_reviews_us_Electronics_v1_00.tsv.gz"
#path_cw = "C:/Users/corde/OneDrive/Documents/BA820/TeamProject/amazon_reviews_us_Electronics_v1_00.tsv.gz"
path_cb = "/Users/chinarb/Downloads/amazon_reviews_us_Electronics_v1_00.tsv"
df = pd.read_csv(path_cb, sep='\t', error_bad_lines=False)


# Data ingestion and pre-processing
# change file path to your local path before running
#path_ym = '/Users/yuxuanmei/Documents/GitHub/BA820_ym/assignments/assignment-01/amazon_reviews_us_Electronics_v1_00.tsv'
#path_cw = "C:/Users/subhi/OneDrive/Documents/GitHub/BA820-Fall-2021/amazon_reviews_us_Electronics_v1_00.tsv.gz"
path_yx = open(os.path.expanduser(
    "/Users/pennyxiong/Downloads/amazon_reviews_us_Electronics_v1_00.tsv"))
df = pd.read_csv(path_yx, sep='\t', error_bad_lines=False)
>>>>>>> Stashed changes

df.head(3).T
# Subset for 2015 review data
pd.to_datetime(df['review_date'])

df = df[df['review_date'] > '2014-12-31']
#df = df.sample(n=50000)

# explore different columns (***This could be part of the EDA***)
df.marketplace.unique()  # only US, safe to drop
# 95% represents a good variation, can drop this col
len(df.customer_id.unique())/len(df)
len(df.review_id.unique())/len(df)  # no duplicates, safe to drop
len(df.product_id.unique())/len(df)  # not sure what we can use it for, keep
df.product_parent  # may be useless in NLP
# maybe can help identify potential bias in goods ratings (hard), keep for now
df.product_title
df.product_category.unique()  # only electronics, safe to drop
df.star_rating.unique()  # ratings 1-5
# can't filter based on this feature as the mean is 0.6, safe to drop
df.helpful_votes.describe()
df.vine.describe()  # Amazon's paid review specialist, useful for filter
df.verified_purchase  # filter for verfied purchases only
df.columns
# keep columns of interest
df = df[['review_date', 'product_id', 'verified_purchase', 'vine', 'review_headline',
         'review_body', 'star_rating']]
df.head(3)
df = df.loc[df['review_date'] >= '2015-01-01']
df.shape

# explore different columns (***This could be part of the EDA***)
df.marketplace.unique()  # only US, safe to drop
# 95% represents a good variation, can drop this col
len(df.customer_id.unique())/len(df)
len(df.review_id.unique())/len(df)  # no duplicates, safe to drop
len(df.product_id.unique())/len(df)  # not sure what we can use it for, keep
df.product_parent  # may be useless in NLP
# maybe can help identify potential bias in goods ratings (hard), keep for now
df.product_title
df.product_category.unique()  # only electronics, safe to drop
df.star_rating.unique()  # ratings 1-5
# can't filter based on this feature as the mean is 0.6, safe to drop
df.helpful_votes.describe()
df.total_votes.describe()
df.vine.describe()  # Amazon's paid review specialist, useful for filter
df.verified_purchase  # filter for verfied purchases only
df.columns
# keep columns of interest
df = df[['review_date', 'customer_id', 'product_id', 'verified_purchase', 'vine', 'review_headline',
         'review_body', 'star_rating', 'helpful_votes', 'total_votes']]
df.head(3).T

df = df[df['review_date'] > '2014-12-31']
df.shape
#df = df.sample(n=50000)
df.shape

# check for missing values
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

# check for duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape

# filter for verified purchases
# Given that 92.8% of the reviews are verified purchases, we can drop the non-verified purchases
df['verified_purchase'].describe()
df['verified_purchase'].replace('Y', 1, inplace=True)
df['verified_purchase'].replace('N', 0, inplace=True)
df = df[df['verified_purchase'] == 1]
df.drop(columns='verified_purchase', inplace=True)
df.info()
df.shape

df['vine'].replace('Y', 1, inplace=True)
df['vine'].replace('N', 0, inplace=True)
df.vine.describe()  # Given only .0004% are positive, we can drop this variable
df.drop(columns='vine', inplace=True)
df.shape

# # set index as review date
df.set_index('review_date', inplace=True)

# Due to computation limitations, we will partition the dataset
# based upon top products by review count

df['product_id'].nunique()  # There are 80,841 unique product IDs

top_50 = df.product_id.value_counts()[:50].sum()  # 69,140 reviews
top_100 = df.product_id.value_counts()[:100].sum()  # 102,425
top_200 = df.product_id.value_counts()[:200].sum()  # 143,194
top_500 = df.product_id.value_counts()[:500].sum()  # 214,087
top_1000 = df.product_id.value_counts()[:1000].sum()  # 284,131
print(top_50, top_100, top_200, top_500, top_1000)

top_100_list = df.product_id.value_counts()[:100].index
df = df.loc[df.product_id.isin(top_100_list)]
df.shape

# Explore our reduced data set

# 95% still representing a solid mix of customers leaving reviews, this can be dropped now
len(df.customer_id.unique())/len(df)
df.star_rating.unique()  # ratings 1-5
df.star_rating.describe()  # heavily weighted towards 4 and 5 star-rated items
df.star_rating.value_counts()
# can't filter based on this feature as the mean is 0.6, safe to drop
df.helpful_votes.describe()
df.total_votes.describe()
df.loc[df.total_votes != 0].product_id.value_counts()

df.drop(['helpful_votes', 'total_votes', 'customer_id'], axis=1, inplace=True)
df.shape

# making text data uniform

stop_words = stopwords.words('english')
stemmer = PorterStemmer()

cleaned_data = []
for i in range(len(df['review_body'])):
    reviews = re.sub('[^a-zA-Z0-9]', ' ', df['review_body'].iloc[i])
    reviews = reviews.lower().split()
    reviews = [stemmer.stem(word)
               for word in reviews if (word not in stop_words)]
    reviews = ' '.join(reviews)
    cleaned_data.append(reviews)

len(cleaned_data)  #  102393 english reviews

cleaned_headlines = []
for j in range(len(df['review_headline'])):
    headlines = re.sub('[^a-zA-Z0-9]', ' ', df['review_headline'].iloc[j])
    headlines = headlines.lower().split()
    #headlines=[stemmer.stem(word) for word in headlines if (word not in stop_words)]
    headlines = ' '.join(headlines)
    cleaned_headlines.append(headlines)

len(cleaned_headlines)  # 102393 english headlines

# ##### Tokenization using TFIDF ######

# tokenizing reviews
len(cleaned_data)
tfidf = TfidfVectorizer(max_features=3000)
selected_rows = cleaned_data.copy()
tfidf.fit(selected_rows)
idf = tfidf.transform(selected_rows)
idf = pd.DataFrame(idf.toarray(), columns=tfidf.get_feature_names_out())
idf.shape

# tokenizing review headlines
tfidf.fit(cleaned_headlines)
idf1 = tfidf.transform(cleaned_headlines)
idf1 = pd.DataFrame(idf1.toarray(), columns=tfidf.get_feature_names_out())
idf1.shape

###########DIMENSION REDUCTION##########
# USING PCA
# Check for ideal number of components
# scaler=StandardScaler()
# scaler.fit(idf)
# df1=scaler.fit_transform(idf)

##############################################
# need modifications for PCA
##############################################
PCA = PCA(.85)
pcs = PCA.fit_transform(idf.values)
pcs

# PLOT
plt.figure(figsize=(16, 10))
sns.scatterplot(x="pca-one", y="pca-two",
                palette=sns.color_palette("hls", 10), data=idf)
plt.show()
# Variance Ratio
varexpratio = PCA.explained_variance_ratio_
print("Information explained is:", varexpratio)

# Cumulative view
plt.title("Explained Varaince Ratio by Component")
sns.lineplot(range(1, len(varexpratio)+1), np.cumsum(varexpratio))
plt.axhline(1)
plt.grid()
plt.show()

# Explained variance
varexp1 = Final_PCA.explained_variance_
sns.lineplot(range(1, len(varexp1)+1), varexp1)
plt.axhline(1)
plt.grid()
plt.show()
varexp1.shape

# UMAP
u = UMAP(random_state=820, n_neighbors=10)
u.fit(idf.values)
embeds = u.transform(idf.values)
embeds.shape
# put it in a dataframe
umap_df = pd.DataFrame(embeds, columns=['e1', 'e2'])
umap_df
# Scatter plot
PAL = sns.color_palette("bright", 10)
plt.figure(figsize=(6, 4))
sns.scatterplot(x="e1", y="e2", data=umap_df, legend="full", palette=PAL)
plt.xlabel("e1")
plt.ylabel("e2")
plt.show()
# comps=Final_PCA.components_
# len(comps)
# cols=["PC"+str(i) for i in range(1,len(comps)+1)]
# loadings=pd.DataFrame(comps.T,columns=cols,index=idf.columns)
# comps_df=pcs[:,:5]
# df2=pd.DataFrame(comps_df,columns=['c1','c2','c3','c4'],index=idf.index)
# df2.head(3)
# sns.scatterplot(data=df2,x='c1',y='c2')
# plt.grid()
# plt.show()


######################################################Clustering####################################
# KMeaans
# Clustering
# K-Means
KS = range(2, 15)
# storage
inertia = []
silo = []

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
labs = km.fit_predict(idf)
inertia.append(km.inertia_)
silo.append(silhouette_score(idf, labs))

# plot inertia
sns.lineplot(5, inertia)
plt.title('interia')
plt.axvline(x=5, color='blue')
plt.show()

# plot silo-score
sns.lineplot(5, silo)
plt.title('silhouette score')
plt.axvline(x=7, color='blue')
plt.axvline(x=6, color='red')
plt.show()

# k=5
k5 = KMeans(n_clusters=5)
k5_labs = k5.fit_predict(idf)

# metrics
k5_silo = silhouette_score(idf, k5_labs)
k5_ssamps = silhouette_samples(idf, k5_labs)
np.unique(k5_labs)

skplot.metrics.plot_silhouette(
    idf, k5_labs, title="KMeans - 5", figsize=(5, 5))
plt.show()

forums2 = idf.copy()
forums2['k5_labs'] = k5_labs

k5profile = forums2.groupby('k5_labs').mean()

sc5 = StandardScaler()
k5profile_scaled = sc5.fit_transform(k5profile)

plt.figure(figsize=(15, 5))
pal = sns.color_palette("vlag", as_cmap=True)
sns.heatmap(k5profile_scaled, center=0, cmap=pal,
            xticklabels=k5profile.columns)
for k in KS:
    km = KMeans(k)
    labs = km.fit_predict(idf)
    inertia.append(km.inertia_)
    silo.append(silhouette_score(idf, labs))

# plot inertia
sns.lineplot(KS, inertia)
plt.title('interia')
plt.axvline(x=7, color='blue')
plt.axvline(x=6, color='red')
plt.show()

# plot silo-score
sns.lineplot(KS, silo)
plt.title('silhouette score')
plt.axvline(x=7, color='blue')
plt.axvline(x=6, color='red')
plt.show()
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
