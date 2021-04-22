from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from nlp4all.models import Tweet, TweetTagCategory, Project, D2VModel
from nlp4all import db
from nlp4all.utils import assign_colors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


"""
### Doc2Vec

class D2VModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(50))
    model = db.Column(db.PickleType)

    def save(self, gensim_model):
        # /!\ Erases an possible already saved model
        pickled_model = pickle.dumps(gensim_model)
        self.model = pickled_model

    def load(self):
        return pickle.loads(self.model)

# parameters
# Arbitrary. Could be changed
vector_size = 50
min_count = 2
epochs = 50


# Initializing the model
d2v_model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)


# creating the database
tweets = Tweet.query.all()
train_corpus = []

for tweet in tweets:
    train_corpus.append(TaggedDocument(tweet.text, [tweet.id]))


# Create the vocabulary and train the model
d2v_model.build_vocab(train_corpus)
d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)


# save d2v_model into the database
d2v = D2VModel(id=1, description="trained on the entire corpus")
d2v.save(d2v_model)
"""


### Visualization functions


def reduce_with_PCA(data, n_components=2, verbose=False):
    # data should be a list of numpy arrays, or a pandas dataframe, or something similar
    # row = a tweet ; column = a dimension of the word space
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data)
    if verbose:
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return pca_results


# probably takes a few minutes -> do it in the background
def reduce_with_TSNE(data, n_components=2, perplexity=30, n_iter=1000):
    # if too much dimensions, use PCA to reduce first
    if len(data[0]) > 50:
        data = reduce_with_PCA(data, n_components=50)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    tsne_reduced = tsne.fit_transform(data)
    return tsne_reduced


def projection_2D(data, wv1, wv2):
    # data is a list of (word or doc) vectors
    # wv1 and wv2 are the two word vectors the projection is on
    projected_vectors = []
    for vec in data:
        x1 = cosine_similarity(vec, wv1)
        x2 = cosine_similarity(vec, wv2)
        projected_vectors.append((x1,x2))
    return projected_vectors


def display_2D(data, tweets_cats, nb_cats, title=None, opacity=0.3):
    # data is a list of projected word/doc vectors on 2D
    # tweets_cats is a list of categories associated to each tweet in the same order
    # nb_cats is the number of different categories present is the data (information already conatained in tweets_cats)
    plt.figure()
    if title:
        plt.title(title)
    sns.scatterplot(
        x=data[:,0], y=data[:,1],
        hue=tweets_cats,
        palette=sns.color_palette("hls", nb_cats),
        legend="full",
        alpha=opacity)


def display_from_cats(model, cats_ids, opacity=0.3):
    # model is a gensim model
    # cats_ids is a list of the categories displayed by the algorithm
    tweets = Tweet.query.filter(Tweet.category.in_(cats_ids)).all()  # tweets within the tweet_cats
    tweets_cats = [tweet.category for tweet in tweets]  # their categories
    dv = [model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]  # their word vectors
    #dv = [model.dv[tweet.id] for tweet in tweets]  # possible only if the tweets have been in the training set

    pca_results = reduce_with_PCA(dv)
    tsne_results = reduce_with_TSNE(dv)

    display_2D(pca_results, tweets_cats, len(cats_ids), title="PCA graph", opacity=opacity)
    display_2D(tsne_results, tweets_cats, len(cats_ids), title='t-SNE graph', opacity=opacity)


"""# get doc_id and doc vector
for key in d2v_model.dv.index_to_key:
    print(key, d2v_model.dv[key])"""



### Visualization

## proto visualisation. Deprecated
'''# getting the project categories
project = Project.query.first()
proj_cats = project.categories
colors = assign_colors(proj_cats)
print("proj_cats", proj_cats)
print("colors", colors)
proj_tweets = project.tweets


# getting all the tweets + cats
tweets = Tweet.query.all()
print(tweets == proj_tweets)

tweets_cats = [tweet.category for tweet in tweets]
print("tweets_cats", tweets_cats)
cats = TweetTagCategory.query.all()
print("all cats", cats)


docvecs = [d2v_model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]  # dv for all the project's tweets
print("docvecs[0]", docvecs[0])

pca_results = reduce_with_PCA(docvecs)  # explained variation very low :(
#print('pca_results', pca_results)
#print(pca_results[:, 0])

tsne_results = reduce_with_TSNE(docvecs)
#print('tsne_results', tsne_results)

#display_2D(pca_results, tweets_cats, len(cats))
#display_2D(tsne_results, tweets_cats, len(cats))'''



### try with another model

## get the model trained only on the danish tweets

# the category 6 is shorter than the others (50 tweets instead of 200)
"""for cat in TweetTagCategory.query.all():
    tweets = Tweet.query.filter_by(category=cat.id).all()
    print(len(tweets))"""

danish_cats = [1,2,3,4,5,6,8,9,10,11]
danish_model = D2VModel.query.filter_by(id=2).first().load(verbose=1)

## get the full model from the db (trained on all the tweets)
full_model = D2VModel.query.filter_by(id=1).first().load()


""""## testing data -> clustering with PCA and t-SNE
test_cats = [1,2,3,4,5,8,9,10,11]  # the categories used in the test set

# on danish_model
display_from_cats(danish_model, test_cats)
display_from_cats(danish_model, [2,8])"""


# on full_model
#display_from_cats(full_model, test_cats)



'''### Trying LDA

lda = LinearDiscriminantAnalysis(n_components=2)

tweets = Tweet.query.filter(Tweet.category.in_(test_cats)).all()
X = [danish_model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]
tweets_cats = [tweet.category for tweet in tweets]

X_new = lda.fit_transform(X,tweets_cats)
#print(X_new.shape)

plt.title("Danish_model")
sns.scatterplot(
        x=X_new[:,0], y=X_new[:,1],
        hue=tweets_cats,
        palette=sns.color_palette("hls", len(test_cats)),
        legend="full",
        alpha=0.3)


plot_confusion_matrix(lda, X, tweets_cats)
plt.figure(3)

### LDA on pretrained model

pretrained_model = D2VModel.query.filter_by(id=3).first().load()
print(pretrained_model.corpus_count)
lda = LinearDiscriminantAnalysis(n_components=2)

X = [pretrained_model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]
X_new = lda.fit_transform(X,tweets_cats)

sns.scatterplot(
        x=X_new[:,0], y=X_new[:,1],
        hue=tweets_cats,
        palette=sns.color_palette("hls", len(test_cats)),
        legend="full",
        alpha=0.3)

plt.title("Pretrained model")
plot_confusion_matrix(lda, X, tweets_cats)

## Train the pretrained model

# data
n = pretrained_model.corpus_count
training_set = [TaggedDocument(simple_preprocess(tweet.text),[tweet.id+n]) for tweet in tweets]

print("build vocab")
pretrained_model.build_vocab(training_set, update=True)
print("train")
pretrained_model.train(training_set, total_examples=len(training_set), epochs=pretrained_model.epochs)
print('ok')

# replot LDA
lda = LinearDiscriminantAnalysis(n_components=2)

X = [pretrained_model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]
X_new = lda.fit_transform(X,tweets_cats)

sns.scatterplot(
        x=X_new[:,0], y=X_new[:,1],
        hue=tweets_cats,
        palette=sns.color_palette("hls", len(test_cats)),
        legend="full",
        alpha=0.3)

plt.title("Pretrained model")
plot_confusion_matrix(lda, X, tweets_cats)'''



### data shape
"""from nlp4all.models import TweetTagCategory, Tweet, User, Organization, Project, BayesianAnalysis
from nlp4all import db, bcrypt, models
import json
from datetime import datetime
import time
import json


proj = models.Project.query.first()
print(proj.tf_idf)
analysis = models.BayesianAnalysis.query.first()
print(analysis.data)"""


### tests on pca

cats = [1,2,3]
pca = PCA(n_components=3)

docvecs = []

for cat_id in cats:
    tweets = Tweet.query.filter_by(category=cat_id).all()
    dv = [danish_model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]
    pca.fit(dv)
    docvecs.append(dv)

data_x = []
data_y = []
data_z = []
for dv in docvecs:
    vecs = pca.transform(dv)
    data_x.append(list(vecs[:,0]))
    data_y.append(list(vecs[:,1]))
    data_z.append(list(vecs[:,2]))

print(len(data_x))
print(len(data_x[0]))
