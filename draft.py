from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
from nlp4all.models import Tweet, TweetTagCategory, Project, D2VModel
from nlp4all import db
from nlp4all.utils import assign_colors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import numpy as np


all_cats = [cat.id for cat in TweetTagCategory.query.all()]
print(all_cats)
danish_cats = [1,2,3,4,5,6,8,9,10,11]

full_model = D2VModel.query.filter_by(id=1).first().load()
danish_model = D2VModel.query.filter_by(id=2).first().load(verbose=1)



# test on epoch logger
class EpochLogger(CallbackAny2Vec):
     '''Callback to log information about training'''

     def __init__(self):
        self.epoch = 0

     def on_epoch_end(self, model):
         print("Epoch {} end".format(self.epoch))  # maybe flash instead of print
         self.epoch += 1

epoch_logger = EpochLogger()

training_set = [TaggedDocument(tweet.text, [tweet.id]) for tweet in Tweet.query.all()]
gensim_model = Doc2Vec()
gensim_model.build_vocab(training_set)
gensim_model.train(training_set, total_examples=gensim_model.corpus_count, epochs=gensim_model.epochs,
                   callbacks=[epoch_logger])



### Visualization

## get the model trained only on the danish tweets

# the category 6 is shorter than the others (50 tweets instead of 200)

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


'''### tests on pca

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
print(len(data_x[0]))'''
