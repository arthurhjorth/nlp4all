from nlp4all.models import Tweet
from nlp4all import celery_app
from flask import flash
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models.callbacks import CallbackAny2Vec
import pickle


@celery_app.task
def reduce_with_PCA(data, n_components=2, verbose=False):
    # data should be a list of numpy arrays, or a pandas dataframe, or something similar
    # row = a tweet ; column = a dimension of the word space
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data)
    if verbose:
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return pca_results


@celery_app.task
def reduce_with_TSNE(data, n_components=2, perplexity=30, n_iter=1000):
    # if too much dimensions, use PCA to reduce first
    if len(data[0]) > 50:
        data = reduce_with_PCA(data, n_components=50)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    tsne_reduced = tsne.fit_transform(data)
    return tsne_reduced


@celery_app.task
def separate_by_cat(data, cats):
    cat_lengths = []
    separated_data = []
    for cat in cats:
        tweets = Tweet.query.filter_by(category=cat).all()
        cat_lengths.append(len(tweets))
    current_pos = 0
    for cat_length in cat_lengths:
        separated_data.append(data[current_pos:current_pos+cat_length])
        current_pos = current_pos + cat_length
    return separated_data


@celery_app.task
def reduce_dimension(dv, cats, method, n_components, labels):
    data_x = []
    data_y = []
    data_z = []  # always created, but actually used only in 3D
    if method == 'reduce_with_PCA':
        reduction_function = reduce_with_PCA
    elif method == 'reduce_with_TSNE':
        reduction_function = reduce_with_TSNE
    reduced_dv = reduction_function(dv, n_components=n_components)
    separated_docvecs = separate_by_cat(reduced_dv, cats)
    for cat_vecs in separated_docvecs:
        data_x.append(cat_vecs[:,0].tolist())
        data_y.append(cat_vecs[:,1].tolist())
        if n_components == 3:
            data_z.append(cat_vecs[:,2].tolist())
    return data_x, data_y, data_z, labels


class EpochFlasher(CallbackAny2Vec):
     '''Callback to flash information about training'''

     def __init__(self):
        self.epoch = 0

     def on_epoch_end(self, model):
         flash("Epoch {} end".format(self.epoch))
         self.epoch += 1


@celery_app.task
def train_d2v(gensim_model_json, data):
    # takes as argument an already constructed model and a training set
    # Used exactly in the same context as model.train, the only difference being it can be used with celery
    gensim_model = pickle.loads(gensim_model_json)
    epoch_flasher = EpochFlasher()
    gensim_model.train(data, total_examples=gensim_model.corpus_count, epochs=gensim_model.epochs)


from time import sleep
@celery_app.task
def addition(a, b):
    sleep(5)
    return a + b
