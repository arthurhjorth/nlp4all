import time
from nlp4all.models import Tweet, D2VModel
from nlp4all import celery_app, db
from flask import flash
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models.callbacks import CallbackAny2Vec
import pickle
from sqlalchemy.orm import load_only


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
def separate_by_cat(data, cats, nb_vecs):
    cat_lengths = []  # nb of displayed tweets for each cat
    separated_data = []
    for cat in cats:
        tweets = Tweet.query.filter_by(category=cat).all()
        cat_lengths.append(min(len(tweets), nb_vecs))
    current_pos = 0
    for cat_length in cat_lengths:
        separated_data.append(data[current_pos:current_pos+cat_length])
        current_pos = current_pos + cat_length
    return separated_data


@celery_app.task
def reduce_dimension(dv, cats, method, n_components, labels, nb_vecs):
    data_x = []
    data_y = []
    data_z = []  # always created, but actually used only in 3D
    if method == 'reduce_with_PCA':
        reduction_function = reduce_with_PCA
    elif method == 'reduce_with_TSNE':
        reduction_function = reduce_with_TSNE
    reduced_dv = reduction_function(dv, n_components=n_components)
    separated_docvecs = separate_by_cat(reduced_dv, cats, nb_vecs)
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


@celery_app.task
def make_public(model_id):
    db_model = D2VModel.query.options(load_only(*['id','public'])).filter_by(id=model_id).first()
    if db_model.public == 'no':
        db_model.public = 'all'
        print(time.time())
        db.session.commit()
        print(time.time())
        flash('Your model is now public and can be seen by any user of the site.', 'info')
    elif db_model.public == 'all':
        db_model.public = 'no'
        print(time.time())
        db.session.commit()
        print(time.time())
        flash("Your model isn't public anymore and can be seen and used only by you.", 'info')
