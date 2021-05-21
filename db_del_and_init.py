from nlp4all import db, bcrypt
from nlp4all.utils import add_project, tf_idf_from_tweets_and_cats_objs, create_n_train_and_test_sets
from nlp4all.models import User, Role, Organization, Project
from nlp4all.models import TweetTagCategory, Tweet, User, Organization, Project, BayesianAnalysis, D2VModel
from nlp4all import db, bcrypt
import json, os
from datetime import datetime
import time
import json
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec

#import utils  # potentially used, but see if it's unnecessary

db.drop_all()


db.create_all()
db.session.commit()

admin_role = Role(name='Admin')
db.session.add(admin_role)
student_role = Role(name='Student')
db.session.add(student_role)
teacher_role = Role(name='Teacher')
db.session.add(teacher_role)
db.session.commit()

hp = bcrypt.generate_password_hash("1234")
user = User(username="telma1", email="telma@telma.dk", password=hp)
user.roles = [admin_role,]
db.session.add(user)

org = Organization(name="UBI/CCTD")
db.session.add(org)
db.session.commit()

#user = User(username="arthurhjorth_teacher", email="arthur.hjorth@u.northwestern.edu", password=hp, organizations=[org,])
#user.roles = [teacher_role,]
#db.session.add(user)
user = User(username="telma", email="telma@email.com", password=hp, organizations=[org,])
user.roles = [student_role,]
db.session.add(user)
db.session.commit()


def clean_word(aword):  # added
    if "@" in aword:
        return "@twitter_ID"
    if "#" in aword:
        return "#hashtag"
    if "http" in aword:
        return "http://link"
    return aword


data_dir = 'tweet_data/'
english_files = [f for f in os.listdir(data_dir) if 'json_out.json' in f]
danish_data = '/Users/Boulot/nlp4all/tweet_data/Big data/DanishPartyTweets.json'

existing_tag_names = []

# danish data
with open(danish_data, 'r') as inf:
    counter = 0
    for line in inf.readlines()[::]: # choose how many tweets you want from each party file
        indict = json.loads(line)
        indict["created_at"] = indict["created_at"].replace('T', ' ')
        indict["created_at"] = indict["created_at"][:-5]  # get rid off the .000Z at the end
        #  add category if it does not already exist
        if indict['twitter_id'] not in existing_tag_names:
            category = TweetTagCategory.query.filter_by(name = indict['twitter_id']).first()
            if not category:
                category = TweetTagCategory(name=indict['twitter_id'], description="Tweet from " + indict['twitter_id'],
                                            language='danish')
                db.session.add(category)
                db.session.commit()
                print(category.name)
            existing_tag_names.append(indict['twitter_id'])
        category = TweetTagCategory.query.filter_by(name=indict['twitter_id']).first()
        date_str = indict['created_at']
        date_rep = '%Y-%m-%d %H:%M:%S'
        #date_rep = '%a %b %d %H:%M:%S %z %Y'
        unix_time = time.mktime(datetime.strptime(date_str, date_rep).timetuple())
        timestamp = datetime.fromtimestamp(unix_time)
        t = indict['full_text']
        t.replace(".", " ")
        t.replace("!", " ")
        t.replace("?", " ")
        t.replace(":", " ")
        t.replace("-", " ")
        t.replace("-", " ")
        t.replace(",", " ")
        t.replace("\(", " ")
        t.replace("\)", " ")
        a_tweet = Tweet(
                time_posted=timestamp,
                category=category.id,
                language=category.language,
                full_text=indict['full_text'],
                handle=indict['twitter_id'],
                text=" ".join([clean_word(word) for word in t.split()]),  # changed
                words=[re.sub(r'[^\w\s]','',w) for w in t.lower().split() if "#" not in w and "http" not in w and "@" not in w],
                links=[w for w in t.split() if "http" in w],
                hashtags=[w for w in t.split() if "#" in w],
                mentions=[w for w in t.split() if "@" in w],
                url="https://twitter.com/"+indict['twitter_id']+"/"+str(indict['id'])
                )

        db.session.add(a_tweet)

# english tweets
for f in english_files:
    with open(data_dir+f) as inf:
        print(f)
        counter = 0
        for line in inf.readlines()[::]: # choose how many tweets you want from each party file
            indict = json.loads(line)
            #  add category if it does not already exist
            if indict['twitter_handle'] not in existing_tag_names:
                category = TweetTagCategory.query.filter_by(name=indict['twitter_handle']).first()
                if not category:
                    category = TweetTagCategory(name=indict['twitter_handle'], description="Tweet from " + indict['twitter_handle'],
                                                language='english')
                    db.session.add(category)
                    db.session.commit()
                existing_tag_names.append(indict['twitter_handle'])
            category = TweetTagCategory.query.filter_by(name=indict['twitter_handle']).first()
            date_str = indict['time']
            date_rep = '%a %b %d %H:%M:%S %z %Y'
            unix_time = time.mktime(datetime.strptime(date_str, date_rep).timetuple())
            timestamp = datetime.fromtimestamp(unix_time)
            t = indict['full_text']
            t.replace(".", " ")
            t.replace("!", " ")
            t.replace("?", " ")
            t.replace(":", " ")
            t.replace("-", " ")
            t.replace("-", " ")
            t.replace(",", " ")
            t.replace("\(", " ")
            t.replace("\)", " ")
            a_tweet = Tweet(
                time_posted = timestamp,
                category = category.id,
                language = category.language,
                full_text = indict['full_text'],
                handle = indict['twitter_handle'],
                text= " ".join([clean_word(word) for word in t.split()]), # changed
                words = [re.sub(r'[^\w\s]','',w) for w in t.lower().split() if "#" not in w and "http" not in w and "@" not in w],
                links = [w for w in t.split() if "http" in w],
                hashtags = [w for w in t.split() if "#" in w],
                mentions = [w for w in t.split() if "@" in w],
                url = "https://twitter.com/"+indict['twitter_handle']+"/"+str(indict['id'])
                )
            
            db.session.add(a_tweet)


db.session.commit()
db.session.close()

org = Organization.query.first()
all_cats = TweetTagCategory.query.all()
cats = [all_cats[1], all_cats[7]]
cat_ids = [all_cats[1].id, all_cats[7].id]


# Telma added
tweets1 = Tweet.query.filter_by(category=2).all() # this should be done in a better way..
tweets2 = Tweet.query.filter_by(category=8).all()

mytweets = tweets1 +tweets2

tf_idf = {}
tf_idf['cat_counts'] = { cat.id : 0 for cat in cats}
tf_idf['words'] = {}
all_words = sorted(list(set([word for t in mytweets for word in t.words])))

for tweet in mytweets:
    tf_idf['cat_counts'][tweet.category] = tf_idf['cat_counts'][tweet.category] + 1
    for word in tweet.words:
        the_list = tf_idf['words'].get(word, [])
        the_list.append((tweet.id, tweet.category))
        tf_idf['words'][word] = the_list

cats_objs = TweetTagCategory.query.filter(TweetTagCategory.id.in_(cat_ids)).all()
tweet_objs = [t for cat in cats_objs for t in cat.tweets]
tf_idf = tf_idf_from_tweets_and_cats_objs(tweet_objs, cats_objs)
tweet_id_and_cat = { t.id : t.category for t in tweet_objs }
training_and_test_sets = create_n_train_and_test_sets(30, tweet_id_and_cat)


# Full project
project = Project(name='Full project', description='Project with all the data', organization=org.id, categories=all_cats,
                  tweets=Tweet.query.all(), tf_idf=tf_idf, training_and_test_sets=training_and_test_sets)
db.session.add(project)

#project = Project(name="DF og Ehl", organization=org.id, categories=cats)
#project= add_project(name="DF og ehl", description="", org=org, cat_ids=cats)
project = Project(name='Test project', description='Just a test project', organization=org.id, categories=cats_objs,
                  tweets=tweet_objs, tf_idf=tf_idf, training_and_test_sets=training_and_test_sets)
db.session.add(project)


db.session.commit()


### Doc2Vec


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        self.epoch += 1


# parameters
# Arbitrary. Could be changed
vector_size = 300
min_count = 5
epochs = 300


# Initializing the model
d2v_model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)


# creating the database
tweets = Tweet.query.all()

train_corpus = []
for tweet in tweets:
    train_corpus.append(TaggedDocument(simple_preprocess(tweet.text), [tweet.id]))
print(train_corpus[1])

print("Start full model training")
# Create the vocabulary and train the model
d2v_model.build_vocab(train_corpus)
epoch_logger = EpochLogger()
d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs,
                callbacks=[epoch_logger])


# save d2v_model into the database
d2v = D2VModel(id=1, name='Full model', description="Trained on the entire corpus, dim=300", project=1,
               public='all')
d2v.save(d2v_model)

db.session.add(d2v)
db.session.commit()


## Adding a danish only model to the db

danish_model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)

for cat in TweetTagCategory.query.all():
    print(cat.id, cat.name)
danish_cats_numbers = [cat.id for cat in TweetTagCategory.query.filter_by(language='danish')]
print(danish_cats_numbers)
danish_tweets = Tweet.query.filter(Tweet.category.in_(danish_cats_numbers)).all()  # training set. Only danish tweets
#print(danish_tweets)

train_corpus = []
for tweet in danish_tweets:
    train_corpus.append(TaggedDocument(simple_preprocess(tweet.text), [tweet.id]))

print("Start training danish model")
danish_model.build_vocab(train_corpus)
epoch_logger = EpochLogger()
danish_model.train(train_corpus, total_examples=danish_model.corpus_count, epochs=danish_model.epochs,
                   callbacks=[epoch_logger])

danish_d2v = D2VModel(id=2, name='Danish model', project=1, public='all')
danish_d2v.save(danish_model, description="Model trained only on danish tweets, dim=300")

db.session.add(danish_d2v)
db.session.commit()

print("full model project", d2v.project)
print("danish model project", danish_d2v.project)
print("gensim models of project1", Project.query.first().d2v_models)


## tests

d2vs = D2VModel.query.all()
print("total nb of models", len(d2vs))

new_d2v = d2vs[0]
gensim_d2v = new_d2v.load()  # loaded model. d2v_model is the initial one

# compare some values before and after loading
# therefore all the values should be equal
# vocab length
print("vocab lengths")
print(len(d2v_model.wv.index_to_key))
print(len(gensim_d2v.wv.index_to_key))

# a word vector
print(d2v_model.wv['advare'])
print(gensim_d2v.wv['advare'])

# a doc vector
print(d2v_model.dv[2])
print(gensim_d2v.dv[2])

# a model parameter
print(d2v_model.epochs)
print(gensim_d2v.epochs)


"""for t in Tweet.query.all():
    print(t)"""
