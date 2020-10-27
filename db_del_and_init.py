from nlp4all import db, bcrypt
from nlp4all.utils import add_project, tf_idf_from_tweets_and_cats_objs, create_n_train_and_test_sets
from nlp4all.models import User, Role, Organization, Project
from nlp4all.models import TweetTagCategory, Tweet, User, Organization, Project, BayesianAnalysis
from nlp4all import db, bcrypt
import json, os
from datetime import datetime
import time
import json
#import utils

db.drop_all()


db.create_all()
db.session.commit()

admin_role = Role(name='Admin')
db.session.add(admin_role)
student_role = Role(name='Student')
db.session.add(admin_role)
teacher_role = Role(name='Teacher')
db.session.add(admin_role)
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

def clean_word(aword): # added
    if "@" in  aword:
        return "@twitter_ID"
    if "#" in aword:
        return "#hashtag"
    if "http" in aword:
        return "http://link"
    return aword

data_dir = 'tweet_data/'
files = [f for f in os.listdir(data_dir) if '_out.json' in f]

existing_tag_names = []
for f in files:
    with open(data_dir+f) as inf:
        print(f)
        counter = 0
        for line in inf.readlines()[:200]: # choose how many tweets you want from each party file
            indict = json.loads(line)
#             add cateogry if it does not already exist
            if indict['twitter_handle'] not in existing_tag_names:
                category = TweetTagCategory.query.filter_by(name = indict['twitter_handle']).first()
                if not category:
                    category = TweetTagCategory(name = indict['twitter_handle'], description = "Tweet from " + indict['twitter_handle'])
                    db.session.add(category)
                    db.session.commit()
                existing_tag_names.append(indict['twitter_handle'])
            category = TweetTagCategory.query.filter_by(name = indict['twitter_handle']).first()
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
                full_text = indict['full_text'],
                handle = indict['twitter_handle'],
                text= " ".join([clean_word(word) for word in t.split()]), # changed
                words = [w for w in t.lower().split() if "#" not in w and "http" not in w and "@" not in w],
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
    for word  in tweet.words:
            the_list = tf_idf['words'].get(word, [])
            the_list.append((tweet.id, tweet.category))
            tf_idf['words'][word] = the_list

cats_objs = TweetTagCategory.query.filter(TweetTagCategory.id.in_(cat_ids)).all()
tweet_objs = [t for cat in cats_objs for t in cat.tweets]
tf_idf = tf_idf_from_tweets_and_cats_objs(tweet_objs, cats_objs)
tweet_id_and_cat = { t.id : t.category for t in tweet_objs }
training_and_test_sets = create_n_train_and_test_sets(30, tweet_id_and_cat)


#project = Project(name="DF og Ehl", organization=org.id, categories=cats)
#project= add_project(name="DF og ehl", description="", org=org, cat_ids=cats)
project = Project(name = 'name', description = 'description', organization = org.id, categories = cats_objs, tweets = tweet_objs, tf_idf = tf_idf, training_and_test_sets = training_and_test_sets)
db.session.add(project)


db.session.commit()


for t in Tweet.query.all():
    print(t)