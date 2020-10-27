from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from nlp4all import db, login_manager, app, utils
from flask_login import UserMixin
from sqlalchemy.types import JSON
import collections
import collections, functools, operator 
import statistics

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class BayesianRobot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(25))
    parent = db.Column(db.Integer, db.ForeignKey('bayesian_robot.id'), default=None)
    child = db.Column(db.Integer, db.ForeignKey('bayesian_robot.id'), default=None)
    analysis = db.Column(db.Integer, db.ForeignKey('bayesian_analysis.id'))
    features = db.Column(JSON, default = {})
    accuracy = db.Column(JSON, default = {})
    retired = db.Column(db.Boolean, default=False)
    time_retired = db.Column(db.DateTime)

    def clone(self):
        new_robot = BayesianRobot()
        new_robot.name = self.name
        new_robot.analysis = self.analysis
        new_robot.features = self.features
        new_robot.parent = self.id
        return(new_robot)
    

    # def run_analysis(self):
    def get_analysis():
        return BayesianAnalysis.query.get(self.analysis)


    def word_in_features(self, word):
        for f in self.features.keys():
            feature_string = f.lower()
            if feature_string.startswith('*') and feature_string.endswith('*'):
                if feature_string[1:-1] in word:
                    return(True)
            elif feature_string.startswith('*'):
                if word.endswith(feature_string[1:]):
                    return(True)
            elif feature_string.endswith('*'):
                if word.startswith(feature_string[:1]):
                    return(True)
            else:
                if word == feature_string :
                    return(True)
        return(False)

    def accuracy_for_tnt_set(words, tweets_with_word, words_by_tweet, tnt_sets):
        accuracy_dict = {}
        accuracy = 0
        # find de relevante tweets
        total_predictions_by_word = {}
        for tnt_set in tnt_sets:
            word_predictions = {}
            for word in words:
                categories_with_word_in_training = [n[1] for n in tnt_set[0].items() if n[0] in str(tweets_with_word[word])]
                predictions = {n : categories_with_word_in_training.count(n) / len(categories_with_word_in_training) for n in set(categories_with_word_in_training) }
                word_predictions[word] = predictions
            train_set = tnt_set[1]
            for t in train_set:
                if int(t) in words_by_tweet.keys():
                    category_predictions = collections.Counter({})
                    for word in words_by_tweet[int(t)]:
                        category_predictions = category_predictions + collections.Counter(word_predictions[word])
                    # predicted_category = max(category_predictions, key = lambda  k: category_predictions[k])
                    real_cat = Tweet.query.get(int(t)).category
                else:
                    accuracy_dict['uncategorized'] = []
        return accuracy_dict
            

    def calculate_accuracy(self):
        analysis_obj = BayesianAnalysis.query.get(self.analysis)
        proj_obj = Project.query.get(analysis_obj.project)
        tf_idf = proj_obj.tf_idf
        ## skriv det her om så accuraacy bregnes per feature, og ikke per ord.
        # relevant_words = [word for  word in tf_idf.get('words') if BayesianRobot.word_in_features(self, word)]
        feature_words = {}
        for feature in self.features:
            feature_words[feature] = [word for word in tf_idf.get('words') if BayesianRobot.matches(word, feature)]
        # relevant_words = [w for words in feature_words.values() for w in words]
        # first calculate the predictions, based on the training sets.
        predictions_by_feature = {}
        # initialize test_set_tweets so we dont need to calculate it twice
        test_set_tweets = set()
        cats = [c.id for c in proj_obj.categories]
        
        # make one for individual words too so we can more easily access them later, and make a  list of category names for viewing
        word_category_predictions = {}
        cat_names = {cat.id : cat.name for cat in Project.query.get(analysis_obj.project).categories}
        
        for feature in feature_words:
            predictions_by_feature[feature] = {}
            for word in feature_words[feature]:
                for dataset in proj_obj.training_and_test_sets[:1]:
                    train_set = dataset[0]
                    tweets = tf_idf.get('words').get(word)
                    train_set_tweets = []
                    for t in tweets:
                        if str(t[0]) in train_set.keys():
                            train_set_tweets.append(t)
                        else:
                            test_set_tweets.add(t[0])
                    categories_in_dataset = [dataset[0].get(str(tweet[0])) for tweet in train_set_tweets]
                    cat_counts = {c : categories_in_dataset.count(c) for c in cats}
                    total_cats = sum(cat_counts.values())
                    predictions = 0
                    # if there are no words in the training set to learn from, we simply ignore the word and do not append anything here
                    if total_cats > 0:
                        predictions = {c : cat_counts[c] / sum(cat_counts.values()) for c in cats}
                        category_dict = {"category_prediction" : cat_names[max(predictions.items(), key=operator.itemgetter(1))[0]]}
                        word_category_predictions[word] = category_dict
                        predictions_by_feature[feature][word] = predictions
        # now for each word, figure out which tweets contain them, and build - for each tweet - a classification, that we can then compare to the real value

        test_set = proj_obj.training_and_test_sets[0][1]
        tweet_predictions = {}

        for word_prediction in predictions_by_feature.values():
            for word, predictions in word_prediction.items():
                word_tweets = tf_idf.get('words').get(word)
                test_set_tweets = [tweet for tweet in word_tweets if str(tweet[0]) in test_set.keys()]
                for tweet in test_set_tweets:
                    preds = tweet_predictions.get(tweet[0], {'predictions' : [], 'words' : [], 'category' : tweet[1]})
                    preds['predictions'].append(predictions)
                    preds['words'].append(word)
                    tweet_predictions[tweet[0]] = preds
        # now finally evaluate how well we did, in general and by word
        word_accuracy = {}
        for tweet_key in tweet_predictions:
            prediction_dict = tweet_predictions[tweet_key].copy()
            # for d in prediction_dict['predictions']:
            #     if 'category_prediction' in d.keys():
            #         del d['category_prediction']
            summed_prediction = dict(functools.reduce(operator.add, map(collections.Counter, prediction_dict['predictions'])))
            ## the old code that makes summed_prediction also includes the newly added "category_prediction". Since we don't want to
            ## sum that, we remove it first
            # it can happen that we evaluate a word that we have no information on. In that 
            cat_prediction = max(summed_prediction.items(), key=operator.itemgetter(1))[0] 
            tweet_predictions[tweet_key]['correct'] = test_set[str(tweet_key)] == cat_prediction
            # save a per-word accuracy
            for word in prediction_dict['words']:
                acc = word_accuracy.get(word, [])
                acc.append(tweet_predictions[tweet_key]['correct'])
                word_accuracy[word] = acc
        # and then build a nice dict full of info 
        feature_info = {}
        for feature in feature_words:
            feature_info[feature] = {}
            feature_info[feature]['words'] = {}
            for word in feature_words[feature]:
                word_dict = feature_info[feature].get(word, {})
                if word in word_accuracy: # the word is only in the word_accuracy dict if it was in the test set
                    word_dict['tweets_targeted'] = len(word_accuracy[word])
                    word_dict['accuracy'] = round(len([x for x in word_accuracy[word] if x]) / len(word_accuracy[word]), 2)
                    feature_info[feature]['words'][word] = word_dict
                # else:
                #     # if it's not in the test set, we just take it out.
                #     word_dict['tweets_targeted'] = 0
                #     word_dict['accuracy'] = 0
                # feature_info[feature]['words'][word] = word_dict
            
            
            accuracy_values = [d['accuracy'] for d in feature_info[feature]['words'].values()]
            targeted_values = [d['tweets_targeted'] for d in feature_info[feature]['words'].values()]
            if len(accuracy_values) > 0:
                feature_info[feature]['accuracy'] = sum(accuracy_values) / len(accuracy_values)
                feature_info[feature]['tweets_targeted'] = sum(targeted_values)
            else:
                feature_info[feature]['accuracy'] = 0
                feature_info[feature]['tweets_targeted'] = 0
        tweets_targeted = 0
        table_data = []
        for f in feature_info:
            tweets_targeted = tweets_targeted + feature_info[f]['tweets_targeted']
            feat_dict = {}
            feat_dict['word'] = f
            feat_dict['category_prediction'] = "N/A"
            feat_dict['accuracy'] = feature_info[f]['accuracy']
            feat_dict['tweets_targeted'] = feature_info[f]['tweets_targeted']
            feat_dict['score'] = round(feat_dict['accuracy'] * feat_dict['tweets_targeted'], 2)
            # calculate the most often predicted category. This isn't trivial - should it be by total tweets in test set, or just the most common
            # category across its words? Well, it's obvious. Boo. It needs to be weighted by how many tweets there are.
            # NO  NO NO! I thought about that wrong. We just want the average of each of the category prediction for each word.
            ca_tid_scores = {cat_id : 0 for cat_id in cat_names}
            print(feat_dict)
            table_data.append(feat_dict)
            for word in feature_info[f]['words']:
                feat_dict = {}
                feat_dict['word'] = word
                feat_dict['category_prediction'] = word_category_predictions[word]['category_prediction']
                feat_dict['accuracy'] = feature_info[f]['words'][word]['accuracy']
                feat_dict['tweets_targeted'] = feature_info[f]['words'][word]['tweets_targeted']
                feat_dict['score'] = round(feat_dict['accuracy'] * feat_dict['tweets_targeted'], 2)
                table_data.append(feat_dict)
        if len(tweet_predictions) == 0:
            accuracy = 0
        else:
            accuracy = len([d for d in tweet_predictions.values() if d['correct']]) / len(tweet_predictions)
        accuracy_info = {'accuracy' : round(accuracy, 2), 'tweets_targeted' : tweets_targeted}
        accuracy_info['features'] = feature_info
        accuracy_info['table_data'] = table_data
        return accuracy_info




    def matches(aword, afeature):
        feature_string = afeature.lower()
        if feature_string.startswith('*') and feature_string.endswith('*'):
            if feature_string[1:-1] in aword:
                return True
        elif feature_string.startswith('*'):
            if aword.endswith(feature_string[1:]):
                return True
        elif feature_string.endswith('*'):
            # if aword.startswith(feature_string[:1]): ## this was a bug. Leave it in if people want to see it.
            if aword.startswith(feature_string[:-1]):
                return True
        else:
            if aword == feature_string :
                return True
        return False

    def feature_words(self, a_feature, tf_idf):
        return_list = []
        words = tf_idf.get('words')
        feature_string = a_feature.lower()
        for word in words:
            if feature_string.startswith('*') and feature_string.endswith('*'):
                if feature_string[1:-1] in word:
                    return_list.append(word)
            elif feature_string.startswith('*'):
                if word.endswith(feature_string[1:]):
                    return_list.append(word)
            elif feature_string.endswith('*'):
                if word.startswith(feature_string[:1]):
                    return_list.append(word)
            else:
                if word == feature_string :
                    return_list.append(word)
        return(return_list)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    organizations = db.relationship('Organization', secondary='user_orgs')
    admin = db.Column(db.Boolean, default=False)
    roles = db.relationship('Role', secondary='user_roles')
    analyses = db.relationship('BayesianAnalysis')

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"

# Define the Role data-model
class Role(db.Model):
    __tablename__ = 'role'
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(50), unique=True)


# Define the ProjectCategories association table
class ProjectCategories(db.Model):
    __tablename__ = 'project_categories'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('project.id', ondelete='CASCADE'))
    role_id = db.Column(db.Integer(), db.ForeignKey('tweet_tag_category.id', ondelete='CASCADE'))

# Define the UserOrgs association table
class UserOrgs(db.Model):
    __tablename__ = 'user_orgs'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id', ondelete='CASCADE'))
    role_id = db.Column(db.Integer(), db.ForeignKey('organization.id', ondelete='CASCADE'))


# Define the Tweet-Project association table
class TweetProject(db.Model):
    __tablename__ = 'tweet_project'
    id = db.Column(db.Integer(), primary_key=True)
    tweet = db.Column(db.Integer(), db.ForeignKey('tweet.id', ondelete='CASCADE'))
    project = db.Column(db.Integer(), db.ForeignKey('project.id', ondelete='CASCADE'))

# Define the UserRoles association table
class UserRoles(db.Model):
    __tablename__ = 'user_roles'
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id', ondelete='CASCADE'))
    role_id = db.Column(db.Integer(), db.ForeignKey('role.id', ondelete='CASCADE'))

# Define the Matrix-Categories association table
class MatrixCategories(db.Model):
    __tablename__ = 'confusionmatrix_categories'
    id = db.Column(db.Integer(), primary_key=True)
    matrix_id = db.Column(db.Integer(), db.ForeignKey('confusion_matrix.id', ondelete='CASCADE'))
    category_id = db.Column(db.Integer(), db.ForeignKey('tweet_tag_category.id', ondelete='CASCADE'))

# Define the Tweet-Matrix association table
class TweetMatrix(db.Model):
    __tablename__ = 'tweet_confusionmatrix'
    id = db.Column(db.Integer(), primary_key=True)
    tweet = db.Column(db.Integer(), db.ForeignKey('tweet.id', ondelete='CASCADE'))
    matrix = db.Column(db.Integer(), db.ForeignKey('confusion_matrix.id', ondelete='CASCADE'))

# class Post(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(100), nullable=False)
#     date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
#     content = db.Column(db.Text, nullable=False)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

#     def __repr__(self):
#         return f"Post('{self.title}', '{self.date_posted}')"

class Organization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    users = db.relationship('User', secondary='user_orgs')
    projects = db.relationship('Project')

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    description = db.Column(db.String)
    organization = db.Column(db.Integer, db.ForeignKey('organization.id'))
    analyses = db.relationship('BayesianAnalysis')
    categories = db.relationship('TweetTagCategory', secondary='project_categories')
    tf_idf = db.Column(JSON)
    tweets = db.relationship('Tweet', secondary='tweet_project')
    training_and_test_sets = db.Column(JSON)

    def get_tweets(self):
        return [t for cat in categories for t in cat.tweets]


class TweetTagCategory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    description = db.Column(db.String(100))
    tweets = db.relationship('Tweet')
    tags = db.relationship('TweetTag')
    projects = db.relationship('Project', secondary='project_categories')
    #matrices = db.relationship('ConfusionMatrix', secondary='matrix_categories')

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time_posted = db.Column(db.DateTime)
    category = db.Column(db.Integer, db.ForeignKey('tweet_tag_category.id'))
    projects = db.Column(db.Integer, db.ForeignKey('project.id'))
    handle = db.Column(db.String(15))
    full_text = db.Column(db.String(280))
    words = db.Column(JSON)
    hashtags = db.Column(JSON)
    tags = db.relationship('TweetTag')
    links = db.Column(JSON)
    mentions = db.Column(JSON)
    url = db.Column(db.String(200), unique=True)
    text = db.Column(db.String(300)) 
    
class TweetTag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.Integer, db.ForeignKey('user.id'))
    category = db.Column(db.Integer, db.ForeignKey('tweet_tag_category.id'))
    analysis = db.Column(db.Integer, db.ForeignKey('bayesian_analysis.id', ondelete="CASCADE"))
    tweet = db.Column(db.Integer, db.ForeignKey('tweet.id', ondelete="CASCADE"))
    time_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class BayesianAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(50))
    tags = db.relationship('TweetTag') # this also tells us which tweets
    data = db.Column(JSON)
    project = db.Column(db.Integer, db.ForeignKey('project.id'))
    robots = db.relationship('BayesianRobot')
    shared = db.Column(db.Boolean, default=False)
    tweets = db.Column(JSON, default=[])

    def get_project(self):
        return Project.query.get(self.project)

    def updated_data(self, tweet, category):
        self.data['counts'] = self.data['counts'] + 1
        if category.name not in self.data.keys():
            self.data[category.name] = {'counts' : 0, 'words' : {}}
        self.data[category.name]['counts'] = (self.data[category.name].get('counts', 0)) + 1
        for w in set(tweet.words):
            val = self.data[category.name]['words'].get(w, 0)
            self.data[category.name]['words'][w] = val + 1
        return self.data
    
    def get_predictions_and_words(self, words):
        # take each word  and  calculate a probabilty for each category
        categories = Project.query.get(self.project).categories
        category_names = [c.name for c in categories if c.name in self.data.keys()]
        preds = {}
        predictions = {}
        if self.data['counts'] == 0:
            predictions = {c : {w : 0} for w in words for c in category_names}
            # predictions = {word : {category : 0 for category in category_names} for word in words}
        else:
            for w in words: # only categorize each word once
                preds[w] = {c : 0 for c in category_names}
                for cat in category_names:
                    predictions[cat] = predictions.get(cat, {})
                    prob_ba = self.data[cat]['words'].get(w, 0) / self.data[cat]['counts']
                    prob_a = self.data[cat]['counts'] / self.data['counts'] 
                    prob_b = sum([self.data[c]['words'].get(w, 0) for c in category_names]) / self.data['counts']
                    if  prob_b == 0:
                        preds[w][cat] = 0
                        predictions[cat][w] = 0
                    else:
                        preds[w][cat] = round(prob_ba * prob_a / prob_b, 2)
                        predictions[cat][w] = round(prob_ba * prob_a / prob_b, 2)

        return (preds, {k : round(sum(v.values()) / len(set(words)),2) for k, v in predictions.items()})

class ConfusionMatrix(db.Model):
   
    id = db.Column(db.Integer, primary_key=True)
    categories = db.relationship('TweetTagCategory', secondary='confusionmatrix_categories')
    tweets = db.relationship('Tweet', secondary='tweet_confusionmatrix') # add these later?
    matrix_data = db.Column(JSON) # here to save the TP/TN/FP/FN (+ probability?)
    train_data = db.Column(JSON)
    tf_idf = db.Column(JSON)
    training_and_test_sets = db.Column(JSON)
    threshold = db.Column(db.Float())

    def updated_data(self, tweet, category):
        self.train_data['counts'] = self.train_data['counts'] + 1
        
        if category.name not in self.train_data.keys():
            self.train_data[category.name] = {'counts' : 0, 'words' : {}}
        self.train_data[category.name]['counts'] = (self.train_data[category.name].get('counts', 0)) + 1
        for w in set(tweet.words):
            val = self.train_data[category.name]['words'].get(w, 0)
            self.train_data[category.name]['words'][w] = val + 1
        return self.train_data

    def update_tnt_set(self, ratio):
        tweet_id_and_cat = { t.id : t.category for t in self.tweets}
        self.training_and_test_sets = utils.create_n_split_tnt_sets(30, ratio, tweet_id_and_cat)
        return self.training_and_test_sets

    def get_predictions_and_words(self, words):
        
        categories = self.categories
        category_names = [c.name for c in categories]
        preds = {}
        predictions = {}
        if self.train_data['counts'] == 0:
            predictions = {c : {w : 0} for w in words for c in category_names}
            # predictions = {word : {category : 0 for category in category_names} for word in words}
        else:
            for w in words: # only categorize each word once
                preds[w] = {c : 0 for c in category_names}
                for cat in category_names:
                    predictions[cat] = predictions.get(cat, {})
                    prob_ba = self.train_data[cat]['words'].get(w, 0) / self.train_data[cat]['counts']
                    prob_a = self.train_data[cat]['counts'] / self.train_data['counts'] 
                    prob_b = sum([self.train_data[c]['words'].get(w, 0) for c in category_names]) / self.train_data['counts']
                    if  prob_b == 0:
                        preds[w][cat] = 0
                        predictions[cat][w] = 0
                    else:
                        preds[w][cat] = round(prob_ba * prob_a / prob_b, 2)
                        predictions[cat][w] = round(prob_ba * prob_a / prob_b, 2)

        return (preds, {k : round(sum(v.values()) / len(set(words)),2) for k, v in predictions.items()})

    

