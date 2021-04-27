import os, time
import secrets
import nlp4all.utils
from random import sample, shuffle
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, abort, jsonify
from nlp4all import app, db, bcrypt, mail, celery_app
from nlp4all.forms import *
from nlp4all.models import User, Organization, Project, BayesianAnalysis, TweetTagCategory, TweetTag,\
    BayesianRobot, Tweet, ConfusionMatrix, TweetAnnotation, D2VModel
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
import datetime
import json, ast
from sqlalchemy.orm.attributes import flag_modified
from nlp4all.utils import get_user_projects, get_user_project_analyses, reduce_with_PCA, reduce_with_TSNE,\
    separate_by_cat, reduce_dimension
import operator
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from celery.result import AsyncResult


@app.route("/")
@app.route("/home")
@app.route("/home/", methods=['GET', 'POST'])
@login_required
def home():
    my_projects = get_user_projects(current_user)
    return render_template('home.html', projects=my_projects)

@app.route("/robot_summary", methods=['GET', 'POST'])
def robot_summary():
    analysis_id = request.args.get('analysis', 0, type=int)
    analysis = BayesianAnalysis.query.get(analysis_id)
    robots = [r for r in analysis.robots if r.retired]
    return render_template('robot_summary.html', analysis=analysis, robots=robots)

@app.route("/data_table")
def data_table():
    my_projects = get_user_projects(current_user) 
    test_data = [{'a' : 25, 'b': 200, 'c': 400}]
    test_data.append({'a' : 25, 'b': 200, 'c': 80})
    return render_template('data_table.html', table_data=test_data)

@app.route("/add_project", methods=['GET', 'POST'])
def add_project():
    form = AddProjectForm()
    # find forst alle mulige organizations
    form.organization.choices = [( str(o.id), o.name ) for o in Organization.query.all()]
    form.categories.choices = [( str(s.id), s.name ) for s in TweetTagCategory.query.all()]
    if form.validate_on_submit():
        # orgs = [int(n) for n in form.organization.data]
        # orgs_objs = Organization.query.filter(Organization.id.in_(orgs)).all()
        org = Organization.query.get(int(form.organization.data))
        cats = [int(n) for n in form.categories.data]
        project = nlp4all.utils.add_project(name=form.title.data, description=form.description.data, org=org.id, cat_ids=cats)
        project_id = project.id
        return(redirect(url_for('home', project=project_id)))
    return render_template('add_project.html', title='Add New Project', form=form)


@app.route("/project2", methods=['GET', "POST"])
@login_required
def project2():
    project_id = request.args.get('project', None, type=int)
    project = Project.query.get(project_id)
    analyses = project.analyses if (project and project.analyses) else []
    bayesian_form = AddBayesianAnalysisForm()
    embedding_form = AddWordEmbeddingForm(vector_size=50, epochs=40, d2v=True)
    embedding_form.training_set.choices = [(str(cat.id), cat.name) for cat in TweetTagCategory.query.all()]

    if bayesian_form.submit_bay.data and bayesian_form.validate_on_submit():
        userid = current_user.id  # useful?
        name = bayesian_form.name.data
        number_per_category = bayesian_form.number.data
        analysis_tweets = []
        if bayesian_form.shared.data:
            tweets_by_cat = {cat : [t.id for t in project.tweets if t.category == cat.id] for cat in project.categories}
            for cat, tweets in tweets_by_cat.items():
                analysis_tweets.extend(sample(tweets, number_per_category))
        # make sure all students see tweets in the same order. So shuffle them now, and then
        # put them in the database
        shuffle(analysis_tweets)
        analysis = BayesianAnalysis(user=userid, name=name, project=project.id, data={"counts": 0, "words": {}},
                                    shared=bayesian_form.shared.data, tweets=analysis_tweets, annotation_tags={},
                                    annotate=bayesian_form.annotate.data )
        db.session.add(analysis)
        db.session.commit()
        return(redirect(url_for('project2', project=project_id)))

    if embedding_form.submit_emb and embedding_form.validate_on_submit():
        #userid = current_user.id  # useful?
        name = embedding_form.name.data
        vector_size = embedding_form.vector_size.data
        epochs = embedding_form.epochs.data
        training_cats = embedding_form.training_set.data
        tweets = Tweet.query.filter(Tweet.category.in_(training_cats)).all()
        if embedding_form.d2v.data:  # d2v model
            # should be done in the background
            training_set = [TaggedDocument(simple_preprocess(tweet.text), [tweet.id]) for tweet in tweets]
            gensim_d2v = Doc2Vec(vector_size=vector_size, min_count=5, epochs=epochs)
            gensim_d2v.build_vocab(training_set)
            print("training the d2v model")
            gensim_d2v.train(training_set, total_examples=gensim_d2v.corpus_count, epochs=gensim_d2v.epochs)
            print("done")
            # save the model to the db
            new_id = max([model.id for model in D2VModel.query.all()]) + 1
            d2v = D2VModel(id=new_id, description=name)
            d2v.save(gensim_d2v)
            db.session.add(d2v)
            db.session.commit()
            flash("The model has been trained and saved to the database")
            # Redirect to the project page
            return redirect(url_for('project2', project=project_id))
        else:  # w2v only model
            training_set = [simple_preprocess(tweet.text) for tweet in tweets]
            #gensim_w2v = Word2Vec(vector_size=vector_size, min_count=5, epochs=epochs)
            #
        #return(redirect(url_for('project2', project=project_id)))

    return render_template('project2.html', title='About', project=project, analyses=analyses,
                           bayesian_form=bayesian_form, embedding_form=embedding_form)


@app.route("/project", methods=['GET', "POST"])
def project():
    project_id = request.args.get('project', None, type=int)
    project = Project.query.get(project_id)
    form = AddBayesianAnalysisForm()
    analyses = BayesianAnalysis.query.filter_by(user = current_user.id).filter_by(project=project_id).all()
    analyses = nlp4all.utils.get_user_project_analyses(current_user, project)
    if form.validate_on_submit():
        userid = current_user.id
        name = form.name.data
        number_per_category = form.number.data
        analysis_tweets = [] 
        if form.shared.data:
            tweets_by_cat = {cat : [t.id for t in project.tweets if t.category == cat.id] for cat in project.categories}
            for cat, tweets in tweets_by_cat.items():
                analysis_tweets.extend(sample(tweets, number_per_category))
        # make sure all students see tweets in the same order. So shuffle them now, and then 
        # put them in the database
        shuffle(analysis_tweets)
        analysis = BayesianAnalysis(user = userid, name=name, project=project.id, data = {"counts" : 0, "words" : {}}, shared=form.shared.data, tweets=analysis_tweets, annotation_tags={}, annotate=form.annotate.data )
        db.session.add(analysis)
        db.session.commit()
        return(redirect(url_for('project', project=project_id)))
    return render_template('project.html', title='About', project=project, analyses=analyses, form=form)


@app.route("/test", methods=['GET', 'POST'])
def test():
    a = ['1', '3', '4']
    return render_template('test.html', title='Test', test_var = a)


@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)


@app.route("/ajax")
def ajax():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    number = a + b
    return render_template('ajax.html', title='Test')


@app.route("/robot", methods=['GET', 'POST'])
def robot():
    # first get user's robot associated with
    robot_id = request.args.get('robot', 0, type=int)
    # find the analysis and check if it belongs to the user
    robot = BayesianRobot.query.get(robot_id)
    if robot.retired:
        acc_dict = robot.accuracy
        if robot.parent != None:
            parent_robot = BayesianRobot.query.get(robot.parent)
            parent_accuracy = parent_robot.accuracy
            acc_dict['parent_accuracy'] = parent_accuracy['accuracy']
            acc_dict['parent_tweets_targeted'] = parent_accuracy['tweets_targeted']
    else:
        acc_dict = {'features' : [], 'accuracy' : 0, 'tweets_targeted' : 0, 'features' : {}, 'table_data' : []}
    if request.method == "POST" and 'delete' in request.form.to_dict():
        del robot.features[request.form.to_dict()['delete']]
        flag_modified(robot, "features")
        db.session.add(robot)
        db.session.merge(robot)
        db.session.flush()
        db.session.commit()
        redirect(url_for('robot', robot=robot_id))
    form = BayesianRobotForms()
    if request.method == "POST" and 'add_feature_form-submit' in request.form.to_dict():
        if " " not in form.add_feature_form.data and len(form.add_feature_form.feature.data) > 3 and len(form.add_feature_form.reasoning.data) > 15 and len(robot.features) <= 20:
            new_feature = {form.add_feature_form.feature.data.strip() : form.add_feature_form.reasoning.data}
            robot.features.update(new_feature)
            flag_modified(robot, "features")
            db.session.add(robot)
            db.session.merge(robot)
            db.session.flush()
            db.session.commit()
            return redirect(url_for('robot', robot=robot_id))
    form = BayesianRobotForms()
    if request.method == "POST" and 'run_analysis_form-run_analysis' in request.form.to_dict():
        if len(robot.features) > 0:
            robot.accuracy = robot.calculate_accuracy()
            flag_modified(robot, "accuracy")
            db.session.merge(robot)
            robot.accuracy = BayesianRobot.calculate_accuracy(robot)
            robot.retired = True
            robot.time_retired = datetime.datetime.utcnow()
            child_robot = robot.clone()
            db.session.add(child_robot)
            db.session.flush()
            robot.child = child_robot.id
            db.session.add(robot)
            db.session.commit()
            return redirect(url_for('robot', robot = robot.id))
    return render_template('robot.html', title='Robot', r = robot, form = form, acc_dict=acc_dict)

@app.route("/shared_analysis_view", methods=['GET', 'POST'])
def shared_analysis_view():
    analysis_id = request.args.getq('analysis', 0, type=int)
    tweet_info = {}
    all_words = []
    if analysis_id != 0:
        analysis = BayesianAnalysis.query.get(analysis_id)
        # if not analysis.shared:
        #     return(redirect(url_for('home')))
        if analysis.shared:
            tweet_info = {t : {"correct": 0, "incorrect" : 0, "%" : 0} for t in analysis.tweets}
        else:
            tweet_info = {t.tweet : {"correct" : 0, "incorrect" : 0, "%" : 0} for t in analysis.tags}
        # for tag in analysis.tags:
        non_empty_tags = [t for t in analysis.tags if t.tweet != None]
        for tag in non_empty_tags:
            user = User.query.get(tag.user)
            tweet = Tweet.query.get(tag.tweet)
            all_words.extend(tweet.words)
            tweet_info[tweet.id]["full_text"] = tweet.full_text
            tweet_info[tweet.id]["category"] = TweetTagCategory.query.get(tweet.category).name
            if tweet.category == tag.category:
                tweet_info[tweet.id]["correct"]  = tweet_info[tweet.id]["correct"]   + 1
            else:
                tweet_info[tweet.id]["incorrect"]  = tweet_info[tweet.id]["incorrect"]   + 1
        for tweet_id in list(tweet_info.keys()):
            # if they haven't been categorized by anyone, remove them
            if tweet_info[tweet_id]["correct"] == 0 and tweet_info[tweet_id]["incorrect"] == 0:
                del tweet_info[ tweet_id ]
            else:
                tweet_info[tweet_id].update({"%": (tweet_info[tweet_id]["correct"] / (tweet_info[tweet_id]["incorrect"] + tweet_info[tweet_id]["correct"])) * 100})
    tweet_info = sorted([t for t in tweet_info.items()], key=lambda x:x[1]["%"], reverse=True)
    data={}
    percent_values = [d[1]["%"] for d in tweet_info]
    percent_counts = [{'label' : str(percent), 'estimate' : percent_values.count(percent)} for percent in set(percent_values)]
    for d in percent_counts:
        color = float(d['label']) / 100 * 120
        color = int(color)
        d.update({'color' : f"hsl({color}, 50%, 70%)", 'bg_color' : f"hsl({color}, 50%, 70%)"})
    chart_data = {'title' : 'Antal korrekte', 'data_points' : percent_counts}
    data['chart_data'] = chart_data
    # words = [word for x in tweet_info for word in x[1]["words"]]
    pred_by_word, data['predictions'] = analysis.get_predictions_and_words(all_words)
    # word_info = {word : {'predictions' : pred_by_word[word], 'counts' : words.count(word)} for word in set(words)}
    word_info = []
    for word in set(all_words):
        word_dict = {'word' : word, 'counts' : all_words.count(word)}
        for k,v in pred_by_word[word].items():
            word_dict[k] = v
        word_info.append(word_dict)
    tweet_info = [t[1] for t in tweet_info]
    # we don't need to sort this since we put it in a datatable anyway
    # print(word_info)
    # sorted_word_info = sorted([w for w in word_info], key=lambda x: x['counts'], reverse=True)
    return render_template('shared_analysis_view.html', title='Oversigt over analyse', tweets = tweet_info, word_info = word_info, analysis=analysis, **data)



@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/word")
def word():
    return render_template('word.html', title='Word examples')

@app.route("/analysis", methods=['GET', 'POST'])
def analysis():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    analysis_id = request.args.get('analysis', 1, type=int)
    analysis = BayesianAnalysis.query.get(analysis_id)
    project = Project.query.get(analysis.project)
    if 'tag' in request.form.to_dict():
        # category = TweetTagCategory.query.get(int(form.choices.data))
        tag_info = ast.literal_eval(request.form['tag'])
        tweet_id = tag_info[0]
        category_id = tag_info[1]
        the_tweet = Tweet.query.get(tweet_id)
        category = TweetTagCategory.query.get(category_id)
        the_tweet = Tweet.query.get(tweet_id)
        analysis.data = analysis.updated_data(the_tweet, category)
        ## all this  stuff is necessary  because the database backend doesnt resgister
        ## changes on JSON
        flag_modified(analysis, "data")
        db.session.add(analysis)
        db.session.merge(analysis)
        db.session.flush()
        db.session.commit()
        tag = TweetTag (category = category.id, analysis = analysis.id, tweet=the_tweet.id, user = current_user.id)
        db.session.add(tag)
        db.session.commit()
        # redirect(url_for('home'))
        return redirect(url_for('analysis', analysis=analysis_id))
    if len(analysis.robots) < 1: #TODO: move the creation of a robot to the creation of an analysis.
        robot = BayesianRobot(name =current_user.username + "s robot", analysis=analysis.id)
        db.session.add(robot)
        db.session.flush()
        db.session.commit()
        return redirect(url_for('analysis', analysis=analysis_id))
    ## check if user has access to this.
    ## either it is a shared project. In that case the user needs to be a member of 
    # projects.organization
    # or it is not a shared project, in which case the user must be the owner.
    # owned = False
    # if analysis.shared :
    #     if project.organization in [org.id for org in current_user.organizations]:
    #         owned = True
    # if analysis.user == current_user.id or current_user.admin:#or current_user.
    #     owned = True
    # if owned == False:
    #     return redirect(url_for('home'))
    categories = TweetTagCategory.query.filter(TweetTagCategory.id.in_([p.id for p in project.categories])).all() # TODO: pretty sure we can just get project.categories
    tweets = project.tweets
    the_tweet = None
    if analysis.shared:
        completed_tweets = [t.tweet for t in analysis.tags if t.user == current_user.id]
        uncompleted_tweets = [t for t in analysis.tweets if t not in completed_tweets]
        if len(uncompleted_tweets) > 0:
            the_tweet_id = uncompleted_tweets[0]
            the_tweet = Tweet.query.get(the_tweet_id)
        else:
            flash('Du er kommet igennem alle tweetsene. Vent på resten af klassen nu :)', 'success')
            the_tweet = Tweet(full_text = "", words = [])
    else:
        the_tweet = sample(tweets, 1)[0]
    form = TaggingForm()
    form.choices.choices  = [( str(c.id), c.name ) for c in categories]
    number_of_tagged = len(analysis.tags)
    data = {}
    data['number_of_tagged']  = number_of_tagged
    data['words'], data['predictions'] = analysis.get_predictions_and_words(set(the_tweet.words))
    data['word_tuples'] = nlp4all.utils.create_css_info(data['words'], the_tweet.full_text, categories)
    data['chart_data'] = nlp4all.utils.create_bar_chart_data(data['predictions'], "Computeren gætter på...")
    # filter robots that are retired, and sort them alphabetically
    # data['robots'] = sorted(robots, key= lambda r: r.name)
    data['analysis_data'] = analysis.data
    data['user'] = current_user
    data['user_role'] = current_user.roles
    data['tag_options'] = project.categories

    if form.validate_on_submit() and form.data:
        category = TweetTagCategory.query.get(int(form.choices.data))
        analysis.data = analysis.updated_data(the_tweet, category)
        ## all this  stuff is necessary  because the database backend doesnt resgister
        ## changes on JSON
        flag_modified(analysis, "data")
        db.session.add(analysis)
        db.session.merge(analysis)
        db.session.flush()
        db.session.commit()
        tag = TweetTag (category = category.id, analysis = analysis.id, tweet=the_tweet.id, user = current_user.id)
        db.session.add(tag)
        db.session.commit()
        # redirect(url_for('home'))
        return redirect(url_for('analysis', analysis=analysis_id))
    # tags per user
    ann_tags = TweetAnnotation.query.filter(TweetAnnotation.user==current_user.id).all()
    tag_list = list(set([a.annotation_tag for a in ann_tags]))
    for i in categories:
        if i.name not in tag_list:
            tag_list.append(i.name)
    return render_template('analysis.html', analysis=analysis, tag_list=tag_list, tweet = the_tweet, form = form, **data)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    form.organizations.choices = [(str(o.id), o.name) for o in Organization.query.all()]
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        org = Organization.query.get(int(form.organizations.data))
        user = User(username=form.username.data, email=form.email.data, password=hashed_password, organizations=[org])
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/bar_chart")
def bar_chart():
    values = [0, 1, 2, 3, 4, 5]
    labels = ['0', '1', '2', '3', '4', '5']
    return render_template('bar_chart.html', title="test chart", values=values, labels=labels)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account',
                           image_file=image_file, form=form)

@app.route("/manage_categories", methods=['GET', 'POST'])
@login_required
def manage_categories():
    form = AddTweetCategoryForm()
    categories = [cat.name for cat in TweetTagCategory.query.all()]
    if form.validate_on_submit():
        nlp4all.utils.add_tweets_from_account(form.twitter_handle.data)
        flash('Added tweets from the twitter handle', 'success')
        return redirect(url_for('manage_categories'))
    return render_template('manage_categories.html', form=form, categories=categories)

@app.route("/add_org", methods=['GET', 'POST'])
@login_required
def add_org():
    form = AddOrgForm()
    orgs = Organization.query.all()
    if form.validate_on_submit():
        org = Organization(name=form.name.data)
        db.session.add(org)
        db.session.commit()
        flash('Your organization has been created!', 'success')
        return redirect(url_for('add_org'))
    return render_template('add_org.html', form=form, orgs=orgs)


@app.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data, content=form.content.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('home'))
    return render_template('create_post.html', title='New Post',
                           form=form, legend='New Post')


@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)


@app.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post',
                           form=form, legend='Update Post')


@app.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('home'))


# @app.route("/user/<string:username>")
# def user_posts(username):
#     page = request.args.get('page', 1, type=int)
#     user = User.query.filter_by(username=username).first_or_404()
#     posts = Post.query.filter_by(author=user)\
#         .order_by(Post.date_posted.desc())\
#         .paginate(page=page, per_page=5)
#     return render_template('user_posts.html', posts=posts, user=user)


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)


@app.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title='Reset Password', form=form)


@app.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title='Reset Password', form=form)


@app.route("/create_matrix", methods=['GET', 'POST'])
def create_matrix():
    form = CreateMatrixForm()
    form.categories.choices = [( str(s.id), s.name ) for s in TweetTagCategory.query.all()]
   
    if form.validate_on_submit():
        cats = [int(n) for n in form.categories.data]
        tweets = Tweet.query.filter(Tweet.category.in_(cats)).all()
        ratio = form.ratio.data
        matrix = nlp4all.utils.add_matrix(cat_ids=cats, ratio=ratio)
    
        db.session.add(matrix)
        db.session.commit()
        return(redirect(url_for('create_matrix')))
    return render_template('create_matrix.html', form=form)

@app.route("/matrix/<matrix_id>", methods=['GET', 'POST'])
@login_required
def matrix(matrix_id):
    userid = current_user.id
    matrix = ConfusionMatrix.query.get(matrix_id)
    matrices = ConfusionMatrix.query.filter(ConfusionMatrix.user== userid).all() # all matrices for the other tabs
    all_cats = TweetTagCategory.query.all() # all cats for the other tabs

    categories = matrix.categories
    cat_names = [c.name for c in categories]
    form = ThresholdForm()
    # get a tnt_set
    tnt_sets = matrix.training_and_test_sets
    if "tnt_nr" in request.args.to_dict().keys():
            tnt_nr = request.args.get('tnt_nr', type=int)
            a_tnt_set = tnt_sets[tnt_nr]
    else:
        a_tnt_set = tnt_sets[0]
        tnt_nr = 0

    train_tweet_ids = a_tnt_set[0].keys()
    train_set_size = len(a_tnt_set[0].keys())
    test_tweets = [Tweet.query.get(tweet_id) for tweet_id in a_tnt_set[1].keys()]

    if form.validate_on_submit():
        if form.threshold.data:
            matrix.threshold = form.threshold.data
            flag_modified(matrix, "threshold")
        if form.ratio.data:
            matrix.ratio = round(form.ratio.data * 0.01,3)
            flag_modified(matrix, "ratio")
            matrix.training_and_test_sets = matrix.update_tnt_set()
            flag_modified(matrix, "training_and_test_sets")
            db.session.add(matrix)
            db.session.merge(matrix)
            db.session.flush()
            db.session.commit()
        if form.shuffle.data:
            tnt_list = list(range(0, len(tnt_sets)))
            tnt_nr = sample(tnt_list, 1)[0]
            a_tnt_set = tnt_sets[tnt_nr] # tnt_set id
            train_tweet_ids = a_tnt_set[0].keys()
            train_set_size = len(a_tnt_set[0].keys())
            test_tweets = [Tweet.query.get(tweet_id) for tweet_id in a_tnt_set[1].keys()]

            # train on the training set:
            matrix.train_data = matrix.train_model(train_tweet_ids)
            flag_modified(matrix, "train_data")

            # make matrix data
            matrix_data = matrix.make_matrix_data(test_tweets, cat_names)
            matrix.matrix_data = {i[0]: i[1] for i in matrix_data}
            flag_modified(matrix, "matrix_data")

            # filter according to the threshold
            incl_tweets = sorted([t for t in matrix.matrix_data.items() if t[1]['probability'] >= matrix.threshold and t[1]['class'] != 'undefined'], key=lambda x:x[1]["probability"], reverse=True)
            excl_tweets = sorted([t for t in matrix.matrix_data.items() if t[1]['probability'] < matrix.threshold or t[1]['class'] == 'undefined'], key=lambda x:x[1]["probability"], reverse=True)

            # count different occurences
            class_list = [t[1]['class'] for t in incl_tweets]
            class_list_all = [str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c-1])) for c in range(len(cat_names))]+[str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c])) for c in range(len(cat_names))]
            matrix_classes = {c:0 for c in class_list_all} 
            for i in set(class_list):
                matrix_classes[i] = class_list.count(i)

            true_keys = [str("Pred_"+i+"_Real_"+i) for i in cat_names]
            True_dict = dict(filter(lambda item: item[0] in true_keys, matrix_classes.items()))
            
            accuracy = round((sum(True_dict.values()) / sum(matrix_classes.values())), 3)
            metrics = {i: {'category':i, 'recall': 0, 'precision':0} for i in cat_names}
            for i in cat_names:
                selected_cat = i
                tp_key = str("Pred_"+selected_cat+"_Real_"+selected_cat)
                recall_keys = [str("Pred_"+selected_cat+"_Real_"+i) for i in cat_names]
                if sum([matrix_classes [x] for x in recall_keys]) >0:
                    metrics[i]['recall'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in recall_keys]),2)
                
                precision_keys = [str("Pred_"+i+"_Real_"+selected_cat) for i in cat_names]
                if sum([matrix_classes [x] for x in precision_keys]) > 0:
                    metrics[i]['precision'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in precision_keys]),2)

            # summarise data
            matrix.data = {'matrix_classes' : matrix_classes,'accuracy':accuracy, 'metrics':metrics, 'nr_test_tweets': len(test_tweets), 'nr_train_tweets': train_set_size, 'nr_incl_tweets':len(incl_tweets), 'nr_excl_tweets': len(excl_tweets)}
            flag_modified(matrix, "data")
            db.session.add(matrix)
            db.session.merge(matrix)
            db.session.flush()
            db.session.commit()
            return redirect(url_for('matrix', matrix_id=matrix.id, tnt_nr= tnt_nr)) 
    
        # train on the training set:
        matrix.train_data = matrix.train_model(train_tweet_ids)
        flag_modified(matrix, "train_data")

        # make matrix data
        matrix_data = matrix.make_matrix_data(test_tweets, cat_names)
        matrix.matrix_data = {i[0]: i[1] for i in matrix_data}
        flag_modified(matrix, "matrix_data")

        # filter according to the threshold
        incl_tweets = sorted([t for t in matrix.matrix_data.items() if t[1]['probability'] >= matrix.threshold and t[1]['class'] != 'undefined'], key=lambda x:x[1]["probability"], reverse=True)
        excl_tweets = sorted([t for t in matrix.matrix_data.items() if t[1]['probability'] < matrix.threshold or t[1]['class'] == 'undefined'], key=lambda x:x[1]["probability"], reverse=True)

        # count different occurences
        class_list = [t[1]['class'] for t in incl_tweets]
        class_list_all = [str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c-1])) for c in range(len(cat_names))]+[str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c])) for c in range(len(cat_names))]
        matrix_classes = {c:0 for c in class_list_all} 
        for i in set(class_list):
            matrix_classes[i] = class_list.count(i)

        true_keys = [str("Pred_"+i+"_Real_"+i) for i in cat_names]
        True_dict = dict(filter(lambda item: item[0] in true_keys, matrix_classes.items()))
        
        # accuracy = sum(correct predictions)/sum(all matrix points)
        accuracy = round((sum(True_dict.values()) / sum(matrix_classes.values())), 3)
        metrics = {i: {'category':i, 'recall': 0, 'precision':0} for i in cat_names}
        for i in cat_names:
            selected_cat = i
            tp_key = str("Pred_"+selected_cat+"_Real_"+selected_cat)
            recall_keys = [str("Pred_"+selected_cat+"_Real_"+i) for i in cat_names]
            if sum([matrix_classes [x] for x in recall_keys]) >0:
                metrics[i]['recall'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in recall_keys]),2)
            
            precision_keys = [str("Pred_"+i+"_Real_"+selected_cat) for i in cat_names]
            if sum([matrix_classes [x] for x in precision_keys]) > 0:
                metrics[i]['precision'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in precision_keys]),2)

        # summarise data
        matrix.data = {'matrix_classes' : matrix_classes,'accuracy':accuracy,  'metrics':metrics,'nr_test_tweets': len(test_tweets), 'nr_train_tweets': train_set_size, 'nr_incl_tweets':len(incl_tweets), 'nr_excl_tweets': len(excl_tweets)}
        flag_modified(matrix, "data")
        db.session.add(matrix)
        db.session.merge(matrix)
        db.session.flush()
        db.session.commit()
        return redirect(url_for('matrix', matrix_id=matrix.id, tnt_nr= tnt_nr))
        
    # prepare data for matrix table
    counts = matrix.make_table_data(cat_names)
    [counts[i].insert(0, cat_names[i]) for i in range(len(counts))]
    index_list = []
    for i in range(len(counts)):
        p = cat_names[i]
        t = [str('Pred_'+counts[i][0]+'_Real_'+p) for i in range(len(counts))]
        index_list.append(t)
    [index_list[i].insert(0, cat_names[i]) for i in range(len(index_list))]
    index_list =[[[counts[j][i], index_list[j][i], (j,i)] for i in range(0, len(counts[j]))] for j in range(len(counts))]
    index_list = nlp4all.utils.matrix_css_info(index_list)

    metrics = sorted([t for t in matrix.data['metrics'].items()], key=lambda x:x[1]["recall"], reverse=True)
    metrics = [t[1] for t in metrics]
    return render_template('matrix.html', cat_names = cat_names, form=form, matrix=matrix, all_cats= all_cats, matrices=matrices, index_list=index_list, metrics=metrics)
  
@app.route("/matrix_tweets/<matrix_id>", methods=['GET', 'POST'])
@login_required
def matrix_tweets(matrix_id):
    matrix = ConfusionMatrix.query.get(matrix_id)
    # request tweets from the correct quadrant
    cm = request.args.get('cm', type=str) 
    title = str("Tweets classified as " + cm)
    if cm in [c.name for c in matrix.categories]:
        id_c = [{int(k):{'probability':v['probability'], 'relative probability':v['relative probability'], 'prediction' :v['pred_cat']} for k, v in matrix.matrix_data.items() if v['real_cat'] == cm and v['probability'] >= matrix.threshold}][0]
        tweets = Tweet.query.filter(Tweet.id.in_(id_c.keys())).all()
    else:
        id_c = [{int(k):{'probability':v['probability'], 'relative probability':v['relative probability'],'prediction' :v['pred_cat']} for k, v in matrix.matrix_data.items() if v['class'] == cm and v['probability'] >= matrix.threshold}][0]
        tweets = Tweet.query.filter(Tweet.id.in_(id_c.keys())).all()
    
    cm_info = { t.id : {'text' : t.full_text, 'category': t.handle,'prediction': id_c[t.id]["prediction"],'probability' : round(id_c[t.id]['probability'], 3),'relative probability' : round(id_c[t.id]['relative probability'], 3)} for t in tweets}
    cm_info = sorted([t for t in cm_info.items()], key=lambda x:x[1]["probability"], reverse=True)
    cm_info = [t[1] for t in cm_info]
    return render_template('matrix_tweets.html', cm_info = cm_info, matrix=matrix, title=title)

@app.route("/my_matrices", methods=['GET', 'POST'])
@login_required
def my_matrices():
    userid = current_user.id
    matrices = ConfusionMatrix.query.filter(ConfusionMatrix.user== userid).all()

    form = CreateMatrixForm()
    form.categories.choices = [( str(s.id), s.name ) for s in TweetTagCategory.query.all()]
   
    if form.validate_on_submit():
        userid = current_user.id
        cats = [int(n) for n in form.categories.data]
        tweets = Tweet.query.filter(Tweet.category.in_(cats)).all()
        ratio = form.ratio.data*0.01 # convert back to decimals
        matrix = nlp4all.utils.add_matrix(cat_ids=cats, ratio=ratio, userid = userid)
        db.session.add(matrix)
        db.session.merge(matrix)
        db.session.flush()
        db.session.commit() # not sure if this is necessary

        cat_names = [c.name for c in matrix.categories]
        a_tnt_set = matrix.training_and_test_sets[0] # as a default
        train_tweet_ids = a_tnt_set[0].keys()
        train_set_size = len(a_tnt_set[0].keys())
        test_tweets = [Tweet.query.get(tweet_id) for tweet_id in a_tnt_set[1].keys()]

        # train on the training set:
        matrix.train_data = matrix.train_model(train_tweet_ids)
        flag_modified(matrix, "train_data")
    
        # make matrix data
        matrix_data = matrix.make_matrix_data(test_tweets, cat_names)
        matrix.matrix_data = {i[0]: i[1] for i in matrix_data}
        flag_modified(matrix, "matrix_data")

        # filter according to the threshold
        incl_tweets = sorted([t for t in matrix.matrix_data.items() if t[1]['probability'] >= matrix.threshold and t[1]['class'] != 'undefined'], key=lambda x:x[1]["probability"], reverse=True)
        excl_tweets = sorted([t for t in matrix.matrix_data.items() if t[1]['probability'] < matrix.threshold or t[1]['class'] == 'undefined'], key=lambda x:x[1]["probability"], reverse=True)

        # count different occurences
        class_list = [t[1]['class'] for t in incl_tweets]
        class_list_all = [str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c-1])) for c in range(len(cat_names))]+[str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c])) for c in range(len(cat_names))]
        matrix_classes = {c:0 for c in class_list_all} 
        for i in set(class_list):
            matrix_classes[i] = class_list.count(i)

        true_keys = [str("Pred_"+i+"_Real_"+i) for i in cat_names]
        True_dict = dict(filter(lambda item: item[0] in true_keys, matrix_classes.items()))
        
        # accuracy = sum(correct predictions)/sum(all matrix points)
        accuracy = round((sum(True_dict.values()) / sum(matrix_classes.values())), 3)
        metrics = {i: {'category':i, 'recall': 0, 'precision':0} for i in cat_names}
        for i in cat_names:
            selected_cat = i
            tp_key = str("Pred_"+selected_cat+"_Real_"+selected_cat)
            recall_keys = [str("Pred_"+selected_cat+"_Real_"+i) for i in cat_names]
            if sum([matrix_classes [x] for x in recall_keys]) >0:
                metrics[i]['recall'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in recall_keys]),2)
            
            precision_keys = [str("Pred_"+i+"_Real_"+selected_cat) for i in cat_names]
            if sum([matrix_classes [x] for x in precision_keys]) > 0:
                metrics[i]['precision'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in precision_keys]),2)

        
        # summarise data
        matrix.data = {'matrix_classes' : matrix_classes,'accuracy':accuracy, 'metrics':metrics ,'nr_test_tweets': len(test_tweets), 'nr_train_tweets': train_set_size, 'nr_incl_tweets':len(incl_tweets), 'nr_excl_tweets': len(excl_tweets)}
        flag_modified(matrix, "data")
        db.session.add(matrix)
        db.session.merge(matrix)
        db.session.flush()
        db.session.commit()
        return(redirect(url_for('my_matrices')))

    return render_template('my_matrices.html', matrices=matrices, form=form) 

@app.route("/included_tweets/<matrix_id>", methods=['GET', 'POST'])
@login_required
def included_tweets(matrix_id):
    matrix = ConfusionMatrix.query.get(matrix_id)
    title = "Included tweets"
    # filter according to the threshold
    id_c = [{int(k):{'probability':v['probability'], 'relative probability':v['relative probability'],'pred_cat':v['pred_cat'],  'class' : v['class']} for k, v in matrix.matrix_data.items() if v['probability'] >= matrix.threshold and v['class'] != 'undefined'}][0]
    
    tweets = Tweet.query.filter(Tweet.id.in_(id_c.keys())).all()
    # collect necessary data for the table
    cm_info = { t.id : {'text' : t.full_text, 'category': t.handle, 'predicted category': id_c[t.id]['pred_cat'], 'probability' : round(id_c[t.id]['probability'],3),'relative probability' : id_c[t.id]['relative probability'] } for t in tweets}
    
    cm_info = sorted([t for t in cm_info.items()], key=lambda x:x[1]["probability"], reverse=True)
    cm_info = [t[1] for t in cm_info]
    return render_template('matrix_tweets.html', cm_info = cm_info, matrix=matrix, title = title)

@app.route("/excluded_tweets/<matrix_id>", methods=['GET', 'POST'])
@login_required
def excluded_tweets(matrix_id):
    matrix = ConfusionMatrix.query.get(matrix_id)
    title = 'Excluded tweets'
    # filter according to the threshold (or if prediction is undefined)
    id_c = [{int(k):{'probability':v['probability'],'relative probability':v['relative probability'], 'pred_cat':v['pred_cat'],  'class' : v['class']} for k, v in matrix.matrix_data.items() if v['probability'] < matrix.threshold or v['class'] == 'undefined'}][0]
    
    tweets = Tweet.query.filter(Tweet.id.in_(id_c.keys())).all()
    # collect necessary data for the table
    cm_info = { t.id : {'text' : t.full_text, 'category': t.handle, 'predicted category': id_c[t.id]['pred_cat'], 'probability' : round(id_c[t.id]['probability'],3), 'relative probability' : id_c[t.id]['relative probability'] } for t in tweets}
    
    cm_info = sorted([t for t in cm_info.items()], key=lambda x:x[1]["probability"], reverse=True)
    cm_info = [t[1] for t in cm_info]
    return render_template('matrix_tweets.html', cm_info = cm_info, matrix=matrix, title = title)


@app.route("/matrix_overview", methods=['GET', 'POST'])
@login_required
def matrix_overview():
    userid = current_user.id # get matrices for the user
    matrices = ConfusionMatrix.query.filter(ConfusionMatrix.user== userid).all()
    all_cats = TweetTagCategory.query.all()
    
    matrix_info = {m.id : {"accuracy": m.data['accuracy'],"threshold" : m.threshold, "ratio" : m.ratio, "categories" : [', '.join([c.name for c in m.categories][0 : ])][0], "excluded tweets (%)" : round(m.data["nr_excl_tweets"]/m.data["nr_test_tweets"]*100,3) } for m in matrices}
    matrix_info = sorted([t for t in matrix_info.items()], key=lambda x:x[1]["accuracy"], reverse=True)
    matrix_info = [m[1] for m in matrix_info]
    form = ThresholdForm()
    return render_template('matrix_overview.html', matrices=matrices, matrix_info=matrix_info, form = form, userid=userid, all_cats=all_cats)


@app.route('/matrix_loop', methods=['POST','GET'])
def matrix_loop():
    matrix_id = request.args.get('matrix_id')
    matrix = ConfusionMatrix.query.get(matrix_id) #so the id should be always specified in the url
    cat_names = [n.name for n in matrix.categories]
    
    return render_template('matrix_table_base.html', matrix=matrix, cat_names=cat_names)


@app.route('/get_aggregated_data', methods=['GET', 'POST'])
def aggregate_matrix():
    args = request.args.to_dict()
    m_id = args['matrix_id']
    matrix = ConfusionMatrix.query.get(int(m_id))
    
    if "n" in request.args.to_dict().keys():
        n = request.args.get('n', type=int)
    else:
        n=3

    used_tnt_sets = [] # log used tnt sets
    agg_data = {m:{'data':{}} for m in range(n)}
    accuracy_list = []
    metrics_list =[]
    list_excluded = []
    list_included = []
    counts_list = []
    cat_names = [n.name for n in matrix.categories]
    
    # loop
    for m in range(n):
        new_mx = matrix.clone()
        db.session.add(new_mx)
        db.session.flush()
        db.session.commit()
        tnt_sets = new_mx.training_and_test_sets
        # select a new
        tnt_list = [x for x in list(range(0, len(matrix.training_and_test_sets))) if x not in used_tnt_sets]

        tnt_nr = sample(tnt_list, 1)[0]
        #print(tnt_nr)
        used_tnt_sets.append(tnt_nr) # log used sets
        a_tnt_set = tnt_sets[tnt_nr]
        train_tweet_ids = a_tnt_set[0].keys()
        train_set_size = len(a_tnt_set[0].keys())
        test_tweets = [Tweet.query.get(tweet_id) for tweet_id in a_tnt_set[1].keys()]

        # train on the training set:
        new_mx.train_data = new_mx.train_model(train_tweet_ids)
        flag_modified(new_mx, "train_data")
        db.session.flush()
        
        # make matrix data
        matrix_data = new_mx.make_matrix_data(test_tweets, cat_names)
        new_mx.matrix_data = {i[0]: i[1] for i in matrix_data}
    
        flag_modified(matrix, "matrix_data")
        db.session.flush()

        # filter according to the threshold
        incl_tweets = sorted([t for t in new_mx.matrix_data.items() if t[1]['probability'] >= matrix.threshold and t[1]['class'] != 'undefined'], key=lambda x:x[1]["probability"], reverse=True)
        excl_tweets = sorted([t for t in new_mx.matrix_data.items() if t[1]['probability'] < matrix.threshold or t[1]['class'] == 'undefined'], key=lambda x:x[1]["probability"], reverse=True)

        list_excluded.append(len(excl_tweets))
        list_included.append(len(incl_tweets))
        
        # count different occurences
        class_list = [t[1]['class'] for t in incl_tweets]
        class_list_all = [str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c-1])) for c in range(len(cat_names))]+[str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c])) for c in range(len(cat_names))]
        matrix_classes = {c:0 for c in class_list_all} 
        for i in set(class_list):
            matrix_classes[i] = class_list.count(i)

        true_keys = [str("Pred_"+i+"_Real_"+i) for i in cat_names]
        True_dict = dict(filter(lambda item: item[0] in true_keys, matrix_classes.items()))
        
        # accuracy = sum(correct predictions)/sum(all matrix points)
        accuracy = round((sum(True_dict.values()) / sum(matrix_classes.values())), 3)
        accuracy_list.append(accuracy)
        metrics = {i: {'category':i, 'recall': 0, 'precision':0} for i in cat_names}
        for i in cat_names:
            selected_cat = i
            tp_key = str("Pred_"+selected_cat+"_Real_"+selected_cat)
            recall_keys = [str("Pred_"+selected_cat+"_Real_"+i) for i in cat_names]
            if sum([matrix_classes [x] for x in recall_keys]) >0:
                metrics[i]['recall'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in recall_keys]),2)
            
            precision_keys = [str("Pred_"+i+"_Real_"+selected_cat) for i in cat_names]
            if sum([matrix_classes [x] for x in precision_keys]) > 0:
                metrics[i]['precision'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in precision_keys]),2)

        # summarise data
        new_mx.data = {'matrix_classes' : matrix_classes,'accuracy':accuracy,  'metrics':metrics,'nr_test_tweets': len(test_tweets), 'nr_train_tweets': train_set_size, 'nr_incl_tweets':len(incl_tweets), 'nr_excl_tweets': len(excl_tweets)}
        flag_modified(new_mx, "data")
        db.session.add(new_mx)
        db.session.merge(new_mx)
        db.session.flush()
        db.session.commit()
        metrix = new_mx.data['metrics'].items()
        metrix = sorted([t for t in new_mx.data['metrics'].items()], key=lambda x:x[1]["recall"], reverse=True)
        metrix = [t[1] for t in metrix]
        metrics_list.append(metrix)
        agg_data[m]['data'] = new_mx.data
        # build matrix data
        currentDataClass = [new_mx.matrix_data[i].get('real_cat') for i in new_mx.matrix_data.keys()]
        predictedClass = [new_mx.matrix_data[i].get('pred_cat') for i in new_mx.matrix_data.keys()]
        number_list= list(range(len(cat_names)))

        for i in number_list:
            for o in range(len(currentDataClass)):
                if currentDataClass[o] == cat_names[i]:
                    currentDataClass[o] = i+1
        for i in number_list:
            for p in range(len(predictedClass)):
                if predictedClass[p] == cat_names[i]:
                    predictedClass[p] = i+1
        classes = int(max(currentDataClass) - min(currentDataClass)) + 1 #find number of classes
        counts = [[sum([(currentDataClass[i] == true_class) and (predictedClass[i] == pred_class) 
                        for i in range(len(currentDataClass))])
                for pred_class in range(1, classes + 1)] 
                for true_class in range(1, classes + 1)]
        counts_list.append(counts)
        
    # accuracy, excluded, included
    averages = [round(sum(accuracy_list)/len(accuracy_list),3), round(sum(list_included)/len(list_included),2), round(sum(list_excluded)/n,2)]
    avg_metrix = [[metrics_list[0][0]['category'],round(sum(i[0]['recall'] for i in metrics_list)/len(metrics_list),3),round(sum(i[0]['precision'] for i in metrics_list)/len(metrics_list),3)], [metrics_list[0][1]['category'],round(sum(i[1]['recall'] for i in metrics_list)/len(metrics_list),3), round(sum(i[1]['precision'] for i in metrics_list)/len(metrics_list),3)]]
    # quadrants
    avg_quadrants = {}
    quadrants = [agg_data[m]["data"]['matrix_classes'] for m in agg_data]
    for dictionary in quadrants:
        for key, value in dictionary.items():
            if key in avg_quadrants.keys():
                avg_quadrants[key] = value + avg_quadrants[key]
            else:
                avg_quadrants[key] = value
    avg_quadrants = [round(m/n,3) for m in avg_quadrants.values()]
    # get info from each iteration to show how it varies
    loop_table = [[i+1, accuracy_list[i], list_included[i], list_excluded[i],list(metrics_list[i][0].values())[1:],list(metrics_list[i][1].values())[1:]] for i in range(n)]
    count_sum = [[[counts_list[j][l][i] + counts_list[j][l][i] for i in range(len(counts_list[0]))] 
        for l in range(len(counts_list[0]))]
        for j in range(len(counts_list))][0]
    matrix_values = [[round(count_sum[i][j]/len(counts_list),2) for j in range(len(count_sum[i]))] for i in range(len(count_sum))]
    [matrix_values[i].insert(0, cat_names[i]) for i in range(len(matrix_values))]
    # add cell indices
    for i in range(len(matrix_values)):
        t=0
        for j in range(len(matrix_values[i])):
            matrix_values[i][j] = [matrix_values[i][j],(i,0+t)]
            t += 1
    matrix_values = nlp4all.utils.matrix_css_info(matrix_values)
    return jsonify(avg_quadrants, averages, n, loop_table, matrix_values, avg_metrix)

@app.route('/get_matrix_categories', methods=['GET', 'POST'])
def get_matrix_categories():
    args = request.args.to_dict()
    m_id = args['matrix_id']
    matrix = ConfusionMatrix.query.get(int(m_id))
    
    cat_ids = [c.id for c in matrix.categories]
    cat_names = [c.name for c in matrix.categories]
    all_cats = TweetTagCategory.query.all()
    new_cats = [[i.id, i.name] for i in all_cats if i.id not in cat_ids]
    return jsonify(cat_ids, cat_names, new_cats)


@app.route('/get_compare_matrix_data', methods=['GET', 'POST'])
def get_compare_matrix_data():
    args = request.args.to_dict()
    m_id = args['matrix_id']
    alt_cat = args['alt_cat']
    new_cat = args['new_cat']
    matrix = ConfusionMatrix.query.get(int(m_id))
    alt_cat = TweetTagCategory.query.get(int(alt_cat))
    new_cat = TweetTagCategory.query.get(int(new_cat))
    old_cats = [c.id for c in matrix.categories]
    cat_ids = [new_cat.id if x==alt_cat.id else x for x in old_cats]

    # create a new matrix
    matrix2 = nlp4all.utils.add_matrix(cat_ids, ratio= matrix.ratio, userid='')
    matrix2.threshold = matrix.threshold
    flag_modified(matrix2, "threshold") 
    db.session.add(matrix2)
    db.session.merge(matrix2)
    db.session.flush()
    db.session.commit() # not sure if this is necessary

    tnt_sets = matrix2.training_and_test_sets
    
    # select a new
    tnt_list = [x for x in list(range(0, len(matrix2.training_and_test_sets))) ]
    tnt_nr = sample(tnt_list, 1)[0]
    a_tnt_set = tnt_sets[tnt_nr]
    train_tweet_ids = a_tnt_set[0].keys()
    train_set_size = len(a_tnt_set[0].keys())
    test_tweets = [Tweet.query.get(tweet_id) for tweet_id in a_tnt_set[1].keys()]

    # train on the training set:
    matrix2.train_data = matrix2.train_model(train_tweet_ids)
    flag_modified(matrix2, "train_data")
   
    # make matrix data
    cat_names = [c.name for c in matrix2.categories]
    matrix_data = matrix2.make_matrix_data(test_tweets, cat_names)
    matrix2.matrix_data = {i[0]: i[1] for i in matrix_data}
    flag_modified(matrix2, "matrix_data")

    # filter according to the threshold
    incl_tweets = sorted([t for t in matrix2.matrix_data.items() if t[1]['probability'] >= matrix.threshold and t[1]['class'] != 'undefined'], key=lambda x:x[1]["probability"], reverse=True)
    excl_tweets = sorted([t for t in matrix2.matrix_data.items() if t[1]['probability'] < matrix.threshold or t[1]['class'] == 'undefined'], key=lambda x:x[1]["probability"], reverse=True)
    
    # count different occurences
    class_list = [t[1]['class'] for t in incl_tweets]
    class_list_all = [str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c-1])) for c in range(len(cat_names))]+[str('Pred_'+str(cat_names[c])+"_Real_"+str(cat_names[c])) for c in range(len(cat_names))]
    matrix_classes = {c:0 for c in class_list_all} 
    for i in set(class_list):
        matrix_classes[i] = class_list.count(i)

    true_keys = [str("Pred_"+i+"_Real_"+i) for i in cat_names]
    True_dict = dict(filter(lambda item: item[0] in true_keys, matrix_classes.items()))
    
    # accuracy = sum(correct predictions)/sum(all matrix points)
    accuracy = round((sum(True_dict.values()) / sum(matrix_classes.values())), 3)
    metrics = {i: {'category':i, 'recall': 0, 'precision':0} for i in cat_names}
    for i in cat_names:
        selected_cat = i
        tp_key = str("Pred_"+selected_cat+"_Real_"+selected_cat)
        recall_keys = [str("Pred_"+selected_cat+"_Real_"+i) for i in cat_names]
        if sum([matrix_classes [x] for x in recall_keys]) >0:
            metrics[i]['recall'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in recall_keys]),2)
        
        precision_keys = [str("Pred_"+i+"_Real_"+selected_cat) for i in cat_names]
        if sum([matrix_classes [x] for x in precision_keys]) > 0:
            metrics[i]['precision'] = round(matrix_classes[tp_key] / sum([matrix_classes [x] for x in precision_keys]),2)

    # summarise data
    matrix2.data = {'matrix_classes' : matrix_classes,'accuracy':accuracy,  'metrics':metrics,'nr_test_tweets': len(test_tweets), 'nr_train_tweets': train_set_size, 'nr_incl_tweets':len(incl_tweets), 'nr_excl_tweets': len(excl_tweets)}
    flag_modified(matrix2, "data")
    db.session.add(matrix2)
    db.session.merge(matrix2)
    db.session.flush()
    db.session.commit()

    # prepare data for matrix table
    old_names =[c.name for c in matrix.categories]
    counts1 = matrix.make_table_data(old_names)
    counts2 = matrix2.make_table_data(cat_names)

    [counts1[i].insert(0, old_names[i]) for i in range(len(counts1))]
    [counts2[i].insert(0, cat_names[i]) for i in range(len(counts2))]
    for i in range(len(counts1)):
        t=0
        for j in range(len(counts1[i])):
            counts1[i][j] = [counts1[i][j],(i,0+t)]
            t += 1
    for i in range(len(counts2)):
        t=0
        for j in range(len(counts2[i])):
            counts2[i][j] = [counts2[i][j],(i,0+t)]
            t += 1
    
    table_data = [[m.id, m.data['accuracy'],  m.data['nr_incl_tweets'], m.data['nr_excl_tweets']] for m in [matrix, matrix2]]
    counts1=nlp4all.utils.matrix_css_info(counts1)
    counts2=nlp4all.utils.matrix_css_info(counts2)
    metrics= [list(matrix.data['metrics'][cat].values()) for cat in old_names ] , [list(matrix2.data['metrics'][cat].values()) for cat in cat_names ]

    return jsonify(counts1, counts2, matrix.threshold, matrix.ratio, table_data,metrics)

@app.route("/compare_matrices", methods=['GET', 'POST'])
@login_required
def compare_matrices():
    # there you can try out the comparison templates
    userid = current_user.id
    all_cats = TweetTagCategory.query.all()
    matrices = ConfusionMatrix.query.filter(ConfusionMatrix.user== userid).all()
    cat_names = [c.name for c in all_cats]

    return render_template('matrix_compare_base.html', cat_names = cat_names,matrices = matrices, all_cats=all_cats)


# this is not used rn
@app.route("/tweet_annotation", methods=['GET', 'POST']) #here you need the analysis in request args!
@login_required
def tweet_annotation():
    analysis_id = request.args.get('analysis', 0, type=int)
    analysis = BayesianAnalysis.query.get(analysis_id)
    project = Project.query.get(analysis.project)
    tweets = project.tweets #Tweet.query.all()
    a_tweet = sample(tweets,1)[0]
    categories = project.categories#TweetTagCategory.query.all()
    tweet_table = {}
    # if category in request_dict_keys
    
    if "cat" in request.args.to_dict().keys():
        cat_id = request.args.get('cat', type=int)
        tweets = Tweet.query.filter(Tweet.category==cat_id)
        tweet_table = { t.id : {'tweet': t.full_text, 'category': t.handle, "id": t.id} for t in tweets}
        tweet_table = sorted([t for t in tweet_table.items()], key=lambda x:x[1]["id"], reverse=True)
        tweet_table = [t[1] for t in tweet_table]
        
    if request.method == "POST" and 'select-category' in request.form.to_dict():
        myargs = request.form.to_dict()
        cat_id = myargs['select-category']
        tweets = Tweet.query.filter(Tweet.category==cat_id)
        tweet_table = { t.id : {'tweet': t.full_text, 'category': t.handle, "id": t.id} for t in tweets}
        tweet_table = sorted([t for t in tweet_table.items()], key=lambda x:x[1]["id"], reverse=True)
        tweet_table = [t[1] for t in tweet_table]
        return redirect(url_for('tweet_annotation', analysis=analysis.id,cat = cat_id))
        
    return render_template('tweet_annotate.html', tweet_table= tweet_table, categories=categories, analysis=analysis)

@app.route("/annotation_summary/<analysis_id>", methods=['GET', 'POST'])
@login_required
def annotation_summary(analysis_id):
    
    analysis = BayesianAnalysis.query.get(analysis_id)
    all_tags = list(analysis.annotation_tags.keys())

    if request.method == "POST" and 'select-tag' in request.form.to_dict():
        myargs = request.form.to_dict()
        new_tag = myargs['select-tag']
        return redirect(url_for('annotation_summary',analysis_id=analysis.id ,tag = new_tag))

    # get annotations by selected tag
    if "tag" in request.args.to_dict():
        a_tag = request.args.get('tag', type=str)
        
    else:
        a_tag = all_tags[0]
    
    # relevant annotations for a_tag
    tag_anns = TweetAnnotation.query.filter(TweetAnnotation.annotation_tag==a_tag).all()
    tagged_tweets = list(set([t.tweet for t in tag_anns]))
    
    #tweet_anns = TweetAnnotation.query.filter(TweetAnnotation.annotation_tag==a_tag).filter(TweetAnnotation.tweet.in_(tagged_tweets)).all()
    tag_table = {t: {'tweet':t} for t in tagged_tweets}
    for t in tagged_tweets:
        t_anns = TweetAnnotation.query.filter(TweetAnnotation.annotation_tag==a_tag).filter(TweetAnnotation.tweet==t).all()
        users = len(set([i.user for i in t_anns ]))
        tag_table[t]['tag_count'] = len(t_anns)
        tag_table[t]['users'] = users
    tag_table = sorted([t for t in tag_table.items()], key=lambda x:x[1]["tweet"], reverse=True)
    tag_table = [t[1] for t in tag_table]
    
    # do the same for all tags:
    tagdict = {t:{'tag':t} for t in all_tags}
    
    for tag in all_tags:
        tag_anns = TweetAnnotation.query.filter(TweetAnnotation.annotation_tag==tag).all()
        tagdict[tag]['tag_count'] = len(tag_anns)
        tagdict[tag]['users'] = len(set([an.user for an in tag_anns]))
        tagged_tweets = list(set([t.tweet for t in tag_anns]))
        tagdict[tag]['nr_tweets'] = len(tagged_tweets)
    alltag_table = sorted([t for t in tagdict.items()], key=lambda x:x[1]["nr_tweets"], reverse=True)
    alltag_table = [t[1] for t in alltag_table]

    chart_data = nlp4all.utils.create_bar_chart_data({tag:tagdict[tag]['users'] for tag in all_tags}, title="Annotation tags")

    return render_template('annotation_summary.html', ann_table=tag_table, analysis=analysis, tag=a_tag, all_tags=all_tags, allann_table=chart_data)

@app.route("/annotations", methods=['GET', 'POST'])
@login_required
def annotations():
    page = request.args.get('page', 1, type=int)
    analysis_id = request.args.to_dict()['analysis_id']#, 0, type=int)
    analysis = BayesianAnalysis.query.get(analysis_id)
    project = Project.query.get(analysis.project)
    anns = TweetAnnotation.query.filter(TweetAnnotation.analysis==analysis_id).all()
    a_list = set([a.tweet for a in anns])
    tweets = Tweet.query.filter(Tweet.id.in_(a_list)).all()
    
    ann_info ={a.id :{'annotation': a.text, 'tag': a.annotation_tag} for a in anns}
    #ann_table =  {t.id : {'annotation': t.text,'tag':t.annotation_tag , "tweet_id": t.tweet, 'tag_counts':1}for t in anns}
    
    ann_dict = analysis.annotation_counts(tweets)

    word_tuples=[]
    ann_tags = list(analysis.annotation_tags.keys())
    for a_tweet in tweets:
        mytagcounts = nlp4all.utils.get_tags(analysis,set(a_tweet.words), a_tweet)
        myanns = TweetAnnotation.query.filter(TweetAnnotation.tweet==a_tweet.id).all()
        my_tuples = nlp4all.utils.ann_create_css_info(mytagcounts, a_tweet.full_text,ann_tags, myanns)
        word_tuples.append(my_tuples)
    
    ann_list = Tweet.query.join(TweetAnnotation , (TweetAnnotation.tweet == Tweet.id)).filter_by(analysis=analysis_id).order_by(Tweet.id).distinct().paginate(page, per_page=1)
    
    next_url = url_for('annotations', analysis_id=analysis_id, page=ann_list.next_num) \
        if ann_list.has_next else None
    prev_url = url_for('annotations', analysis_id=analysis_id,page=ann_list.prev_num) \
        if ann_list.has_prev else None
    
    return render_template('annotations.html',  anns=ann_list.items, next_url=next_url, prev_url=prev_url, word_tuples=word_tuples, page=page, analysis=analysis, ann_dict=ann_dict)


@app.route('/save_annotation', methods=['GET', 'POST'])
def save_annotation():
    args = request.args.to_dict()
    t_id = int(args['tweet_id'])
    tweet = Tweet.query.get(t_id)
    text = str(args['text'])
    atag = str(args['atag'])
    pos = int(args['pos'])
    analysis_id = int(args['analysis'])
    analysis = BayesianAnalysis.query.get(analysis_id)
    analysis.updated_a_tags(atag, tweet)
    flag_modified(analysis, "annotation_tags") 
    db.session.add(analysis)
    db.session.merge(analysis)
    db.session.flush()
    db.session.commit()
    
    if 'start' in args:
        txtstart = min(int(args['start']), int(args['end']))
        txtend = max(int(args['start']), int(args['end']))
    else:
        txtstart= tweet.full_text.find(text) # make sure this is the full text in the final version!
        if txtstart < 0:
            txtstart=0
        txtend = txtstart + len(text)
    
    coordinates = {}
    
    tag_count =0 
    tag_count_list =[]
    coords = {}
    for t in text.split():
        tag_count_list.append((tag_count+pos, t))
        txt = re.sub(r'[^\w\s]','',t.lower())
        tag_pos = tag_count+pos
        coords[tag_pos] = (txt, t)
        tag_count +=1
    coordinates['txt_coords'] = coords

    words = tweet.full_text.split()
    length = list(range(len(words)))
    word_locs = {}
    left_text = tweet.full_text
    s= 0
    word_count = 0
    for w in range(len(words)):
        word_locs[word_count] = words[w] 
        word_count += 1
    words = [re.sub(r'[^\w\s]','',w) for w in text.lower().split() if "#" not in w and "http" not in w and "@" not in w]#text.split() 
    coordinates['word_locs'] = word_locs
    annotation = TweetAnnotation(user = current_user.id, text=text, analysis=analysis_id, tweet=t_id, coordinates=coordinates, words=words,annotation_tag=atag.lower())
    db.session.add(annotation)
    db.session.commit()
   
    return jsonify(words,coordinates['txt_coords'])


@app.route('/plotly_scatter_plot', methods=['GET'])
def plotly_scatter_plot():
    model = D2VModel.query.filter_by(id=2).first().load()
    cats = [1,2,3]
    data_x = []
    data_y = []
    labels = [TweetTagCategory.query.filter_by(id=cats[i]).first().name for i in range(len(cats))]
    tsne = TSNE(n_components=2)
    tweets = Tweet.query.filter(Tweet.category.in_(cats)).all()
    dv = [model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]
    reduced_dv = tsne.fit_transform(dv)
    separated_docvecs = separate_by_cat(reduced_dv, cats)
    for cat_vecs in separated_docvecs:
        data_x.append(list(cat_vecs[:,0]))
        data_y.append(list(cat_vecs[:,1]))
    print(labels)
    return render_template('plotly_scatterplot.html', data_x=data_x, data_y=data_y, labels=json.dumps(labels))


@app.route('/3D_scatter_plot', methods=['GET'])
def scatter_plot_3D():
    model = D2VModel.query.filter_by(id=2).first().load()
    cats = [2,8,9,10]
    data_x = []
    data_y = []
    data_z = []
    labels = [TweetTagCategory.query.filter_by(id=cats[i]).first().name for i in range(len(cats))]
    docvecs = []
    pca = PCA(n_components=3)
    for cat_id in cats:
        tweets = Tweet.query.filter_by(category=cat_id).all()
        dv = [model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]  # doc vecs of 1 cat
        pca.fit(dv)
        docvecs.append(dv)
    for dv in docvecs:
        vecs = pca.transform(dv)
        data_x.append(list(vecs[:,0]))  # size = (n_cats, n_doc_per_cat)
        data_y.append(list(vecs[:,1]))
        data_z.append(list(vecs[:,2]))
    return render_template('3D_scatter_plot.html', data_x=data_x, data_y=data_y, data_z=data_z,
                           labels=json.dumps(labels))


@app.route("/word_embedding", methods=['GET', 'POST'])
@login_required
def word_embedding():
    # form to display what you want
    model_id = request.args.get('model', None, type=int)
    model = D2VModel.query.filter_by(id=model_id).first().load()
    display_form = DisplayWordEmbeddingForm()
    display_form.displayed_set.choices = [(str(cat.id), cat.name) for cat in TweetTagCategory.query.all()]  # should be changed to contain only the project's cats
    display_form.method.choices = [('1', 'PCA'), ('2', 't-SNE')]
    task_id = request.args.get('task', None, type=str)
    show_form = ShowForm()

    if display_form.validate_on_submit():
        displayed_cats = display_form.displayed_set.data
        n_components = 2
        labels = [TweetTagCategory.query.filter_by(id=cat).first().name for cat in displayed_cats]
        if display_form.dimension.data:  # if 3D
            n_components = 3
        # choosing the reduction method
        # the model should have a 'reduce_with' method
        if display_form.method.data == '1':
            reduction_function = 'reduce_with_PCA'
        elif display_form.method.data == '2':
            reduction_function = 'reduce_with_TSNE'
        tweets = Tweet.query.filter(Tweet.category.in_(displayed_cats)).all()
        dv = np.array([model.infer_vector(simple_preprocess(tweet.text)) for tweet in tweets]).tolist()
        reduced_dv = reduce_dimension.delay(dv, displayed_cats, reduction_function, n_components, labels)
        flash("The dimension reduction is processing")
        return redirect(url_for('word_embedding', model=model_id, task=reduced_dv.id))
        #return render_template("word_embedding.html", title='Display vectors', display_form=display_form,
        #                       show_form=show_form)

    if show_form.validate_on_submit():
        reduced_dv = AsyncResult(task_id, app=celery_app)
        if reduced_dv.state == 'PENDING':
            flash("The process is not over yet :(")
        elif reduced_dv.state == 'SUCCESS':
            data_x, data_y, data_z, labels = reduced_dv.get()
            if data_z != []:
                return render_template('3D_scatter_plot.html', data_x=data_x, data_y=data_y, data_z=data_z,
                               labels=json.dumps(labels))
            return render_template('plotly_scatterplot.html', data_x=data_x, data_y=data_y, labels=json.dumps(labels))
        else:
            flash('Something went wrong somewhere... :(')

    show_form_param = show_form if task_id else None
    return render_template("word_embedding.html", title='Display vectors', display_form=display_form,
                           show_form=show_form_param)


from nlp4all.utils import addition
@app.route('/html_de_ses_morts', methods=['GET'])
def test_html():
    a = 2
    b = 4
    add = addition.delay(a, b)
    form = ShowForm()
    return render_template('html_de_ses_morts.html', add=add, form=form)


from time import sleep
#background process happening without any refreshing
@app.route('/background_process_test')
def background_process_test():
    a = 2
    b = 4
    sleep(2)
    return str(a + b)
