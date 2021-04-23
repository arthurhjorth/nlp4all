from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField, TextAreaField,\
    SelectMultipleField, SelectField, RadioField, FormField, FloatField, HiddenField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from nlp4all.models import User, Project 
from wtforms.fields.core import Label


class RegistrationForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    organizations = SelectField('Select your organization', validators=[DataRequired()])
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')

class TaggingForm(FlaskForm):
    # choices = RadioField()
    choices = SelectField("Hvem har skrevet dette tweet?")
    submit = SubmitField("Submit")

class TagButton(FlaskForm):
    submit = SubmitField("This doesn't matter apparently")
    
    def set_name(self, new_name):
        self.submit.label = Label("", new_name)

class AddTweetCategoryForm(FlaskForm):
    twitter_handle = StringField('Twitter handle', validators=[DataRequired()])
    submit = SubmitField('Create')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class AddProjectForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    description = StringField('Description', validators=[DataRequired()])
    categories = SelectMultipleField('Categories for your students to work with', validators=[DataRequired()])
    organization = SelectField('Which student group should participate?', validators=[DataRequired()])
    submit = SubmitField("Create Project")

    def validate_title(self, title):
        title = Project.query.filter_by(name=title.data).first()
        if title:
            raise ValidationError('That project name is taken. Please choose a different one.')


class AddBayesianAnalysisForm(FlaskForm):
    name = StringField('Title of Analysis', validators=[DataRequired()])
    shared = BooleanField('Shared project?')
    annotate = BooleanField('Annotate')
    number = IntegerField("How many of each category?")
    submit_bay = SubmitField('Create Analysis')

class RunBayesianAnalysisRobot(FlaskForm):
    run_analysis = SubmitField('Kør analysen!')

class AddBayesianRobotForm(FlaskForm):
    name = StringField('Title of Robot', validators=[DataRequired()])
    submit = SubmitField('Create New Robot')

class CreateMatrixForm(FlaskForm):
    categories = SelectMultipleField('Categories to compare', validators=[DataRequired()])
    ratio = IntegerField('Training tweets proportion (%)', validators=[DataRequired()])
    submit = SubmitField('Create Matrix')

class ThresholdForm(FlaskForm):
    shuffle = BooleanField('Shuffle tweets')
    threshold =  FloatField("Set a threshold between 0 and 1")
    ratio = IntegerField('Change training tweets proportion (%)')
    submit = SubmitField('Update')

class AnalysisForm(FlaskForm):
    robort_form = FormField(AddBayesianRobotForm)
    add_category_form = FormField(AddTweetCategoryForm)

class AddBayesianRobotFeatureForm(FlaskForm):
    feature = StringField('Skriv et søgeterm som du vil have at din robot skal lede efter. Du kan bruge wildcard (*) til at søge flere ord. F.eks. *bank vil give alle ord som slutter med bank. bank* vil give alle ord som starter med bank, og *bank* giver alle ord som indeholder bank.')
    reasoning = StringField("Forklar her hvorfor du tror at lige netop dette søgeterm vil være godt til at skelne mellem de to partier. Er det et emne som et af partierne taler mere om end det andet? Er det et ord som kun dét parti bruger? Skriv et 1-3 sætninger om hvorfor I vælger det.")
    submit = SubmitField("Tilføj søgeord")

class BayesianRobotForms(FlaskForm):
    add_feature_form = FormField(AddBayesianRobotFeatureForm)
    run_analysis_form = FormField(RunBayesianAnalysisRobot)

class UpdateAccountForm(FlaskForm):
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    picture = FileField('Update Profile Picture', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('Update')

    def validate_username(self, username):
        if username.data != current_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('That email is taken. Please choose a different one.')


class PostForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    content = TextAreaField('Content', validators=[DataRequired()])
    submit = SubmitField('Post')

class AddOrgForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    submit = SubmitField('Add Organization')


class RequestResetForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account with that email. You must register first.')


class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

class AnnotationForm(FlaskForm):
    text = TextAreaField('selectedtext', validators=[DataRequired()])
    start = IntegerField('start')
    end = IntegerField('end')
    hidden = HiddenField()
    submit = SubmitField('save text')


class SelectCategoryForm(FlaskForm):
    category = IntegerField('cat_id')
    submit = SubmitField('select category')


class AddWordEmbeddingForm(FlaskForm):
    name = StringField('Title of Analysis', validators=[DataRequired()])
    d2v = BooleanField('Doc2Vec model')  # create a Doc2Vec or a Word2Vec
    vector_size = IntegerField('Embedded vector size')
    epochs = IntegerField('Epochs')
    training_set = SelectMultipleField("Data to train on", validators=[DataRequired()])
    submit_emb = SubmitField('Create W2V Analysis')


class DisplayWordEmbeddingForm(FlaskForm):
    displayed_set = SelectMultipleField("Tweets to display", validators=[DataRequired()])
    method = SelectField("Choose the reduction method", validators=[DataRequired()])
    dimension = BooleanField("3D")
    submit_display = SubmitField("Display")
