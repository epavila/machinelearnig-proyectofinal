from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    username = StringField('Nombre usuario', validators=[DataRequired()])
    passwored = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Enviar')
