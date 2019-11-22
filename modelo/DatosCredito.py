from flask_wtf import FlaskForm
from wtforms import SubmitField


class DatosCredito(FlaskForm):
    submit = SubmitField('Procesar Modelo')
