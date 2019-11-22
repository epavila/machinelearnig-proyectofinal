from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField
from wtforms.validators import DataRequired
from wtforms.widgets.html5 import NumberInput


class Cliente(FlaskForm):
    lista_genero = [('', 'seleccionar genero'), ('M', 'MASCULINO'), ('F', 'FEMENINO')]
    genero = SelectField('Genero', choices=lista_genero, validators=[DataRequired()])

    edad = IntegerField('Edad', widget=NumberInput(min=23, max=93))

    lista_estado_civil = [('', 'seleccionar estado civil'), ('C', 'CASADO'), ('D', 'DIVORCIADO'), ('S', 'SOLTERO'),
                          ('U', 'UNION LIBRE'), ('V', 'VIUDO')]
    estado_civil = SelectField('Estado civil', choices=lista_estado_civil, validators=[DataRequired()])

    lista_nivel_estudios = [('', 'seleccionar nivel'), ('G', 'POSTGRADO'), ('N', 'SIN ESTUDIOS'), ('P', 'PRIMARIA'),
                            ('S', 'SECUNDARIA'), ('U', 'UNIVERSITARIA')]
    nivel_estudios = SelectField('Nivel estudios', choices=lista_nivel_estudios, validators=[DataRequired()])

    ingresos = IntegerField('Ingresos', widget=NumberInput(min=0, max=40000))
    gastos = IntegerField('Gastos', widget=NumberInput(min=0, max=7410.45))

    provincia = StringField('Provincia', validators=[DataRequired()])
    ciudad = StringField('Ciudad', validators=[DataRequired()])

    lista_tipo_residencia = [('', 'seleccionar nivel'), ('A', 'ARRENDADA'), ('F', 'VIVE CON FAMILIARES'),
                             ('N', 'PROPIA NO HIPOTECADA'),
                             ('P', 'PROPIA HIPOTECADA'), ('S', 'PRESTADA')]
    tipo_residencia = SelectField('Tipo residencia', choices=lista_tipo_residencia, validators=[DataRequired()])

    # submit = SubmitField('Procesar Modelo')
