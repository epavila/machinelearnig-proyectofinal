from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired
from wtforms.widgets.html5 import NumberInput


class Credito(FlaskForm):
    lista_genero = [('', 'seleccionar genero'), ('LP', 'VEHICULOS LIVIANOS USO PARTICULAR'),
                    ('LT', 'VEHICULOS LIVIANOS USO DE TRABAJO'),
                    ('PT', 'VEHICULOS PESADO USO DE TRABAJO')]
    tipo_credito = SelectField('Tipo crédito', choices=lista_genero, validators=[DataRequired()])

    porcentaje_entrada = IntegerField('Entrada %', widget=NumberInput(min=0, max=100))
    monto_prestamo = IntegerField('Monto prestamo', widget=NumberInput(min=3748.4, max=39956.10))
    numero_cuotas = IntegerField('Numero cuotas', widget=NumberInput(min=1, max=120))
    monto_cuota = IntegerField('Monto cuotas', widget=NumberInput(min=15.715, max=4018.095))
    plazo_prestamo = IntegerField('Plazo prestamo', widget=NumberInput(min=1, max=60))
    tasa_prestamo = IntegerField('Tasa prestamo', widget=NumberInput(min=11.23, max=21.70))

    # submit = SubmitField('Procesar Modelo')
