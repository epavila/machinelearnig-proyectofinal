from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField
from wtforms.validators import DataRequired
from wtforms.widgets.html5 import NumberInput


class Vehiculo(FlaskForm):
    valor_vehiculo = IntegerField('Valor vehículo', widget=NumberInput(min=6500, max=60000))

    lista_marca_vehiculo = [('', 'seleccionar vehículo'), (1, 'RENAULT'),
                            (2, 'HYUNDAI'), (3, 'NISSAN'), (4, 'KIA'), (5, 'CHEVROLET'), (6, 'MAZDA'),
                            (7, 'TOYOTA'), (8, 'HONDA'), (9, 'SUZUKI'), (10, 'WOLKSWAGEN'), (11, 'MITSUBISHI'),
                            (8, 'PEUGEOT'), (9, 'HINO'), (10, 'FORD'), (11, 'GREATWALL'), (12, 'BYD'),
                            (13, 'FIAT'), (14, 'DFSK'), (15, 'CHERY'), (16, 'CHANGHE'), (17, 'OTROS')
                            ]
    marca_vehiculo = SelectField('Marca Vehículo', choices=lista_marca_vehiculo, validators=[DataRequired()])

    lista_clase_vehiculo = [('', 'seleccionar clase vehículo'), (1, 'AUTOMÓVIL'),
                            (2, 'BUS'), (3, 'CAMIÓN'), (4, 'CAMIONETA'), (5, 'FURGONETA'), (6, 'JEEP'),
                            (7, 'TRACTOR'), (8, 'OTROS')
                            ]
    clase_vehiculo = SelectField('Clase Vehículo', choices=lista_clase_vehiculo, validators=[DataRequired()])

    lista_estado_vehiculo = [('', 'seleccionar clase vehículo'), (1, 'NUEVO'),
                             (2, 'USADO')
                             ]
    estado_vehiculo = SelectField('Estado Vehículo', choices=lista_estado_vehiculo, validators=[DataRequired()])

    # submit = SubmitField('Procesar Modelo')
