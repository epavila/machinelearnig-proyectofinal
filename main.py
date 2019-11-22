import pandas as pd
from flask import Flask, redirect, render_template, session, url_for, request, json
from flask import flash
from flask_bootstrap import Bootstrap

import modelo_entrenado.ModeloEnProduccion as me
from modelo.Cliente import Cliente
from modelo.Credito import Credito
from modelo.LoginForm import LoginForm
from modelo.Vehiculo import Vehiculo

# TEST modelo:
#set FLASK_APP=main.py
#set FLASK_DEBUG=1
#set FLASK_ENV=development
#flask run


app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['SECRET_KEY'] = 'SUPER SECRETO'


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html', error=error)


@app.route('/', methods=['GET', 'POST'])
def index():
    login_form = LoginForm()

    context = {
        'login_form': login_form
    }

    if login_form.validate_on_submit():
        username = login_form.username.data

        if username == 'Equipo':
            session['username'] = username

            return redirect(url_for('ingreso_datos'))
        else:
            session['username'] = 'none'
            flash('Usuario no encontrado')

            return render_template('index.html', **context)

    return render_template('index.html', **context)


@app.route('/ingreso_datos', methods=['GET', 'POST'])
def ingreso_datos():
    if 'username' not in session:
        return redirect(url_for('index'))

    username = session.get('username')
    modelo_cliente = Cliente()
    modelo_credito = Credito()
    modelo_vehiculo = Vehiculo()

    context = {
        'username': username,
        'modelo_cliente': modelo_cliente,
        'modelo_credito': modelo_credito,
        'modelo_vehiculo': modelo_vehiculo
    }

    return render_template('form_ingreso_datos.html', **context)


@app.route('/procesar_modelo', methods=['POST'])
def procesar_modelo():

    #print("1: ")
    data_request = json.dumps({
        'genero': request.form['genero'],
        'edad': request.form['edad'],
        'estado_civil': request.form['estado_civil'],
        'nivel_estudios': request.form['nivel_estudios'],
        'ingresos': request.form['ingresos'],
        'gastos': request.form['gastos'],
        #'provincia': request.form['provincia'],
        'provincia': 1,
        'ciudad': request.form['ciudad'],
        'tipo_residencia': request.form['tipo_residencia'],

        'tipo_credito': request.form['tipo_credito'],
        'porcentaje_entrada': request.form['porcentaje_entrada'],
        'monto_prestamo': request.form['monto_prestamo'],
        'numero_cuotas': request.form['numero_cuotas'],
        'monto_cuota': request.form['monto_cuota'],
        'plazo_prestamo': request.form['plazo_prestamo'],
        'tasa_prestamo': request.form['tasa_prestamo'],

        'valor_vehiculo': request.form['valor_vehiculo'],
        'marca_vehiculo': request.form['marca_vehiculo'],
        'clase_vehiculo': request.form['clase_vehiculo'],
        'estado_vehiculo': request.form['estado_vehiculo']
    })

    #print("2: ")
    diccionario = json.loads(data_request)

    print('resultado DataFrame: ')
    data_frame = pd.DataFrame([diccionario])
    print(data_frame[['genero', 'edad', 'estado_civil', 'nivel_estudios', 'ingresos']])
    print(data_frame.columns)
    print(data_frame)

    modelo_produccion = me.ejecucionModelo(data_frame)

    print("Modelo Produccion :)")
    print(modelo_produccion)


    return json.dumps({
        'buen_pagador': modelo_produccion.loc[0,'Buen_Pagador'],
        'certeza': modelo_produccion.loc[0,'Certeza']
     })


