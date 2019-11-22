# Importar Librerias
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from keras.models import load_model
from keras import backend as K


# import pickle


# Leer modelo guardado
def open_model(filename):
    model = load_model(filename)
    return model


# Calificar Datos Ingresados
def calificacion(modelo, datos):
    result = modelo.predict(datos)
    result_df = pd.DataFrame(columns=['SI', 'NO', 'Buen_Pagador', 'Certeza'])
    result_df['SI'] = result[:, 1]
    result_df['NO'] = result[:, 0]
    det = result_df['SI'] > result_df['NO']
    result_df['Buen_Pagador'][det] = 'SI'
    result_df['Buen_Pagador'][~det] = 'NO'
    result_df['Certeza'][det] = result_df['SI']
    result_df['Certeza'][~det] = result_df['NO']
    result_df = result_df[['Buen_Pagador', 'Certeza']]
    result_df = result_df.drop(len(result_df) - 1)
    result_df = result_df.drop(len(result_df) - 1)
    return result_df


def preparacion(data):
    info = pd.DataFrame(columns=['plazoOperacion', 'montoCuotaOperacion', 'ingresosCliente',
                                 'montoOperacion', 'gastosCliente', 'valorVehiculo',
                                 'numeroCuotasInicioOperacion', 'edadCliente',
                                 'codigoProvinciaCliente_10', 'codigoProvinciaCliente_9',
                                 'codigoProvinciaCliente_otro', 'marcaVehiculo_Chery',
                                 'marcaVehiculo_Chevrolet', 'marcaVehiculo_DFSK',
                                 'marcaVehiculo_Hyundai', 'marcaVehiculo_otro',
                                 'tipoResidenciaCliente_A', 'tipoResidenciaCliente_F',
                                 'tipoResidenciaCliente_otro', 'claseVehiculo_Camion',
                                 'codigoNivelEstudiosCliente_P'])

    info['plazoOperacion'] = data['plazo_prestamo']
    info['montoCuotaOperacion'] = data['monto_cuota']
    info['ingresosCliente'] = data['ingresos']
    info['montoOperacion'] = data['monto_prestamo']
    info['gastosCliente'] = data['gastos']
    info['valorVehiculo'] = data['valor_vehiculo']
    info['numeroCuotasInicioOperacion'] = data['numero_cuotas']
    info['edadCliente'] = data['edad']

    info[['codigoProvinciaCliente_10', 'codigoProvinciaCliente_9',
          'codigoProvinciaCliente_otro', 'marcaVehiculo_Chery',
          'marcaVehiculo_Chevrolet', 'marcaVehiculo_DFSK',
          'marcaVehiculo_Hyundai', 'marcaVehiculo_otro',
          'tipoResidenciaCliente_A', 'tipoResidenciaCliente_F',
          'tipoResidenciaCliente_otro', 'claseVehiculo_Camion',
          'codigoNivelEstudiosCliente_P']] = 0

    data['provincia'] = data['provincia'].astype(str)
    data['provincia'] = data['provincia'].str.strip()
    info['codigoProvinciaCliente_10'][data['provincia'] == '10'] = 1
    info['codigoProvinciaCliente_9'][data['provincia'] == '9'] = 1
    provincia_otro = (data['provincia'] == '9') | (data['provincia'] == '10')
    info['codigoProvinciaCliente_otro'][~provincia_otro] = 1

    data['marca_vehiculo'] = data['marca_vehiculo'].astype(str)
    data['marca_vehiculo'] = data['marca_vehiculo'].str.strip()
    info['marcaVehiculo_Chery'][data['marca_vehiculo'] == '15'] = 1
    info['marcaVehiculo_Chevrolet'][data['marca_vehiculo'] == '5'] = 1
    info['marcaVehiculo_DFSK'][data['marca_vehiculo'] == '14'] = 1
    info['marcaVehiculo_Hyundai'][data['marca_vehiculo'] == '2'] = 1
    marca_otro = (data['marca_vehiculo'] == '15') | (data['marca_vehiculo'] == '5') | (
            data['marca_vehiculo'] == '14') | (data['marca_vehiculo'] == '2')
    info['marcaVehiculo_otro'][~marca_otro] = 1

    data['tipo_residencia'] = data['tipo_residencia'].astype(str)
    data['tipo_residencia'] = data['tipo_residencia'].str.strip()
    info['tipoResidenciaCliente_A'][data['tipo_residencia'] == 'A'] = 1
    info['tipoResidenciaCliente_F'][data['tipo_residencia'] == 'F'] = 1
    residencia_otro = (data['tipo_residencia'] == 'A') | (data['tipo_residencia'] == 'F')
    info['tipoResidenciaCliente_otro'][~residencia_otro] = 1

    data['clase_vehiculo'] = data['clase_vehiculo'].astype(str)
    data['clase_vehiculo'] = data['clase_vehiculo'].str.strip()
    info['claseVehiculo_Camion'][data['clase_vehiculo'] == '3'] = 1

    data['nivel_estudios'] = data['nivel_estudios'].astype(str)
    data['nivel_estudios'] = data['nivel_estudios'].str.strip()
    info['codigoNivelEstudiosCliente_P'][data['nivel_estudios'] == 'P'] = 1

    return info


def ejecucionModelo(data):
    valores = preparacion(data)
    X = valores.values
    PATH = "modelo_entrenado/x_min_max.csv"
    x_min_max = lecturaCSV(PATH)
    x_min_max = x_min_max.values
    X = numpy.concatenate((X, x_min_max), axis=0)
    print(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print(X)
    PATH1 = "modelo_entrenado/nn1.h5"
    model = open_model(PATH1)
    print(model.summary())
    cal = calificacion(model, X)
    K.clear_session()
    return cal


def lecturaCSV(path):
    data = read_csv(path)
    return data
