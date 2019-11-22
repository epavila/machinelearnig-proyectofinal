# Importar Librerias
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from keras.layers.core import Dense
from keras.models import Sequential
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


# funcion para Leer CSV
def lecturaCSV(path):
    data = read_csv(path)
    return data


# Funcion para Evaluar modelos
def evaluar(modelo, datos_train, etiqueta_train, datos_test, etiqueta_test):
    y_pred_train = modelo.predict(datos_train)
    if len(y_pred_train.shape) > 1:
        prediction_train = modelo.predict(X_train)
        prediction_train = prediction_train[:, 0].round()
        prediction_test = modelo.predict(X_test)
        prediction_test = prediction_test[:, 0].round()
        error_train = mean_absolute_error(y_train[:, 0], prediction_train)
        error_test = mean_absolute_error(y_test[:, 0], prediction_test)
        f_score_train = f1_score(y_train[:, 0], prediction_train)
        f_score_test = f1_score(y_test[:, 0], prediction_test)
        roc_auc_train = roc_auc_score(y_train[:, 0], prediction_train)
        roc_auc_test = roc_auc_score(y_test[:, 0], prediction_test)
        fpr_train, tpr_train, thresholds_train = roc_curve(y_train[:, 0], prediction_train)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test[:, 0], prediction_test)
        rep_test = classification_report(y_test[:, 0], prediction_test)
        rep_train = classification_report(y_train[:, 0], prediction_train)
        del thresholds_train
        del thresholds_test
        print('Matriz de confución Entrenamiento \n')
        print(confusion_matrix(y_train[:, 0], prediction_train))
        print('Reporte de clasificacion Entrenamiento \n')
        print(rep_train)
        print('Matriz de confución Prueba \n')
        print(confusion_matrix(y_test[:, 0], prediction_test))
        print('Reporte de clasificacion Prueba \n')
        print(rep_test)
    else:
        y_pred_train = modelo.predict(datos_train)
        y_pred_test = modelo.predict(datos_test)
        error_train = mean_absolute_error(etiqueta_train[:, 0], y_pred_train)
        error_test = mean_absolute_error(etiqueta_test[:, 0], y_pred_test)
        f_score_train = f1_score(etiqueta_train[:, 0], y_pred_train)
        f_score_test = f1_score(etiqueta_test[:, 0], y_pred_test)
        roc_auc_train = roc_auc_score(etiqueta_train[:, 0], y_pred_train)
        roc_auc_test = roc_auc_score(etiqueta_test[:, 0], y_pred_test)
        fpr_train, tpr_train, thresholds_train = roc_curve(etiqueta_train[:, 0], y_pred_train)
        fpr_test, tpr_test, thresholds_test = roc_curve(etiqueta_test[:, 0], y_pred_test)
        rep_test = classification_report(etiqueta_test[:, 0], y_pred_test)
        rep_train = classification_report(etiqueta_train[:, 0], y_pred_train)
        print('Matriz de confución Entrenamiento \n')
        print(confusion_matrix(y_train[:, 0], y_pred_train))
        print('Reporte de clasificacion Entrenamiento \n')
        print(rep_train)
        print('Matriz de confución Prueba \n')
        print(confusion_matrix(y_test[:, 0], y_pred_test))
        print('Reporte de clasificacion Entrenamiento \n')
        print(rep_test)

    error = pd.DataFrame(columns=['Correcto', 'Equivocado', 'F-Score', 'ROC-AUC'], index=['Entrenamiento', 'Prueba'])
    error.loc['Entrenamiento', 'Correcto'] = (1 - error_train)
    error.loc['Entrenamiento', 'Equivocado'] = error_train
    error.loc['Entrenamiento', 'F-Score'] = f_score_train
    error.loc['Entrenamiento', 'ROC-AUC'] = roc_auc_train
    error.loc['Prueba', 'Correcto'] = (1 - error_test)
    error.loc['Prueba', 'Equivocado'] = error_test
    error.loc['Prueba', 'F-Score'] = f_score_test
    error.loc['Prueba', 'ROC-AUC'] = roc_auc_test
    display(error)
    plt.plot(fpr_train, tpr_train, color='orange', label='ROC_train')
    plt.plot(fpr_test, tpr_test, color='blue', label='ROC_test')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


# Obtener Datos
PATH = "muestra_equilibrada1.csv"
datos = lecturaCSV(PATH)

# transformar de SI o NO a 0 y 1
datos = pd.get_dummies(datos, columns=['Es_Bueno'])
buenos = datos.pop('Es_Bueno_SI')
datos['Es_Bueno_SI'] = buenos

# Plotear las variables numéricas para ver distribución de los datos

columnas = ['plazoOperacion', 'montoCuotaOperacion', 'ingresosCliente',
            'montoOperacion', 'gastosCliente', 'valorVehiculo',
            'numeroCuotasInicioOperacion', 'edadCliente']
columnas1 = ['plazoOperacion', 'montoCuotaOperacion', 'ingresosCliente',
             'montoOperacion', 'gastosCliente', 'valorVehiculo',
             'numeroCuotasInicioOperacion', 'edadCliente']
etiqueta = datos.columns[-1]
bueno = datos[etiqueta] == 1
malo = datos[etiqueta] == 0
for col in columnas:
    for col1 in columnas1:
        if col != col1:
            plt.xlabel(col, fontsize=15)
            plt.ylabel(col1, fontsize=15)
            plt.scatter(datos[col][bueno], datos[col1][bueno], color='b')
            plt.scatter(datos[col][malo], datos[col1][malo], color='r')
            plt.show()
    columnas1.remove(col)

# Separar en sets de entreamiento y prueba
todo = datos.values
X = todo[:, 0:21]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
xmin = scaler.data_min_
xmax = scaler.data_max_
x_min_max = pd.DataFrame(data=[xmin, xmax])
PATH1 = "x_min_max.csv"
x_min_max.to_csv(PATH1, index=False)
Y = todo[:, 21:23]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# crear red neuronal y entrenarla
nn = Sequential()
nn.add(Dense(12, activation='tanh', input_dim=21))
nn.add(Dense(12, activation='sigmoid', input_dim=12))
nn.add(Dense(2, activation='softmax'))
nn.compile(optimizer='rmsprop',
           loss='binary_crossentropy',
           metrics=['accuracy'])

nn.fit(X_train, y_train, epochs=100)

# Evaluar Red Neuronal
evaluar(nn, X_train, y_train, X_test, y_test)

# Crear y Evaluar modelo de regresion logística
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train[:, 0])
evaluar(lr, X_train, y_train, X_test, y_test)

# Crear y Evaluar SVC
svclassifier = SVC(kernel='rbf', gamma='scale')
svclassifier.fit(X_train, y_train[:, 0])
evaluar(svclassifier, X_train, y_train, X_test, y_test)

# KNN

n_k = 28
knn = KNeighborsClassifier(n_k)
knn.fit(X_train, y_train[:, 0])
evaluar(svclassifier, X_train, y_train, X_test, y_test)

# Guardar el mejor modelo en el disco
filename = "nn1.h5"
nn.save(filename)
