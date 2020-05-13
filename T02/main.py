
import numpy as np
import png
import pickle
import itertools
import PIL
import os
from sklearn.neighbors import KNeighborsClassifier as KNN                       
from pybalu.performance_eval import performance
from pybalu.feature_extraction import lbp_features
from sfs import sfs

# +++++++++++++++ LEER IMPORTANTE ++++++++++++++++ #
# main es el metodo que se pide que se pueda llamar para iterar, en mi caso lo definí
# con el path de la imagen a mostrar y n siendo el numero de caracteristicas a usar.
# Para que esto funcione primero se deben de haber corrido 2 funciones, create_info() y
# train(n) con el n con que se va a correr main.
# A su vez para poder correr train(n) se debe de haber corrido create_info() el cual basicamente
# obtiene los lbps de cada imagen y los guarda en un diccionario, eso acelera considerablemente todo
# el proceso de training y testing, notar que train() no reentrena si ya fue ejecutado para cierto n, 
# de igual manera create_info() solo crea la data una vez, si ya existe retorna sin hacer nada.

# El unico detalle a tener en consideracion es que se debe de tener todos estos modulos dentro de
# una carpeta "T02" para que los path funcionen y tambien las carpetas "Testing_0", "Testing_1",
# "Training_0" y "Training_1" con las fotos respectivas dentro de ellas, todo lo demas viene en el zip (incluidas las carpetas
# con las imagenes de train y test). 
# Tambien tener en cuenta que SIEMPRE que se quiera ejecutar test(n) se DEBE DE HABER EJECUTADO 
# PREVIAMENTE train(n)

# Si se quiere ver como corre todo el programa recomiendo:
# - Borrar todo dentro de "trained classifier" pero NO BORRAR LA CARPETA
# - Borrar "results.p"
# - Borrar "test_lbps.p"
# - Borrar "train_lbps.p"

# Y correr el siguiente script:
# create_info()
# for i in range(1,51):
#     train(i)
# results()

# WARNING: Se demora harto

# Si solo se quiere ver como corre el mejor basta con:
# create_info()
# train()
# test()



# Dado el path a una imagen retornamos su clasificacion (0 o 1)
def main(path, n= 50):
    #Abrimos el clasificador que entrenamos previamente
    with open('T02/trained_classifiers/{}.p'.format(n), 'rb') as fp:
        classifier, selected_chars = pickle.load(fp)
    _width, _height, rows, _info = png.Reader(filename= path).read()
    img_lbp = [lbp_features(get_greyscale_matrix(rows), hdiv=1, vdiv=1)]
    Y_test = filter_matrix_by_columns(img_lbp, selected_chars)
    return classifier.predict(Y_test)


# Entrena el clasificador acorde a SFS(n) y KNN=3
# Tambien guarda este clasificador y las caracteristicas seleccionadas
# en trained_classifier.p
def train(n = 50):
    if os.path.exists('T02/trained_classifiers/{}.p'.format(n)):
        return
    with open('T02/train_lbps.p', 'rb') as fp:
        data = pickle.load(fp)
    X_train = data[0] + data[1]
    d_train = [0]*len(data[0]) + [1]*len(data[1])
    # Aplicar SFS sobre X_train
    selected_chars = sfs(X_train, d_train, n)
    X_train = filter_matrix_by_columns(X_train, selected_chars)
    knn = KNN(n_neighbors=3)
    knn.fit(X_train,d_train)
    with open('T02/trained_classifiers/{}.p'.format(n), 'wb') as fp:
        data = [knn, selected_chars]
        pickle.dump(data, fp)


# Dada una matriz y una lista de columnas retorna la matrix con solo aquellas columnas
def filter_matrix_by_columns(matrix, selected_columns):
    return [[row[column] for column in selected_columns] for row in matrix]


# Si antes se ejecutó "create_info"  y "train" testeará que tan 
# bueno es el clasificador
def test(n= 50):
    # Abrimos el clasificador que entrenamos previamente
    with open('T02/trained_classifiers/{}.p'.format(n), 'rb') as fp:
        classifier, selected_chars = pickle.load(fp)
    with open('T02/test_lbps.p', 'rb') as fp:
        data = pickle.load(fp)
    X_test = data[0] + data[1]
    X_test = filter_matrix_by_columns(X_test, selected_chars)
    d_test = [0]*len(data[0]) + [1]*len(data[1])
    Y_pred = classifier.predict(X_test)
    correct = 0
    total = 0
    for i in range(len(Y_pred)):
        total += 1
        if Y_pred[i] == d_test[i]:
            correct += 1
    accuracy = correct/total
    print("Accuracy (n={}): {}%".format(n, accuracy*100))
    return n, accuracy


# Dado los paths de cada training y test set obtiene los lbp de cada 
# imagen y los guarda respecticamente en train_lbps.p y test_lbps.p
def create_info(train_0_path= "T02/Training_0/",
  train_1_path= "T02/Training_1/",
  test_0_path= "T02/Testing_0/",
  test_1_path= "T02/Testing_1/"):
    train = {0: [], 1: []}
    test = {0: [], 1: []}

    # All features from train 
    if not os.path.exists('T02/train_lbps.p'):
        for img_name in os.listdir(train_0_path):
            if ".png" not in img_name:
                continue
            _width, _height, rows, _info = png.Reader(filename="{}{}".format(train_0_path, img_name)).read()
            train[0].append(lbp_features(get_greyscale_matrix(rows), vdiv=1, hdiv=1))
        for img_name in os.listdir(train_1_path):
            if ".png" not in img_name:
                continue
            _width, _height, rows, _info = png.Reader(filename="{}{}".format(train_1_path, img_name)).read()
            train[1].append(lbp_features(get_greyscale_matrix(rows), vdiv=1, hdiv=1))
        with open('T02/train_lbps.p', 'wb') as fp:
            pickle.dump(train, fp)

    # All features from test
    if not os.path.exists('T02/test_lbps.p'):
        for img_name in os.listdir(test_0_path):
            if ".png" not in img_name:
                continue
            _width, _height, rows, _info = png.Reader(filename="{}{}".format(test_0_path, img_name)).read()
            test[0].append(lbp_features(get_greyscale_matrix(rows), vdiv=1, hdiv=1))
        for img_name in os.listdir(test_1_path):
            if ".png" not in img_name:
                continue
            _width, _height, rows, _info = png.Reader(filename="{}{}".format(test_1_path, img_name)).read()
            test[1].append(lbp_features(get_greyscale_matrix(rows), vdiv=1, hdiv=1))
        with open('T02/test_lbps.p', 'wb') as fp:
            pickle.dump(test, fp)


# Dada una matriz 2D en formato RGB retorna su matriz en greyscale
def get_greyscale_matrix(image):
    grey_channel = []
    for row in image:
        pixels = grouper(3, row)
        grey_row = []
        for pix in pixels:
            if None in pix:
                print(pix)
            grey_row.append(sum(pix)/3)
        grey_channel.append(grey_row)
    return np.array(grey_channel)


# Retorna un iterable en grupos de n
def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


# Esta funcion corre todos los clasificadores y guarda
# sus resultados en results.p (de n=1 hasta n=50)
def results():
    results = {}
    for i in range(1,51):
        n, acc = test(i)
        results[n] = acc
    with open('T02/results.p', 'wb') as fp:
        pickle.dump(results, fp)