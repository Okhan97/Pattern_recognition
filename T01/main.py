import png
import numpy
import PIL
from PIL import Image
import itertools
from moments import hus
import os
from operator import itemgetter
import pickle

# Letras a usar y sus conversiones
letters = set('ASDFG')
letters_to_numbers = {
    'A': 1,
    'S': 2,
    'D': 3,
    'F': 4,
    'G': 5,
}

# Retorna un iterable en grupos de n
def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

# Recibe una matriz de 1s y 0s (blanco-negro) y retorna las coordenadas de la region blanca.
def get_region(image):
    region = []
    # Añado [1:-1] porque sabemos que no van a 
    # haber pixeles blancos en el borde de la imagen
    for i in range(len(image))[1:-1]:
        row = image[i]
        for j in range(len(row))[1:-1]:
            if image[i][j] == 1:
                region.append((i,j))
    return numpy.array(region)


# Dado un path a una imagen retorna su matrix blanco-negro
def black_n_white(path):
    _width, _height, rows, info = png.Reader(filename=path).read()
    matrix = numpy.vstack(map(numpy.uint8, rows))
    new_matrix = []
    for row in matrix:
        #Agrupamos respecto a cuantas variables tenemos por pixel
        pixels = grouper(info['planes'], row)
        new_row = []
        for pix in pixels:
            #Cualquier pixel que no sea negro entero se cuenta como blanco
            #Solo vemos el primer parametro puesto que sabemos que las imagenes
            #están en escala de grises
            if pix[0] > 0:
                new_row.append(1)
            else:
                new_row.append(0)
        new_matrix.append(new_row)
    return new_matrix

# Crea el picke (.p) que guarda los datos de los momentos
# Para crear este archivo es necesario tener la siguiente estructura de carpetas:
# T01/training/[letra_mayusc]/[examples].png
# Probablemente tengas que crear una carpeta T01 y luego tirar todo dentro para que funcione :)
# (tambien puedes cambiar todos los paths que están escritos pero eso suena mas largo)
# Es decir una carpeta de entrenamiento, dentro de ella carpetas de cada letra y dentro
# de estas todos los ejemplos en png de esta letra
# Todo este proceso toma aprox 200s
def create_info():
    info = {}
    root_path = 'T01/training/'
    for letter in letters:
        info[letter] = []
        path = root_path + letter 
        files = os.listdir(path)
        files = filter(lambda name: '.png' in name, files)
        for file in files:
            this_file_path = '{}/{}'.format(path, file)
            matrix = black_n_white(this_file_path)
            region = get_region(matrix)
            temp_hus = hus(region)
            info[letter].append(temp_hus)
    with open('T01/training_moments.p', 'wb') as fp:
        pickle.dump(info, fp)

# ---------------------------------------

def main(path):
    with open('T01/training_moments.p', 'rb') as fp:
        data = pickle.load(fp)
    img_matrix = black_n_white(path)
    region = get_region(img_matrix)
    img_hus = hus(region)
    scores_by_letter = {}
    for letter in letters:
        scores_by_letter[letter] = []
        for temp_hus in data[letter]:
            difs = [abs((temp_hus[i]-img_hus[i])/img_hus[i])**-1 for i in range(3)]
            scores_by_letter[letter].append(min(difs))
        scores_by_letter[letter] = max(scores_by_letter[letter])
    letter = max([t for t in scores_by_letter.items()],key=itemgetter(1))[0] 
    return letter


# ====================================
# Acá dejo comentado el script con el que corrí mis tests

## Correr solo una vez:
# create_info()

# Corre todos los test y printea sus resultados por letra y en total
# Tambien printea cuales fueron las respuestas incorrectas
tests_per_letter = 36
total_correct = 0
for letter in letters:
    print('-----------------------------')
    print('LETRA: {}'.format(letter))
    answers = []
    for i in range(tests_per_letter):
        path = 'T01/test/{}/{}.png'.format(letter, i) 
        answers.append(main(path))
    correct = 0
    for ans in answers:
        if ans == letter:
            correct += 1
        else:
            print("Creí que era una: {}".format(ans))
    total_correct += correct
    print("Porcentaje de acierto: {}%".format(correct/len(answers)*100))
    print("Aciertos: {}/{}".format(correct, len(answers)))
print('===========================')
print("Porcentaje de acierto: {}%".format(total_correct/(tests_per_letter*5)*100))
print("Aciertos: {}/{}".format(total_correct, tests_per_letter*5))