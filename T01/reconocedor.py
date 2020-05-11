from main import *

# Este modulo tiene al reconocedor, todas las demas funciones relevantes están en main.py
# Mi metodo implementa un diccionario para guardar los momentos de Hu de todas las imagenes
# del training, si aquel archivo no se encuentra creado como 'training_moments.p'
# entonces el programa lo crea en la primera iteracion (y esto demora aprox 200s).

# Para crear este archivo es necesario tener la siguiente estructura de carpetas:
# T01/training/[letra_mayusc]/[examples].png
# Es decir una carpeta de entrenamiento, dentro de ella carpetas de cada letra y dentro
# de estas todos los ejemplos en png de esta letra

def reconocedor(img_matrix):
    #Si exite 'training_moments' lo abrimos, en caso contrario lo creamos
    try:
        with open('T01/training_moments.p', 'rb') as fp:
            data = pickle.load(fp)
    except IOError:
        create_info()
        with open('T01/training_moments.p', 'rb') as fp:
            data = pickle.load(fp)
    #Obtenemos la region que corresponde a la letra
    region = get_region(img_matrix)
    #Calculamos los momentos de Hu de la region
    img_hus = hus(region)
    scores_by_letter = {}
    #Iteramos sobre cada letra
    for letter in letters:
        #Una lista que guardará primeramente los scores de todos los ejemplos
        scores_by_letter[letter] = [] 
        for temp_hus in data[letter]:
            #Calculamos el inverso de la diferencia absoluta
            #De esta manera mientras menor el score significa que fue mas distinto
            difs = [abs((temp_hus[i]-img_hus[i])/img_hus[i])**-1 for i in range(3)]
            #Elegimos el score mas bajo de entre los 3
            scores_by_letter[letter].append(min(difs))
        #Elegimos el mejor representante de cada letra
        scores_by_letter[letter] = max(scores_by_letter[letter])
    #Elegimos la letra que tuvo mejor score
    letter = max([t for t in scores_by_letter.items()],key=itemgetter(1))[0] 
    return letters_to_numbers[letter]

# path = 'T01/test/G/0.png'
# print(reconocedor(black_n_white(path)))

      