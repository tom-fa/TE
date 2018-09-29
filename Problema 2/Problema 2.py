# -*- coding: utf-8 -*-
# Modulo con funciones a utilizar (debe ser importado, no copiar y pegar las funciones!)

# -*- coding: utf-8 -*-
#Antenas de Celular
#¿Dónde posicionarlas?
#Número de Grupo
#Nombres
#Problema: Encontrar las mejores ubicaciones para instalar N antenas
#según distintos casos para diferentes escenarios.

#Detalles: PDF del problema 2 en Moodle.

# Modulo con funciones a utilizar (debe ser importado, no copiar y pegar las funciones!)
from utilities import getSignal,PSO,plotAntennas,plotPopulation,getPopulation
import numpy as np
#Funciones para obtener datos
#Debe conocer la estructura de los archivos para entender lo que debe retornar.


def leerAntenas(nombre_archivo):
    archivo = open(nombre_archivo)
    antenas = []
    for linea in archivo:
        x,y,p = map(float,linea.strip().split(','))
        antenas.append((x,y,p))
    return antenas
print leerAntenas('//home//anastasiia.fedorova//Escritorio//Problema 2//antenas1.txt')

#ver si el usos de x,y,p = archivo.readline() es mejor?
#parece que no

def leerPoblacion(nombre_archivo):
    archivo = open(nombre_archivo)
    poblacion = {}
    dic = ''
    for linea in archivo:
        l = linea.strip().split(',')
        if l[0].isalpha(): #no es un numero
            #poblacion[l[0]] = {}
            #poblacion[l[0]]['pos'] = l[1],l[2]
            #poblacion[l[0]]['poblacion'] = []
            #poblacion[l[0]]['radio'] = l[3],l[4]
            poblacion.setdefault(l[0],{})
            poblacion[l[0]].setdefault('pos',(float(l[1]),float(l[2])))
            poblacion[l[0]].setdefault('poblacion',[])
            poblacion[l[0]].setdefault('radio',(float(l[3]),float(l[4])))
            dic = poblacion[l[0]]['poblacion']
        # guarda nombre_poblacion: blah
        # el metodo setdefault trabaja unas 0.030s
        else:
            dic.append(tuple(map(float,l)))
    return poblacion
print leerPoblacion('//home//anastasiia.fedorova//Escritorio//Problema 2//poblacion1.txt')

#Caso 1: Señales
#Solo tienen la información de las antenas existentes.


def buscarPorSenal(N, antenas):
    # N: numero de antenas a buscar
    # antenas: informacion de las antenas ya existentes , lista de tuplas.
    nuevas_antenas = list(antenas)
    aux = list(antenas)
    for x in range(0,N):
        a = PSO(getSignal(nuevas_antenas),N,2,3.05,2.05,0.7,\
        [[10,0],[0,10]],30,'min')
        nueva_array = np.append(a,5) #aca aregar funcion que promedia las potencias en vez de 5 
        nuevas_antenas.append(nueva_array)
    return nuevas_antenas


#mejor 3.05,2.05,0.7
#Escenario 1

# Quizas sea buena idea ver la senal
plotAntennas(leerAntenas('//home//anastasiia.fedorova//Escritorio//Problema 2//antenas1.txt'))
# Busquemos nuevas antenas

# Ver la nueva senal?
plotAntennas(buscarPorSenal(20,leerAntenas('//home//anastasiia.fedorova//Escritorio//Problema 2//antenas1.txt')))


#Escenario 2

#plotAntennas(leerAntenas('antenas2.txt'))
# Busquemos nuevas antenas

# Ver la nueva senal?
#plotAntennas(buscarPorSenal(15,leerAntenas('antenas2.txt')))


'''
Caso 2: Población
Tienen la información tanto de las antenas ya existentes como de la ubicación de la población.
def buscarPorPoblacion(N, poblacion, antenas):
    # N: numero de antenas a buscar
    # poblacion: informacion de la poblacion, diccionario.
    # antenas: informacion de las antenas, lista de tuplas.
    nuevas_antenas = []
    #maximizar poblacion, minimizar antenas? dos ciclos?
    # Quizas se necesite optimizar algo...
    return nuevas_antenas
Escenario 1
# Quizas sea buena idea ver la poblacion
plotPopulation(buscarPorPoblacion(N, poblacion, antenas))
# Busquemos nuevas antenas
# Ver la nueva configuracion
plotPopulation(buscarPorPoblacion(N, poblacion, antenas))
#Escenario 2
# Lo mismo de arriba?
#Graficar población y señal
def graficarPoblacionAntenas(poblacion, antenas):
    # poblacion: informacion de la poblacion
    # antenas: informacion de las antenas
    # Crear un grafico bonito
    # No retorno nada ;)
# Probar el nuevo grafico
#Conclusiones y comentarios
'''

