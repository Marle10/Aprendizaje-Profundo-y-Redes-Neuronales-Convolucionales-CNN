# Explorando-el-Mundo-del-Aprendizaje-Profundo-y-Redes-Neuronales-Convolucionales-CNN-Objetivo-


Conv2D aplica filtros para extraer características de las imágenes, aumentando la capacidad de detección de patrones.

MaxPooling2D reduce las dimensiones espaciales, disminuyendo la cantidad de parámetros y mitigando el riesgo de sobreajuste.

Flatten convierte la salida de las capas convolucionales en un vector que puede ser usado en capas densas.

Dense aplica funciones de activación no lineales y conecta cada neurona a la salida, asignando probabilidades a las clases.

Este modelo es una red de tipo CNN (Convolutional Neural Network), ideal para clasificación de imágenes en el conjunto de datos CIFAR-10.

Resumen de las capas y parámetros del modelo:

Capa                 Salida	             Parámetros

Conv2D	           (32, 32, 32)	           896
MaxPooling2D	     (16, 16, 32)            	0
Conv2D	            (14, 14, 64)	        18,496
MaxPooling2D	      (7, 7, 64)	            0
Conv2D             	(5, 5, 64)	           36,928
Flatten	              (1600)	              0
Dense	                (64)	              102,464
Dense	                (10)	               650
