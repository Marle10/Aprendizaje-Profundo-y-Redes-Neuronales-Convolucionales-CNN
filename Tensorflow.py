# Importar la biblioteca TensorFlow y el módulo de datasets de Keras
import tensorflow as tf
from tensorflow.keras import datasets

# Cargar el conjunto de datos CIFAR-10
# Se separa en datos de entrenamiento (train_images, train_labels) y de prueba (test_images, test_labels)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar los valores de los píxeles de las imágenes al rango [0, 1]
# Esto mejora el rendimiento del modelo durante el entrenamiento
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definir el modelo secuencial con capas de convolución y de agrupamiento (pooling)
model = tf.keras.models.Sequential([
    # Primera capa convolucional: 32 filtros, cada uno de 3x3, con función de activación ReLU
    # La capa recibe imágenes de entrada con dimensiones 32x32 y 3 canales (RGB)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    # Capa de agrupamiento por máximos: reduce las dimensiones de la salida anterior a la mitad (16x16x32)
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Segunda capa convolucional: 64 filtros de 3x3 y función de activación ReLU
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Capa de agrupamiento por máximos: reduce nuevamente las dimensiones a la mitad (8x8x64)
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Tercera capa convolucional: 64 filtros de 3x3 y función de activación ReLU
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Capa de aplanado: convierte la salida en un vector unidimensional (de 1600 unidades)
    tf.keras.layers.Flatten(),
    
    # Capa densa completamente conectada con 64 neuronas y activación ReLU
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Capa de salida: 10 neuronas sin función de activación
    # Cada neurona representa una de las 10 clases en el conjunto de datos CIFAR-10
    tf.keras.layers.Dense(10)
])

# Configurar el modelo para el entrenamiento
# Optimizador: Adam
# Función de pérdida: Sparse Categorical Crossentropy con logits (sin softmax)
# Métrica de evaluación: precisión
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenar el modelo con los datos de entrenamiento y validar en el conjunto de prueba
# Epochs: 10, el modelo pasará 10 veces por el conjunto de datos completo
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
