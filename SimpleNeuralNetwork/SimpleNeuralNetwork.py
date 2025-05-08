
"""
    Clase que implementa un ejemplo de Red Neural Simple .
    
    El código se basa en el tutorial :
        https://medium.com/@prxdyu/simple-neural-network-in-python-from-scratch-2814443a3050
"""
import numpy as np

class SimpleNeuralNetwork:
    """
        Esta vez, vamos a crear una red neural que tiene 1 capas ocultas por eso tendremos W1 que es el peso entre el input y la capa 
            y el peso 2 que es el peso entre la capa oculta y el output.
        Mas adelante, se puede plantear hacerlo parametrizable y experimentar con parámetros.

        Parámetros del constructor.
        n_features: numero de caracteristicas.
        n_classes: número de clases, es el número de neuronas en la capa de salida.
        n_hidden: numero de neuronas en las capas ocultas.
    """
    # Constructor, aqui Inicializamos la red neuronal.
    def __init__(self, n_features, n_classes, n_hidden):
        # Inicializamos los parámetros de la red.
        self._features = n_features
        self._classes = n_classes
        self._hidden = n_hidden
        
        # Config desde el input a la capa oculta 1.
        # Inicializamos la matriz de pesos de las capa de entrada y la oculta. Al inicio será con pesos aleatorios.
        self._W1  = 0.01 * np.random.randn(self._hidden * self._features)
        
        # Inicializamos el bias TODO: Documentar.
        self._b1 = np.zeros(self._hidden * self._features)

        # Config de la capa oculta 1 a la capa de salida.
        # Inicializamos la matriz de pesos de la red neural. 
        self._W2 = 0.01 *np.random.randn(self._hidden * self._classes)

        # Inicializamos la matriz de bias TODO: explicacion.
        self._b2 = np.zeros(self._hidden * self._classes)

    # Función de propagación hacia adelante.
    def frwd_prop(self, x):
        # De la entrada a la capa oculta...
        # Recibimos la entrada (vector ) y la multiplicamos por los pesos y le sumamos el bias.
        z1 = np.dot(x, self._W1) + self._b1
        # Aplicamos la función de activacnión. Esta función es responsable de dar soporte a valores no lineales  (pdte revisar documentacion)
        A1 = np.maximum(0, z1)

        # De la capa oculta a la capa de salida...
        # Recibimos la entrada y le aplicamos los pesos.
        z2 = np.dot(A1, self._W2) + self._b2
        # Aplicamos el softmax (TODO: Buscar documentacion)
        A2 = np.exp(z2)
        A2 = A2 /np.sum(A2, axis = 1, keepdims= True)
        return A1, A2

    # Funcion de propagacion hcia atras ( para actualizar los pesos.)
    def bckwd_prop(self, x, y, A1, A2):
        # Obtenemos el numero de muestras
        num_samples = y.shape[0]

        #Obtenemos las derivadas de los wrt CE  de perdida 


