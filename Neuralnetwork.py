import tensorflow as tf
import numpy as np


class NeuralNetwork:

    def __init__(self) -> None:
        self.model = tf.keras.models.load_model('handwritting_model.model')

    def get_prediction(self, image: list) -> int:
        return np.argmax(self.model.predict([image]))

    def get_accuracy(self, image: list) -> int:
        return self.model.evaluate([image])
