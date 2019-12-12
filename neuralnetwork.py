# << Intelligent systems >>
# REPORT#1 : 3-Layer Neural Network
# - Neural Network


import numpy            as np
import pandas           as pd
import configuration    as cf


class NeuralNetwork():
    '''
        ## 3-Layer Neural Network

        Parameters
        ----------

        `cnf`           : Configuration class

        Attributes
        ----------

        Using Method
        ------------

        1. `Initialize()`
        2. `update(gen)`

    '''

    def __init__(self, cnf):
        self.cnf    = cnf


    def predict(self):
        pass

    def loss(self):
        pass

    def accuracy(self):
        pass

    def gradient(self):
        pass


class LearningNeuralNetwork():
    def __init__(self, neuralnetwork):
        self.neuralnetwork  = neuralnetwork
        self.dataset_url    = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


    def main(self):
        '''
            main function
        '''
        self.readDataset()

    def readDataset(self):
        df = pd.read_csv(self.dataset_url, header=None)
        print(df.tail())

    def selectData(self):
        pass


if __name__ == "__main__":
    cnf = cf.Configuration()
    net = NeuralNetwork(cnf)
    lnn = LearningNeuralNetwork(net)
    lnn.main()