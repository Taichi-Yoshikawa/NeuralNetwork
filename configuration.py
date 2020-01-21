# << Intelligent systems >>
# REPORT#1 : 3-Layer Neural Network
# - Configuration


class Configuration():
    '''
        ## Configuration

        Parameters
        ----------

        Attributes
        ----------

    '''

    def __init__(self):
        # ----- Neural network components -----
        ## Neuron numbers in input-layer
        self.input_size         = 4
        ## Neuron numbers in hidden-layer
        self.hidden_size        = 5
        ## Neuron numbers in output-layer
        self.output_size        = 3

        # ----- Neural network options -----
        ## learning method
        ## 'SGD', 'Momentum', 'AdaGrad', 'Adam'
        self.learning_method    = 'SGD'
        self.iterations         = 10000
        self.batch_size         = 100
        self.learning_rate      = 0.1

        # ----- Dataset Configuration -----
        self.dataset_index = {
            'dec'   : [0,1,2,3],
            'obj'   : 4
        }
        self.dataset_one_hot_vector = {
            'Iris-setosa'       : [1,0,0],
            'Iris-versicolor'   : [0,1,0],
            'Iris-virginica'    : [0,0,1]
        }
