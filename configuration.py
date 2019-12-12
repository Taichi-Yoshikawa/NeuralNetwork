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
        # ----- neural network components -----
        ## Neuron numbers in input-layer
        self.input_size         = 4
        ## Neuron numbers in hidden-layer
        self.hidden_size        = 5
        ## Neuron numbers in output-layer
        self.output_size        = 3

        # ----- neural network options -----
        ## learning method
        ## 'SGD', 'Momentum', 'AdaGrad', 'Adam'
        self.learning_method    = 'SGD'

