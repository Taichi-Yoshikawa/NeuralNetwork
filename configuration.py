# << Intelligent systems >>
# REPORT#1 : 3-Layer Neural Network
# - Configuration


class Configuration():
    '''
        Configuration
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
        ## 'SGD', 'Momentum', 'AdaGrad', 'Adam'
        self.learning_method    = 'Adam'
        self.loss_function      = 'cross-entropy-error'#'sum-squared-error'
        self.epoch              = 1000
        self.batch_size         = 10
        # SGD/Momentum/AdaGrad : 0.01  /  Adam : 0.001
        self.learning_rate      = 0.001
        self.weight_init        = 0.01

        # ----- Dataset Configuration -----
        self.dataset_url    = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        self.dataset_index  = {
            'dec'   : [0,1,2,3],
            'obj'   : 4
        }
        self.dataset_one_hot_vector = {
            'Iris-setosa'       : [1,0,0],
            'Iris-versicolor'   : [0,1,0],
            'Iris-virginica'    : [0,0,1]
        }
        self.dataset_ratio = {
            'train' : 100,
            'test'  : 50
        }

        # ----- I/O Configuration -----
        self.path_out = '.'
