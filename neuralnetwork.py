# << Intelligent systems >>
# REPORT#1 : 3-Layer Neural Network
# - Neural Network

import os
import time                         as tm
import numpy                        as np
import pandas                       as pd
from matplotlib  import pyplot      as plt
from collections import OrderedDict as od
import configuration                as cf


# ------------
#  Functions
# ------------
def identity_function(x):
    return x

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # prevent from overflow
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # correct index
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)


# ----------------
#  Neural Network
# ----------------
class NeuralNetwork:
    '''
        3-Layer Neural Network
    '''
    def __init__(self, weight_init_std = 0.01):
        # import configuration
        self.cnf = cf.Configuration()
        self.time = tm.strftime('%Y-%m-%d_%H-%M-%S')
        self.cnf_name = '_'.join([self.cnf.learning_method, self.cnf.loss_function, 'lr='+str(self.cnf.learning_rate)])
        # initialize parameters
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(self.cnf.input_size, self.cnf.hidden_size)
        self.params['b1'] = np.zeros(self.cnf.hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(self.cnf.hidden_size, self.cnf.output_size)
        self.params['b2'] = np.zeros(self.cnf.output_size)
        # Generate Layers
        self.layers = od()
        self.layers['Affine1']  = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']    = Relu()
        self.layers['Affine2']  = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss(self.cnf)


    def main(self):
        '''
            main function
        '''
        try:
            # loading dataset
            (x_train, t_train), (x_test, t_test) = self.loadDataset()
            # number of data
            train_size = x_train.shape[0]
            train_loss_list, train_acc_list, test_acc_list = [], [], []
            iter_per_epoch = max(train_size // self.cnf.batch_size, 1)
            iters = self.cnf.epoch * iter_per_epoch

            for i in range(1,iters+1):
                # select batch-data
                batch_mask = np.random.choice(train_size, self.cnf.batch_size)
                x_batch = x_train[batch_mask].copy()
                t_batch = t_train[batch_mask].copy()

                # calculate gradient
                #grad = network.numerical_gradient(x_batch, t_batch)
                grad = self.gradient(x_batch, t_batch)

                # update parameters
                self.updateParameters(grad)

                # calculate loss function
                loss = self.loss(x_batch, t_batch)

                # record accuracy
                if (i % iter_per_epoch) == 0:
                    train_loss_list.append(loss)
                    train_acc = self.accuracy(x_train, t_train)
                    test_acc = self.accuracy(x_test, t_test)
                    train_acc_list.append(train_acc)
                    test_acc_list.append(test_acc)
                    print('{} epoch : {}[%]\t {}[%]'.format(str(i//iter_per_epoch).rjust(6), round(train_acc*100,3) ,round(test_acc*100,3)))

            # plot figure
            self.plotFigure(test_acc_list, train_acc_list, train_loss_list)
            # save experimental data
            self.saveExperimentalData({'train_acc': train_acc_list , 'test_acc': test_acc_list, 'train_loss': train_loss_list})

        except Exception as e:
            print('Error : {}'.format(e))


    def loadDataset(self):
        '''
            load dataset from URL
        '''
        df = pd.read_csv(self.cnf.dataset_url, header=None)
        # error
        if not df.shape[0] == (sum(self.cnf.dataset_ratio.values())):
            print('Error : dataset_ratio does not match loading dataset')
            return
        rand_index = np.random.permutation(range(df.shape[0]))
        bound = self.cnf.dataset_ratio['train']
        # select [sepal_length, sepal_width, petal_length, petal_width]
        x_all = df[self.cnf.dataset_index['dec']].values
        t_all_origin = df[self.cnf.dataset_index['obj']].values
        # transform one-hot vector
        t_all = []
        for i in range(len(t_all_origin)):
            t_all.append(self.cnf.dataset_one_hot_vector[t_all_origin[i]])
        t_all = np.array(t_all)
        # separate data
        x_train = x_all[rand_index[:bound]]
        x_test  = x_all[rand_index[bound:]]
        t_train = t_all[rand_index[:bound]]
        t_test  = t_all[rand_index[bound:]]

        return (x_train, t_train), (x_test, t_test)


    def plotFigure(self, test_acc_list, train_acc_list, train_loss_list):
        folder_name = 'graph'
        path_graph = self.cnf.path_out + '/' + folder_name
        # make directory
        if not os.path.isdir(path_graph):
            os.makedirs(path_graph)
        x_axis = range(1, self.cnf.epoch + 1)

        # graph-1 : accuracy
        title1 = 'accuracy' + '_' + self.cnf_name
        fig1 = plt.figure(figsize=(10,6))
        ax1 = fig1.add_subplot(1,1,1)
        ax1.plot(x_axis, train_acc_list, color='green', alpha=0.7, label='train', linewidth = 1.0)
        ax1.plot(x_axis, test_acc_list, color='orange', alpha=0.7, label='test', linewidth = 1.0)
        ax1.set_title(title1)
        ax1.set_xlim(0, self.cnf.epoch)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')
        ax1.legend()
        fig1.savefig(path_graph + '/' + title1 + ".png" , dpi=300)

        # graph-2 : loss-function
        title2 = 'loss-function' + '_' + self.cnf_name
        fig2 = plt.figure(figsize=(10,6))
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(x_axis, train_loss_list, linewidth = 1.0)
        ax2.set_title(title2)
        ax2.set_ylim(0.0)
        ax2.set_xlim(0, self.cnf.epoch)
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('loss-function-value')
        fig2.savefig(path_graph + '/' + title2 + ".png" , dpi=300)

        # plt.show()


    def saveExperimentalData(self, data_dict=None):
        '''
            save experimental data
        '''
        if not data_dict is None:
            folder_name = 'table'
            path_table = self.cnf.path_out + '/' + folder_name
            # make directory
            if not os.path.isdir(path_table):
                os.makedirs(path_table)
            df = pd.DataFrame(data_dict.values(), index=data_dict.keys(), columns=range(1,self.cnf.epoch+1)).T
            df.to_csv(path_table + '/' + 'experimental-data_' + self.cnf_name + '.csv')


    def predict(self, x):
        '''
            predict solution (x→y)
        '''
        for layer in self.layers.values():
            x = layer.forward(x)
        return x



    def loss(self, x, t):
        '''
            calculate loss function
            ( x:input-data, t:label-data)
        '''
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def updateParameters(self, grad):
        '''
            update parameters
        '''
        for key in self.params.keys():
            if self.cnf.learning_method == 'SGD':
                self.params[key] -= self.cnf.learning_rate * grad[key]
            else:
                print('Error : learning_method is invalid value.')
                return


    def accuracy(self, x, t):
        '''
            calculate accuracy
            ( x:input-data, t:label-data)
        '''
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def cal_numerical_gradient(self, f, x):
        '''
            numerical differential
            ( f:loss-function, x:input-data)
        '''
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x) # f(x+h)
            x[idx] = tmp_val - h
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)

            x[idx] = tmp_val # 値を元に戻す
            it.iternext()

        return grad


    def numerical_gradient(self, x, t):
        '''
            gradient by numerical differential
            ( x:input-data, t:label-data)
        '''
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = self.cal_numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self.cal_numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self.cal_numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self.cal_numerical_gradient(loss_W, self.params['b2'])

        return grads


    def gradient(self, x, t):
        '''
            gradient by back-propagation
            ( x:input-data, t:label-data)
        '''
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # result
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


# --------
#  Layers
# --------
class Relu:
    '''
        ReLU Layer
    '''
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    '''
        Sigmoid Layer
    '''
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    '''
        Affine Layer
    '''
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # differential of wight(W), bias(b)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        # restore shape of input-data
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    '''
        Softmax with Loss Layer
    '''
    def __init__(self, cnf):
        self.cnf    = cnf
        self.loss   = None
        # output of softmax
        self.y      = None
        # label-data
        self.t      = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        if self.cnf.loss_function == 'sum-squared-error' :
            self.loss = sum_squared_error(self.y, self.t)
        if self.cnf.loss_function == 'cross-entropy-error' :
            self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # the case of label-data is one-hot-vector
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.main()
