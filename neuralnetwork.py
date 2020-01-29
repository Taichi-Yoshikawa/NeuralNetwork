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


# ----------------
#  Neural Network
# ----------------
class NeuralNetwork:
    '''
        3-Layer Neural Network
    '''
    def __init__(self, seed=1):
        # configuration
        self.seed = seed  if isinstance(seed,(range,list)) else [seed]
        self.cnf = cf.Configuration()
        self.random = np.random
        self.time = tm.strftime('%Y-%m-%d_%H-%M-%S')
        self.cnf_name = '_'.join([self.cnf.learning_method, self.cnf.loss_function, 'lr='+str(self.cnf.learning_rate)])


    def main(self):
        '''
            main function
        '''
        try:
            for i in range(len(self.seed)):
                # timer start
                start_time = tm.time()
                # set seed-value of numpy.random
                self.random.seed(self.seed[i])
                # initialize
                self.initialize()
                # loading dataset
                (x_train, t_train), (x_test, t_test) = self.loadDataset(i)
                # number of data
                train_size = x_train.shape[0]
                train_loss_list, train_acc_list, test_acc_list = [], [], []
                iter_per_epoch = max(train_size // self.cnf.batch_size, 1)
                iters = self.cnf.epoch * iter_per_epoch

                for j in range(1,iters+1):
                    # select batch-data
                    batch_mask = self.random.choice(train_size, self.cnf.batch_size)
                    x_batch = x_train[batch_mask].copy()
                    t_batch = t_train[batch_mask].copy()

                    # calculate gradient
                    if self.cnf.gradient_method == 'bp':
                        grad = self.gradient(x_batch, t_batch)
                    elif self.cnf.gradient_method == 'num':
                        grad = self.numerical_gradient(x_batch, t_batch)
                    else:
                        print('Error : gradient_method is invalid value.')

                    # update parameters
                    self.updateParameters(grad, j)

                    # calculate loss function
                    loss = self.loss(x_batch, t_batch)

                    # record accuracy
                    if (j % iter_per_epoch) == 0:
                        train_loss_list.append(loss)
                        train_acc = self.accuracy(x_train, t_train)
                        test_acc = self.accuracy(x_test, t_test)
                        train_acc_list.append(train_acc)
                        test_acc_list.append(test_acc)
                        print('{} epoch : {}[%]\t {}[%]'.format(str(j//iter_per_epoch).rjust(6), round(train_acc*100,3) ,round(test_acc*100,3)))

                # plot figure
                # self.plotFigure(test_acc_list, train_acc_list, train_loss_list)
                # save experimental data
                self.saveExperimentalData(i, {'train_acc': train_acc_list , 'test_acc': test_acc_list, 'train_loss': train_loss_list})

                # timer end
                end_time = tm.time()
                exe_time = end_time - start_time
                print('[ exe-time : {}[s] ]'.format(round(exe_time,1)))

            # summarize experimental data
            self.summarizeExperimentalData()

        except Exception as e:
            print('Error : {}'.format(e))


    def initialize(self):
        # initialize parameters
        self.params = {}
        if self.cnf.init_method == 'Xavier':
            std = [ np.sqrt(1./self.cnf.input_size), np.sqrt(1./self.cnf.hidden_size)]
        elif self.cnf.init_method == 'He':
            std = [ np.sqrt(2./self.cnf.input_size), np.sqrt(2./self.cnf.hidden_size)]
        else:
            print('Error : initial_method is invalid value.')
        self.params['W1'] = self.random.normal(0,std[0],(self.cnf.input_size, self.cnf.hidden_size))
        self.params['b1'] = np.zeros(self.cnf.hidden_size)
        self.params['W2'] = self.random.normal(0,std[1],(self.cnf.hidden_size, self.cnf.output_size))
        self.params['b2'] = np.zeros(self.cnf.output_size)
        self.v, self.h, self.m = None, None, None
        # Generate Layers
        self.layers = od()
        self.layers['Affine1']  = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']    = Relu()
        self.layers['Affine2']  = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss(self.cnf)


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


    def updateParameters(self, grad, iter):
        '''
            update parameters
        '''
        if self.cnf.learning_method == 'SGD':
            for key in self.params.keys():
                self.params[key] -= self.cnf.learning_rate * grad[key]
        elif self.cnf.learning_method == 'Momentum':
            if self.v is None:
                self.momentum = 0.9
                self.v = {}
                for key, val in self.params.items():
                    self.v[key] = np.zeros_like(val)
            for key in self.params.keys():
                self.v[key] = self.momentum * self.v[key] - self.cnf.learning_rate * grad[key]
                self.params[key] += self.v[key]
        elif self.cnf.learning_method == 'AdaGrad':
            if self.h is None:
                self.h = {}
                for key, val in self.params.items():
                    self.h[key] = np.zeros_like(val)
            for key in self.params.keys():
                self.h[key] += grad[key] * grad[key]
                self.params[key] -= self.cnf.learning_rate * grad[key] / (np.sqrt(self.h[key]) + 1e-7)
        elif self.cnf.learning_method == 'Adam':
            if self.m is None:
                self.m, self.v = {}, {}
                self.beta = [0.9, 0.999]
                for key,val in self.params.items():
                    self.m[key], self.v[key] = np.zeros_like(val), np.zeros_like(val)
            for key in self.params.keys():
                self.m[key] = self.beta[0] * self.m[key] + (1. - self.beta[0]) * grad[key]
                self.v[key] = self.beta[1] * self.v[key] + (1. - self.beta[1]) * (grad[key]**2)
                m_hat = self.m[key] / (1. - self.beta[0]**iter)
                v_hat = self.v[key] / (1. - self.beta[1]**iter)
                self.params[key] -= self.cnf.learning_rate * m_hat / ( np.sqrt(v_hat) + 1e-7 )
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


    def _numerical_gradient(self, f, x):
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
        grads['W1'] = self._numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self._numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self._numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self._numerical_gradient(loss_W, self.params['b2'])

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


    def loadDataset(self, i):
        '''
            load dataset from URL
        '''
        if (i==0) and not (os.path.exists(self.cnf.path_out + '/' + self.cnf.dataset_url.split('/')[-1])) :
            df = pd.read_csv(self.cnf.dataset_url, header=None)
            df.to_csv(self.cnf.path_out + '/' + self.cnf.dataset_url.split('/')[-1], header=False, index=False)
        else:
            df = pd.read_csv(self.cnf.path_out + '/' + self.cnf.dataset_url.split('/')[-1], header=None)
        # error
        if not df.shape[0] == (sum(self.cnf.dataset_ratio.values())):
            print('Error : dataset_ratio does not match loading dataset')
            return
        rand_index = self.random.permutation(range(df.shape[0]))
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


    def saveExperimentalData(self, i, data_dict=None):
        '''
            save experimental data
        '''
        if not data_dict is None:
            folder_name_t = 'table'
            self.path_table = self.cnf.path_out + '/' + folder_name_t + '/' + self.cnf_name
            # make directory
            if not os.path.isdir(self.path_table):
                os.makedirs(self.path_table)
            df = pd.DataFrame(data_dict.values(), index=data_dict.keys(), columns=range(1,self.cnf.epoch+1)).T
            df.to_csv(self.path_table + '/' + 'experimental-data_' + self.cnf_name + '_seed=' + str(self.seed[i]) + '.csv')


    def summarizeExperimentalData(self):
        '''
            summarize experimental data
        '''
        folder_name_g = 'graph'
        self.path_graph = self.cnf.path_out + '/' + folder_name_g
        # make directory
        if not os.path.isdir(self.path_graph):
            os.makedirs(self.path_graph)

        df = {}
        for i in range(len(self.seed)):
            # read experimental data
            dt = pd.read_csv(self.path_table + '/' + 'experimental-data_' + self.cnf_name + '_seed=' + str(self.seed[i]) + '.csv', index_col = 0)
            self.dt_columns = dt.columns.values
            for col in self.dt_columns:
                # make dataframe
                if i == 0:
                    df[col] = pd.DataFrame({'seed={}'.format(self.seed[i]) : np.array(dt[col])}, index = dt.index)
                else:
                    df[col]['seed={}'.format(self.seed[i])] = np.array(dt[col])

        # calculate statistics-value
        for col in self.dt_columns:
            _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
            for i in range(len(df[col].index)):
                dtset   = np.array(df[col].loc[df[col].index[i]])
                res     = np.percentile(dtset, [25, 50, 75])
                _min.append(dtset.min())
                _max.append(dtset.max())
                _q25.append(res[0])
                _med.append(res[1])
                _q75.append(res[2])
                _ave.append(dtset.mean())
                _std.append(np.std(dtset))

            # make dataframe for statistics
            _out = pd.DataFrame({
                'min' : np.array(_min),
                'q25' : np.array(_q25),
                'med' : np.array(_med),
                'q75' : np.array(_q75),
                'max' : np.array(_max),
                'ave' : np.array(_ave),
                'std' : np.array(_std)
                },index = df[col].index)

            # save summarized experimental data
            _out.to_csv(self.path_table + '/' + 'summarized-experimental-data_' + col + '_' + self.cnf_name + '.csv')

            # plot figure
            if 'acc' in col:
                title = 'accuracy_' + self.cnf_name
                if 'train' in col:
                    fig = plt.figure(figsize=(10, 6))
                    ax  = fig.add_subplot(1,1,1)
                    ax.plot(_out.index, _med , color='green', alpha=0.7, label='train', linewidth = 1.0)
                    ax.fill_between(_out.index, _q25, _q75, facecolor='green', alpha=0.1)
                    ax.set_title(title)
                    ax.set_xlabel('epochs')
                    ax.set_ylabel('accuracy')
                    ax.set_xlim(0, self.cnf.epoch)
                    ax.set_ylim(0.0, 1.0)
                elif 'test' in col:
                    ax.plot(_out.index, _med , color='orange', alpha=0.7, label='test', linewidth = 1.0)
                    ax.fill_between(_out.index, _q25, _q75, facecolor='orange', alpha=0.1)
                    ax.legend()
                    # save figure
                    fig.savefig(self.path_graph + '/' + title + '.png', dpi=300)
                    plt.close()
            elif 'loss' in col:
                title = 'loss-function_' + self.cnf_name
                fig = plt.figure(figsize=(10, 6))
                ax  = fig.add_subplot(1,1,1)
                ax.plot(_out.index, _med , color='blue', alpha=0.7, label='train', linewidth = 1.0)
                ax.fill_between(_out.index, _q25, _q75, facecolor='blue', alpha=0.1)
                ax.set_title(title)
                ax.set_xlabel('epochs')
                ax.set_ylabel('loss-function-value')
                ax.set_xlim(0, self.cnf.epoch)
                ax.set_ylim(0.0)
                # save figure
                fig.savefig(self.path_graph + '/' + title + '.png', dpi=300)
                plt.close()


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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        out = self.sigmoid(x)
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

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        # prevent from overflow
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def sum_squared_error(self, y, t):
        return 0.5 * np.sum((y-t)**2)

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        # correct index
        if t.size == y.size:
            t = t.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        if self.cnf.loss_function == 'sum-squared-error' :
            self.loss = self.sum_squared_error(self.y, self.t)
        if self.cnf.loss_function == 'cross-entropy-error' :
            self.loss = self.cross_entropy_error(self.y, self.t)

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
    nn = NeuralNetwork(range(10))
    nn.main()
