import numpy            as np
import pandas           as pd
from collections        import OrderedDict
import configuration    as cf

from functions          import *
from mnist              import load_mnist


class NeuralNetwork:

    def __init__(self, cnf, weight_init_std = 0.01):
        self.cnf = cnf
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(self.cnf.input_size, self.cnf.hidden_size)
        self.params['b1'] = np.zeros(self.cnf.hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(self.cnf.hidden_size, self.cnf.output_size)
        self.params['b2'] = np.zeros(self.cnf.output_size)
        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


    # Gradient by Back propagation
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


    # Gradient by Numerical Differential
    def numerical_gradient(f, x):
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


class Relu:
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
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss   = None
        self.y      = None      # softmaxの出力
        self.t      = None      # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 教師データがone-hot-vectorの場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class LearningNeuralNetwork():
    def __init__(self):
        self.dataset_url    = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    def main(self):
        '''
            main function
        '''

        self.cnf = cf.Configuration()

        # データの読み込み
        (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

        network = NeuralNetwork(self.cnf)

        train_size = x_train.shape[0]

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(self.cnf.iterations):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            # 勾配
            #grad = network.numerical_gradient(x_batch, t_batch)
            grad = network.gradient(x_batch, t_batch)

            # 更新
            for key in ('W1', 'b1', 'W2', 'b2'):
                network.params[key] -= learning_rate * grad[key]

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print(train_acc, test_acc)


    def loadDataset(self):
        df = pd.read_csv(self.dataset_url, header=None)
        print(df.head())
        print(df.tail())
        # select [sepal_length, sepal_width, petal_length, petal_width]
        x_train = df[range(4)].values
        y_train_ = df[self.cnf.dataset_index['obj']].values
        for i in range(len(y_train_)):
            
        x_test, y_test = None, None
        #return (x_train, y_train), (x_test, y_test)

    def selectBatchData(self):
        pass


if __name__ == "__main__":
    cnf = cf.Configuration()
    lnn = LearningNeuralNetwork()
    #lnn.main()
    lnn.loadDataset()
