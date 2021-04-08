import numpy as np
import pandas as pd
import sys


class NN:

    def load_data(self):

        train_image = ''
        train_label = ''
        test_image = ''
        if __name__ == '__main__':
            if len(sys.argv) > 1:
                train_image = sys.argv[1]
                train_label = sys.argv[2]
                test_image = sys.argv[3]
            else:
                train_image = 'train_image.csv'
                train_label = 'train_label.csv'
                test_image = 'test_image.csv'

        train_images = np.loadtxt(train_image, delimiter=",")
        train_images = train_images.T / 255
        train_labels = np.loadtxt(train_label)
        test_images = np.loadtxt(test_image, delimiter=",")
        test_images = test_images.T / 255

        return train_images, train_labels, test_images

    def __init__(self, input_nodes, hidden_layer_nodes, outputNodes, epoch, alpha):
        self.x, self.y, self.x_t = self.load_data()
        self.one_hot_encoding()

        w1 = np.random.uniform(-1.0, 1.0, size=(input_nodes * hidden_layer_nodes))
        self.w1 = w1.reshape((hidden_layer_nodes, input_nodes))
        self.b1 = np.zeros((hidden_layer_nodes))

        w2 = np.random.uniform(-1.0, 1.0, size=(hidden_layer_nodes * outputNodes))
        self.w2 = w2.reshape((outputNodes, hidden_layer_nodes))
        self.b2 = np.zeros((outputNodes))
        self.acc = []
        self.epoch = epoch
        self.alpha = alpha
        self.output = []

    def one_hot_encoding(self):
        one_hot = np.zeros((self.y.shape[0], 10))
        for i, val in enumerate(self.y):
            one_hot[i, int(val)] = 1.0
        self.y_enc = one_hot.T

    def softmax(self, x):
        ex = np.exp(x)
        sum_ex = np.sum(ex, axis=0)
        return ex / sum_ex

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derv(self, s):
        return s * (1 - s)

    def feed_forward(self, image, type='train'):
        self.z1 = self.w1.dot(image) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = self.softmax(self.z2)

    def back_propogation(self, image, y_enc, alpha):
        self.dz2 = self.a2 - y_enc
        self.dw2 = self.dz2[np.newaxis,].T.dot(self.a1[np.newaxis])
        self.db2 = self.dz2

        self.da1 = self.w2.T.dot(self.dz2)
        self.dz1 = self.da1 * self.sigmoid_derv(self.a1)
        self.dw1 = self.dz1[np.newaxis].T.dot(image[np.newaxis])
        self.db1 = self.dz1

        self.w1 -= alpha * self.dw1
        self.w2 -= alpha * self.dw2
        self.b1 -= alpha * self.b1
        self.b2 = alpha * self.b2

    def write_output(self):
        pd.DataFrame(self.output).to_csv("test_predictions.csv", index=False, header=False)


myNN = NN(784, 256, 10, 20, 0.005)
for i in range(myNN.epoch):
    epochTotal = 0
    for j in range(myNN.x.shape[1]):
        myNN.feed_forward(myNN.x[:, j])
        myNN.back_propogation(myNN.x[:, j], myNN.y_enc[:, j], myNN.alpha)

        # predict
        y_pred = np.argmax(myNN.a2, axis=0)
        if y_pred == myNN.y[j]:
            epochTotal += 1
    print('Epoch:', i, ' Training acc:', epochTotal / myNN.x.shape[1])

# test
for j in range(myNN.x_t.shape[1]):
    myNN.feed_forward(myNN.x_t[:, j])
    y_pred = np.argmax(myNN.a2, axis=0)
    myNN.output.append(y_pred)

myNN.write_output()