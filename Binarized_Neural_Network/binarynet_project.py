import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import load_model

np.seterr(divide='ignore', invalid='ignore')
np.random.seed(1234)

# activation function
def sign(x):
    x[x >= 0] = 1
    x[x < 0] = -1
    return x

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def softmax(x):  # output probability distribution function
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# deruvatives
def binary_tanh_unit(x):
    return 2. * hard_sigmoid(x) - 1.

def hard_sigmoid(x):
    return np.clip((x + 1.) / 2., 0, 1)

def sigmoid_derv(x):
    return x * (1 - x)

def tanh_derv(x):
    return 1.0 - np.tanh(x) ** 2

# loss functions
def cross_entropy(p, r):
    n_samples = r.shape[0]
    o = p - r
    return o / n_samples

def mean_squared_error(inputs, y):
    loss = np.divide(np.square(np.subtract(inputs, y)),2)
    return loss

def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):  # for weights
    stddev = np.sqrt(stddev_factor / np.sqrt(prev_units + num_units))
    return np.random.normal(0, stddev, [prev_units, num_units])

#apply functions for different activation_funcs and their derivatives
def apply_active(activation_func, data):
    if activation_func == 'relu':
        a1 = relu(data)
    elif activation_func == 'sigmoid':
        a1 = sigmoid(data)
    elif activation_func == 'softmax':
        a1 = softmax(data)
    elif activation_func == 'sign':
        a1 = sign(data)
    elif activation_func == 'tanh':
        a1 = np.tanh(data)
    else:
        print("Error! Activation Function does not exist!")
        a1 = data
    return a1

def apply_derivative(activation_func, data):
    if activation_func == 'relu':
        d1 = relu(data)
    elif activation_func == 'sigmoid':
        d1 = sigmoid_derv(data)
    elif activation_func == 'sign':
        d1 = binary_tanh_unit(data)
    elif activation_func == 'tanh':
        d1 = tanh_derv(data)
    else:
        print("Error! Activation Function does not exist!")
        d1 = data
    return d1

def calculate_loss(loss_function, y, labels_batch):
    if loss_function == "cross_entropy":
        loss1 = cross_entropy(y, labels_batch)
    elif loss_function == "mean_squared_error":
        loss1 = mean_squared_error(y, labels_batch)
    else:
        print("Error! Loss Function does not exist!")
        loss1 = y - labels_batch
    return loss1

class Network:
    y_acc = []     # for graph
    x_epoch = []   # for graph

    def __init__(self,
               num_neuron_in_layers,
               batch_size,
               num_epochs,
               learning_rate, activation_funcition, loss_function
               ):
        self.num_nodes_in_layers = num_neuron_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight1 = []  # _glorot_initializer(self, self.num_nodes_in_layers[0], self.num_nodes_in_layers[1])
        self.bias1 = []  # np.zeros((1, self.num_nodes_in_layers[1]))
        self.activation_function_1 = activation_funcition[0]
        self.weight2 = []  # _glorot_initializer(self, self.num_nodes_in_layers[1], self.num_nodes_in_layers[2])
        self.bias2 = []  # np.zeros((1, self.num_nodes_in_layers[2]))
        self.activation_function_2 = activation_funcition[1]
        self.weight3 = []  # _glorot_initializer(self, self.num_nodes_in_layers[2], self.num_nodes_in_layers[3])
        self.bias3 = []  # np.zeros((1, self.num_nodes_in_layers[3]))
        self.activation_function_3 = activation_funcition[2]
        self.create_model()
        if self.activation_function_1 == 'sign':
            self.binarize()
        self.loss_function = loss_function
        self.loss = []
        self.best_weight1 = self.weight1.copy()
        self.best_weight2 = self.weight2.copy()
        self.best_weight3 = self.weight3.copy()
        self.max_acc = 0

    def train(self, inputs, labels):

        for epoch in range(self.num_epochs):  # training begins
            iteration = 0
            total_iterations = len(inputs) // self.batch_size
            for i in range(total_iterations):
                # batch input
                inputs_batch = inputs[i * self.batch_size:i * self.batch_size + self.batch_size]
                labels_batch = labels[i * self.batch_size:i * self.batch_size + self.batch_size]

                # forward pass
                hidden_1 = np.dot(inputs_batch, self.weight1) + self.bias1
                hidden_1_activation = apply_active(self.activation_function_1, hidden_1)
                hidden_2 = np.dot(hidden_1_activation, self.weight2) + self.bias2
                hidden_2_activation = apply_active(self.activation_function_2, hidden_2)
                hidden_3 = np.dot(hidden_2_activation, self.weight3) + self.bias3
                y = apply_active(self.activation_function_3, hidden_3) 
                
                #train_acc = float(np.sum(np.argmax(y, 1) == np.argmax(labels_batch,1))) / float(len(labels_batch))
                #print("train acc",train_acc)

                # calculate loss
                calculated_loss = calculate_loss(self.loss_function, y, labels_batch)
                self.loss.append(calculated_loss)
                # backward pass
                y_delta = calculated_loss
                hidden_2_delta = np.dot(y_delta, self.weight3.T)
                hidden_2_activation_delta = hidden_2_delta * apply_derivative(self.activation_function_2,
                                                                              hidden_2_activation)
                hidden_1_delta = np.dot(hidden_2_activation_delta, self.weight2.T)
                hidden_1_delta_activation = hidden_1_delta * apply_derivative(self.activation_function_1,
                                                                              hidden_1_activation)
                #update weight and bias
                self.weight3 -= self.learning_rate * np.dot(hidden_2_activation.T, y_delta) 
                self.bias3 = self.bias3 - self.learning_rate * np.sum(y_delta, axis=0, keepdims=True)
                self.weight2 -= self.learning_rate * np.dot(hidden_1_activation.T, hidden_2_activation_delta)
                self.bias2 -= self.learning_rate * np.sum(hidden_2_activation_delta, axis=0)
                self.weight1 -= self.learning_rate * np.dot(inputs_batch.T, hidden_1_delta_activation)
                self.bias1 -= self.learning_rate * np.sum(hidden_1_delta_activation, axis=0)

                if self.activation_function_1 == 'sign':
                    self.binarize()

                iteration += self.batch_size
            self.val(x_val, y_val, epoch)

    def val(self, inputs, labels, index):
        print("Validating...epoch", index)
        hidden_layer1 = apply_active(self.activation_function_1, np.dot(inputs, self.weight1) + self.bias1)
        hidden_layer2 = apply_active(self.activation_function_2, np.dot(hidden_layer1, self.weight2) + self.bias2)
        output_layer = apply_active(self.activation_function_3, np.dot(hidden_layer2, self.weight3) + self.bias3)
        acc = float(np.sum(np.argmax(output_layer, 1) == labels)) / float(len(labels))
        print(acc)
        if acc > self.max_acc:
            self.max_acc = acc
            self.best_weight1 = self.weight1.copy()
            self.best_weight2 = self.weight2.copy()
            self.best_weight3 = self.weight3.copy()
        self.y_acc.append(acc)

        if index == self.num_epochs - 2:
            for i in range(self.num_epochs):
                self.x_epoch.append(i)
            plt.plot(self.x_epoch, self.y_acc)
            plt.title('Accuracy vs. each Epoch')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.show()

    def test(self, inputs, labels):
        hidden_layer = apply_active(self.activation_function_1, np.dot(inputs, self.best_weight1) + self.bias1)
        probs = apply_active(self.activation_function_2, np.dot(hidden_layer, self.best_weight2) + self.bias2)
        probss = apply_active(self.activation_function_3, np.dot(probs, self.best_weight3) + self.bias3)
        acc = float(np.sum(np.argmax(probss, 1) == labels)) / float(len(labels))
        print(acc)

    def create_model(self):
        global a
        model = load_model('keras_mnisttanhs.h5')
        a = model.get_weights()
        self.load_weights(a)

    def load_weights(self, a):
        self.weight1 = a[0].copy()
        self.bias1 = a[1].copy()
        self.weight2 = a[2].copy()
        self.bias2 = a[3].copy()
        self.weight3 = a[4].copy()
        self.bias3 = a[5].copy()

    def binarize(self):
        # binarization for initial and updated weights
        self.weight1 = sign(self.weight1)
        self.weight2 = sign(self.weight2)
        self.weight3 = sign(self.weight3)


# load MNIST and MNIST Fashion data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

print("Training...")
num_classes = 10

# data processing
input_size = 784
X_train = X_train.reshape(X_train.shape[0], input_size).astype('float32')
x_train = X_train[:50000] / 255
x_val = X_train[50000:] / 255
y_train = np.eye(num_classes)[Y_train[:50000]]
y_val = Y_train[50000:]
X_test = X_test.reshape(X_test.shape[0], input_size).astype('float32')
x_test = X_test / 255
y_test = Y_test

net = Network(num_neuron_in_layers=[input_size, 500, 1000, num_classes],
              batch_size=200,
              num_epochs=100,
              learning_rate=0.0001,
              activation_funcition=["sign", "sign", "softmax"],
              loss_function="cross_entropy"
              )
net.val(x_val, y_val, "earlier") #start first val_acc with uploaded weights
net.train(x_train, y_train)      #start trainig

print("Testing...")
net.test(x_test, y_test)         #start test