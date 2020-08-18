import random
from math import exp
import numpy as np
from PIL import Image
import sys
from progress.bar import IncrementalBar


def read_data(input_file):
    with open(input_file) as files:
        file_list = files.read().split()
        x_data = []
        x_label = []
        for file in file_list:
            raw = Image.open(file)
            pixels = np.array(raw)
            pix_array = np.concatenate(pixels, axis=None)
            pix_array = pix_array / 255
            x_data.append(list(pix_array))
            if 'down' in file:
                x_label.append(1)
            else:
                x_label.append(0)

    return x_data, x_label


def sigmoid(s):
    sig = exp(s) / (1 + exp(s))
    return sig


def feed_forward(data_point, computed_weights, hidden_layer_size):
    data_point = [1] + data_point
    x_intermitent = [1]
    for i in range(hidden_layer_size):
        x_intermitent.append(sigmoid(np.dot(data_point, computed_weights[i])))
    # print(np.dot(x_intermitent, computed_weights[hidden_layer_size]))
    x_output = sigmoid(np.dot(x_intermitent, computed_weights[hidden_layer_size]))
    return x_output


def eval_weight_via_BackPropogate(w, train_data, train_label, hidden_layer_size):
    dimension = len(train_data[0])
    eta = 0.1  # learning rate
    for i in range(len(train_data)):
        x_input = train_data[i]
        x_input = [1] + x_input
        x_intermitent = [1]
        for j in range(hidden_layer_size):
            x_intermitent.append(sigmoid(np.dot(x_input, w[j])))
        x_output = sigmoid(np.dot(x_intermitent, w[hidden_layer_size]))
        delta_L = 2 * (x_output - train_label[i]) * (x_output - x_output ** 2)
        delta_intermediate = []
        for j in range(hidden_layer_size + 1):
            delta_intermediate.append(
                ((x_intermitent[j] - x_intermitent[j] ** 2) * (w[hidden_layer_size][j] * delta_L)))

        for j in range(hidden_layer_size):
            for n in range(dimension + 1):
                w[j][n] = w[j][n] - ((eta * x_input[n]) * delta_intermediate[j + 1])

        for j in range(hidden_layer_size + 1):
            w[hidden_layer_size][j] = w[hidden_layer_size][j] - ((eta * x_intermitent[j]) * delta_L)

    return w


def prediction(x_data, x_label, computed_weights, hidden_layer_size):
    output_label = []
    for i in range(len(x_data)):
        y = feed_forward(x_data[i], computed_weights, hidden_layer_size)
        # print(y)
        if y >= 0.5:
            output_label.append(1)
        else:
            output_label.append(0)

    match = 0
    for j in range(len(output_label)):
        if output_label[j] == x_label[j]:
            match = match + 1

    '''print(output_label)
    print(train_data_label)
    print(match)'''
    accuracy = match / len(output_label)
    return output_label, accuracy


if __name__ == '__main__':
    test_file = 'downgesture_test.list'
    train_file = 'downgesture_train.list'
    x_train_data, train_data_label = read_data(train_file)
    x_test_data, test_data_label = read_data(test_file)
    d = len(x_train_data[0])
    size = 100
    # print(d)
    weights = []
    for i in range(size):
        weights.append(list((2 * np.random.random(d + 1) - 1) / 100))
    weights.append(list((2 * np.random.random(size + 1) - 1) / 100))

    epochs = 1000
    bar = IncrementalBar('Processing', max=epochs, stream=sys.stdout)
    for j in range(epochs):
        bar.next()
        weights = eval_weight_via_BackPropogate(weights, x_train_data, train_data_label, size)

    bar.finish()


    train_output, train_accuracy = prediction(x_train_data, train_data_label, weights, size)
    test_output, test_accuracy = prediction(x_test_data, test_data_label, weights, size)


    print("The output label of training",train_output)
    print("The accuracy on training data is ", train_accuracy)
    print("The output label of test data ",test_output)
    print("The accuracy on test data is ", test_accuracy)
