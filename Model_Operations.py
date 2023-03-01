import pandas as pd
import random
import numpy as np
import math
from tkinter import messagebox


training_set = []
evaluating_set = []
weights = []
neurons_outs = []
Data = pd.read_csv('IrisData.txt')


def split_data():
    global training_set, evaluating_set
    training_set = []
    evaluating_set = []

    c1_train = random.sample(range(0, 50), 30)
    c2_train = random.sample(range(50, 100), 30)
    c3_train = random.sample(range(100, 150), 30)
    training_set = c1_train + c2_train + c3_train

    for i in range(0, 150):
        if i not in training_set:
            evaluating_set.append(i)

    random.shuffle(training_set)
    random.shuffle(evaluating_set)


def initialize(no_hidden, no_neurons, bias):
    global weights, neurons_outs
    weights = []
    neurons_outs = []

    for i in range(int(no_hidden)):
        local_outs = []
        for j in range(int(no_neurons[i])):
            local_outs.append(0.0)
        neurons_outs.append(local_outs)
    loc_outs = [0.0, 0.0, 0.0]
    neurons_outs.append(loc_outs)

    for i in range(int(no_hidden)):
        local_weights = []
        for j in range(int(no_neurons[i])):
            if i == 0:
                count = 5
                random_generator = np.random.RandomState(1)
                weight = random_generator.normal(loc=0.0, scale=0.001, size=count)
                if int(bias) == 0:
                    weight[len(weight) - 1] = 0
                local_weights.append(weight)
            else:
                count = int(no_neurons[i - 1]) + 1
                random_generator = np.random.RandomState(1)
                weight = random_generator.normal(loc=0.0, scale=0.002, size=count)
                if int(bias) == 0:
                    weight[len(weight) - 1] = 0
                local_weights.append(weight)
        weights.append(local_weights)
    loc_weights = []
    for i in range(3):
        try:
            count = int(no_neurons[int(no_hidden) - 1]) + 1
        except:
            count = 5
        random_generator = np.random.RandomState(1)
        weight = random_generator.normal(loc=0.0, scale=0.003, size=count)
        if int(bias) == 0:
            weight[len(weight) - 1] = 0
        loc_weights.append(weight)
    weights.append(loc_weights)


def activate(output, active_sort):
    ret = 0.0

    if active_sort == "Sigmoid":
        if output >= 0:
            ret = float(1 / (1 + math.exp(-output)))
        else:
            ret = float(1 - 1 / (1 + math.exp(output)))
    else:
        ret = float((1 - math.exp(-output)) / (1 + math.exp(-output)))

    return ret


def derivative(out, activation):
    if activation == "Sigmoid":
        return float(out * (1.0 - out))
    else:
        return float((1 - out) * (1 + out))


def train(no_hidden, no_neurons, eta, epochs, bias, activation):
    initialize(no_hidden, no_neurons, bias)
    neurons_errors = neurons_outs.copy()

    for i in range(int(epochs)):
        for j in training_set:
            x1 = Data.loc[j, 'X1']
            x2 = Data.loc[j, 'X2']
            x3 = Data.loc[j, 'X3']
            x4 = Data.loc[j, 'X4']
            cls = Data.loc[j, 'Class']

            # forward
            for k in range(int(no_hidden)):
                for h in range(int(no_neurons[k])):
                    if k == 0:
                        neurons_outs[k][h] = float(
                            np.dot(np.array([x1, x2, x3, x4]).T, np.array([weights[k][h][0], weights[k][h][1],
                                                                           weights[k][h][2], weights[k][h][3]])))
                        neurons_outs[k][h] = activate(float(neurons_outs[k][h] + weights[k][h][4]), activation)
                    else:
                        neurons_outs[k][h] = activate(float(
                            np.dot(np.array(neurons_outs[k - 1] + [1]).T, np.array(weights[k][h]))), activation)
            if cls == "Iris-setosa":
                taken = []
                if int(no_hidden) == 0:
                    taken = [x1, x2, x3, x4, 1]
                else:
                    taken = neurons_outs[len(neurons_outs) - 2] + [1]

                neurons_outs[len(neurons_outs) - 1][0] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][0]))), activation)
                neurons_errors[len(neurons_errors) - 1][0] = (1 - neurons_outs[len(neurons_outs) - 1][0]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][0], activation)

                neurons_outs[len(neurons_outs) - 1][1] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][1]))), activation)
                neurons_errors[len(neurons_errors) - 1][1] = (0 - neurons_outs[len(neurons_outs) - 1][1]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][1], activation)

                neurons_outs[len(neurons_outs) - 1][2] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][2]))), activation)
                neurons_errors[len(neurons_errors) - 1][2] = (0 - neurons_outs[len(neurons_outs) - 1][2]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][2], activation)

            elif cls == "Iris-versicolor":
                taken = []
                if int(no_hidden) == 0:
                    taken = [x1, x2, x3, x4, 1]
                else:
                    taken = neurons_outs[len(neurons_outs) - 2] + [1]

                neurons_outs[len(neurons_outs) - 1][0] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][0]))), activation)
                neurons_errors[len(neurons_errors) - 1][0] = (0 - neurons_outs[len(neurons_outs) - 1][0]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][0], activation)

                neurons_outs[len(neurons_outs) - 1][1] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][1]))), activation)
                neurons_errors[len(neurons_errors) - 1][1] = (1 - neurons_outs[len(neurons_outs) - 1][1]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][1], activation)

                neurons_outs[len(neurons_outs) - 1][2] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][2]))), activation)
                neurons_errors[len(neurons_errors) - 1][2] = (0 - neurons_outs[len(neurons_outs) - 1][2]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][2], activation)
            else:
                taken = []
                if int(no_hidden) == 0:
                    taken = [x1, x2, x3, x4, 1]
                else:
                    taken = neurons_outs[len(neurons_outs) - 2] + [1]

                neurons_outs[len(neurons_outs) - 1][0] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][0]))), activation)
                neurons_errors[len(neurons_errors) - 1][0] = (0 - neurons_outs[len(neurons_outs) - 1][0]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][0], activation)

                neurons_outs[len(neurons_outs) - 1][1] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][1]))), activation)
                neurons_errors[len(neurons_errors) - 1][1] = (0 - neurons_outs[len(neurons_outs) - 1][1]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][1], activation)

                neurons_outs[len(neurons_outs) - 1][2] = activate(float(np.dot(
                    np.array(taken).T, np.array(weights[len(weights) - 1][2]))), activation)
                neurons_errors[len(neurons_errors) - 1][2] = (1 - neurons_outs[len(neurons_outs) - 1][2]) * derivative(
                    neurons_outs[len(neurons_outs) - 1][2], activation)

            # backward
            for k in range(int(no_hidden) - 1, -1, -1):
                for h in range(int(no_neurons[k])):
                    errors = np.array(neurons_errors[k + 1])
                    weighs = []
                    for q in weights[k + 1]:
                        weighs.append(q[h])
                    weighs = np.array(weighs)
                    neurons_errors[k][h] = derivative(neurons_outs[k][h], activation) * np.dot(errors.T, weighs)

            # forward
            b = 0
            if int(bias) == 0:
                b = 1
            prev = [x1, x2, x3, x4, 1]
            for k in range(int(no_hidden)):
                for h in range(int(no_neurons[k])):
                    for q in range(len(weights[k][h]) - b):
                        if k == 0:
                            weights[k][h][q] += float(float(eta) * neurons_errors[k][h] * prev[q])
                        else:
                            try:
                                weights[k][h][q] += float(float(eta) * neurons_errors[k][h] * neurons_outs[k - 1][q])
                            except:
                                weights[k][h][q] += float(float(eta) * neurons_errors[k][h] * 1)
            for k in range(3):
                for h in range(len(weights[len(weights) - 1][k]) - b):
                    if int(no_hidden) != 0:
                        try:
                            weights[len(weights) - 1][k][h] += float(float(eta) * neurons_errors[len(neurons_errors) - 1][k] * neurons_outs[len(neurons_outs) - 2][h])
                        except:
                            weights[len(weights) - 1][k][h] += float(float(eta) * neurons_errors[len(neurons_errors) - 1][k] * 1)
                    else:
                        weights[len(weights) - 1][k][h] += float(
                            float(eta) * neurons_errors[len(neurons_errors) - 1][k] *
                            prev[h])


def mp(cl_name):
    if cl_name == 'Iris-setosa':
        return 0
    elif cl_name == 'Iris-versicolor':
        return 1
    else:
        return 2


def evaluate(no_hidden, no_neurons, activation):
    robust_classification = 0
    cl = []
    row = [0, 0, 0]
    cl.append(row)
    row1 = [0, 0, 0]
    cl.append(row1)
    row2 = [0, 0, 0]
    cl.append(row2)

    for i in evaluating_set:
        x1 = Data.loc[i, 'X1']
        x2 = Data.loc[i, 'X2']
        x3 = Data.loc[i, 'X3']
        x4 = Data.loc[i, 'X4']
        cls = Data.loc[i, 'Class']

        for k in range(int(no_hidden)):
            for h in range(int(no_neurons[k])):
                if k == 0:
                    neurons_outs[k][h] = float(
                        np.dot(np.array([x1, x2, x3, x4]).T, np.array([weights[k][h][0], weights[k][h][1],
                                                                       weights[k][h][2], weights[k][h][3]])))
                    neurons_outs[k][h] = activate(float(neurons_outs[k][h] + weights[k][h][4]), activation)
                else:
                    neurons_outs[k][h] = activate(float(
                        np.dot(np.array(neurons_outs[k - 1] + [1]).T, np.array(weights[k][h]))), activation)
        taken = []
        if int(no_hidden) == 0:
            taken = [x1, x2, x3, x4, 1]
        else:
            taken = neurons_outs[len(neurons_outs) - 2] + [1]

        c1 = activate(float(np.dot(
            np.array(taken).T, np.array(weights[len(weights) - 1][0]))), activation)
        c2 = activate(float(np.dot(
            np.array(taken).T, np.array(weights[len(weights) - 1][1]))), activation)
        c3 = activate(float(np.dot(
            np.array(taken).T, np.array(weights[len(weights) - 1][2]))), activation)

        if c1 > c2 and c1 > c3:
            predicted = "Iris-setosa"
        elif c2 > c1 and c2 > c3:
            predicted = "Iris-versicolor"
        else:
            predicted = "Iris-virginica"

        if predicted == cls:
            robust_classification = robust_classification + 1

        cl[mp(cls)][mp(predicted)] = cl[mp(cls)][mp(predicted)] + 1

    print("Accuracy = " + str(float(robust_classification / 60) * 100) + "%")
    print("Confusion_Matrix = ")
    print("     C1", "     C2", "     C3")
    print("C1   " + str(cl[0][0]) + "      " + str(cl[0][1]) + "        ", str(cl[0][2]))
    print("C2   " + str(cl[1][0]) + "      " + str(cl[1][1]) + "        ", str(cl[1][2]))
    print("C3   " + str(cl[2][0]) + "      " + str(cl[2][1]) + "        ", str(cl[2][2]))


def testing(x1, x2, x3, x4, no_hidden, no_neurons, activation):
    for k in range(int(no_hidden)):
        for h in range(int(no_neurons[k])):
            if k == 0:
                neurons_outs[k][h] = float(
                    np.dot(np.array([float(x1), float(x2), float(x3), float(x4)]).T, np.array([weights[k][h][0], weights[k][h][1],
                                                                   weights[k][h][2], weights[k][h][3]])))
                neurons_outs[k][h] = activate(float(neurons_outs[k][h] + weights[k][h][4]), activation)
            else:
                neurons_outs[k][h] = activate(float(
                    np.dot(np.array(neurons_outs[k - 1] + [1]).T, np.array(weights[k][h]))), activation)
    taken = []
    if int(no_hidden) == 0:
        taken = [float(x1), float(x2), float(x3), float(x4), 1]
    else:
        taken = neurons_outs[len(neurons_outs) - 2] + [1]

    c1 = activate(float(np.dot(
        np.array(taken).T, np.array(weights[len(weights) - 1][0]))), activation)
    c2 = activate(float(np.dot(
        np.array(taken).T, np.array(weights[len(weights) - 1][1]))), activation)
    c3 = activate(float(np.dot(
        np.array(taken).T, np.array(weights[len(weights) - 1][2]))), activation)

    if c1 > c2 and c1 > c3:
        predicted = "Iris-setosa"
    elif c2 > c1 and c2 > c3:
        predicted = "Iris-versicolor"
    else:
        predicted = "Iris-virginica"

    messagebox.showinfo("Info", predicted)
