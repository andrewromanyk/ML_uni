import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import LeaveOneOut

import matplotlib
matplotlib.use('TkAgg')

random.seed(456)

ROWS = 5
COLS = 4

ROWS_START = 1
ROWS_END = ROWS_START + ROWS

COLS_START = 1
COLS_END = COLS_START + COLS

ROWS_SPANS = np.linspace(ROWS_START, ROWS_END, ROWS+1)
COLS_SPANS = np.linspace(COLS_START, COLS_END, COLS+1)

COLOR = 'lightblue'

TRAIN_POINTS_AMOUNT = 50

epsilon = 1e-10

TEST_AMOUNT = 1000

def plot_lines():
    for row in ROWS_SPANS:
        plt.plot([row, row], [COLS_START, COLS_END], color=COLOR)

    for col in COLS_SPANS:
        plt.plot([ROWS_START, ROWS_END], [col, col], color=COLOR)

ROWS_PAIRS = []
for row in ROWS_SPANS[:-1]:
    ROWS_PAIRS.append((row, row+1))

COLS_PAIRS = []
for col in COLS_SPANS[:-1]:
    COLS_PAIRS.append((col, col+1))

def get_random_coordinates(start, end):
    return random.uniform(start+epsilon, end)

X_train = []
Y_train = []
Class_train = []
def plot_train_points(to_plot=False):
    for (x1, x2) in ROWS_PAIRS:
        for (y1, y2) in COLS_PAIRS:
            X_train_local = []
            Y_train_local = []
            for i in range(TRAIN_POINTS_AMOUNT):
                X_train_local.append(get_random_coordinates(x1, x2))
                Y_train_local.append(get_random_coordinates(y1, y2))
                Class_train.append(str(int(x1)) + "." + str(int(y1)))
            if (to_plot):
                plt.scatter(X_train_local, Y_train_local)
            X_train.extend(X_train_local)
            Y_train.extend(Y_train_local)

def find_real_class(x, y):
    left = ""
    right = ""
    for i in range(len(ROWS_PAIRS)):
        a, b = ROWS_PAIRS[i]
        if a < x and x <= b:
            left = str(i+1)
    for j in range(len(COLS_PAIRS)):
        a, b = COLS_PAIRS[j]
        if a < y and y <= b:
            right = str(j+1)
    return left+'.'+right

def x_test():
    X_test = []
    for i in range(TEST_AMOUNT):
        x = get_random_coordinates(ROWS_START, ROWS_END)
        y = get_random_coordinates(COLS_START, COLS_END)
        X_test.append([x, y])

    return np.array(X_test)

plot_train_points()

def main():
    plot_lines()
    plt.show()

if __name__ == "__main__":
    main()

XY_train = np.array([[a, b] for a, b in zip(X_train, Y_train)])

def full_leave_one_out(model, X):
    hits = 0
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        inner_x_train = [X[i] for i in train]
        inner_y_train = [find_real_class(*x) for x in inner_x_train]
        testing = X[test[0]]
        model.fit(inner_x_train, inner_y_train)
        if model.predict([testing]) == find_real_class(*testing):
            hits += 1

    return hits / len(X)