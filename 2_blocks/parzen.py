import main
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut

X_test = main.x_test()

def kernel_gauss(distance, h):
    return np.exp(-distance ** 2 / (2 * h ** 2))

def parzen_window_classification(test_points, train_points, train_labels, h=0.2):
    predictions = []
    for test in test_points:
        distances = np.linalg.norm(train_points - test, axis=1)
        weights = kernel_gauss(distances, h)
        weighted_votes = {}
        for i, w in enumerate(weights):
            cls = train_labels[i]
            weighted_votes[cls] = weighted_votes.get(cls, 0) + w
        predictions.append(max(weighted_votes, key=weighted_votes.get))
    return np.array(predictions)

pred_parzen = parzen_window_classification(X_test, main.XY_train, main.Class_train, h=0.15)

def full_leave_one_out_parzen(h, X):
    hits = 0
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        inner_x_train = [X[i] for i in train]
        inner_y_train = [main.find_real_class(*x) for x in inner_x_train]
        testing = X[test[0]]
        # train_points = [[a.tolist(), b] for a, b in zip(inner_x_train, inner_y_train)]
        # print(train_points)
        predict = parzen_window_classification([testing], inner_x_train, inner_y_train, h)
        if predict == main.find_real_class(*testing):
            hits += 1

    return hits / len(X)

def pick_best_window():
    max = 0
    h = 0
    for i in np.arange(0.1, 0.601, 0.05):
        i = round(i, 2)
        rate = full_leave_one_out_parzen(i, X_test)
        print(f"Prediction rate for {i} window is {rate}")
        if rate > max:
            max = rate
            k = i
    print(f"The best window is {k}")

if __name__ == '__main__':
    main.plot_lines()
    for (i, (a, b)) in enumerate(X_test):
        color = 'r'
        if pred_parzen[i] == main.find_real_class(a, b):
            color = 'g'
        plt.scatter(a, b, color=color, marker='s')
    found = 0
    for test in X_test:
        prediction = parzen_window_classification([test], main.XY_train, main.Class_train, h=0.1)
        if prediction[0] == main.find_real_class(*test):
            found += 1
    print(f"Correctly found percent = {found * 100.0 / len(X_test)}%")
    plt.show()
    pick_best_window()

