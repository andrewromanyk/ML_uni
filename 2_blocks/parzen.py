import main
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == '__main__':
    main.plot_lines()
    for (i, (a, b)) in enumerate(X_test):
        color = 'r'
        if pred_parzen[i] == main.find_real_class(a, b):
            color = 'g'
        plt.scatter(a, b, color=color, marker='s')
    found = 0
    for test in X_test:
        prediction = parzen_window_classification([test], main.XY_train, main.Class_train, h=0.15)
        if prediction[0] == main.find_real_class(*test):
            found += 1
    print(f"Correctly found percent = {found * 100.0 / len(X_test)}%")
    plt.show()