import main
from sklearn.neighbors import RadiusNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import sklearn.neighbors as nb
import numpy as np

X_train_standard = []
Y_train_standard = []

for i in range(0, len(main.XY_train), main.TRAIN_POINTS_AMOUNT):
    Class_Train = main.Class_train[i]
    X_average = 0
    Y_average = 0
    for j in range(main.TRAIN_POINTS_AMOUNT):
        X_average += main.XY_train[i+j][0]
        Y_average += main.XY_train[i+j][1]
    X_train_standard.append([X_average/main.TRAIN_POINTS_AMOUNT, Y_average/main.TRAIN_POINTS_AMOUNT])
    Y_train_standard.append(Class_Train)

X_train_standard = np.array(X_train_standard)
Y_train_standard = np.array(Y_train_standard)

standard = nb.KNeighborsClassifier(n_neighbors=1).fit(X_train_standard, Y_train_standard)

_, ax = plt.subplots(ncols=1, figsize=(12, 5))

X_test = main.x_test()

# ax.scatter(main.XY_train[:, 0], main.XY_train[:, 1], color='red', s=100, marker='s')

disp = DecisionBoundaryDisplay.from_estimator(
    standard,
    X_test,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    alpha=0.5,
    ax=ax,
)

# main.plot_train_points(to_plot=True)
main.plot_lines()
ax.scatter(X_train_standard[:, 0], X_train_standard[:, 1], marker="s", c="r")

found = 0
for test in X_test:
    prediction = standard.predict([test])
    color = 'r'
    if prediction == main.find_real_class(*test):
        found += 1
        color = 'g'
    plt.scatter(*test, color=color, marker='s')
print(f"Correctly found percent = {found*100.0/len(X_test)}%")

plt.show()