import sklearn.neighbors as nb
import numpy as np
import main
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

NEIGHBOURS = 10

nbrs = nb.KNeighborsClassifier(n_neighbors=NEIGHBOURS)

nbrs.fit(main.XY_train, main.Class_train)

_, ax = plt.subplots(ncols=1, figsize=(12, 5))

X_test = main.x_test()

# ax.scatter(main.XY_train[:, 0], main.XY_train[:, 1], color='red', s=100, marker='s')

disp = DecisionBoundaryDisplay.from_estimator(
    nbrs,
    X_test,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    alpha=0.5,
    ax=ax,
)
main.plot_lines()
# main.plot_train_points()

found = 0
for test in X_test:
    prediction = nbrs.predict([test])
    color = 'r'
    if prediction == main.find_real_class(*test):
        found += 1
        color = 'g'
    plt.scatter(*test, color=color, marker='s')
print(f"Correctly found percent = {found*100.0/len(X_test)}%")

plt.show()