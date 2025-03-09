import sklearn.neighbors as nb
import numpy as np
import main
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

NEIGHBOURS = 1

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
greens = []
reds = []
for test in X_test:
    prediction = nbrs.predict([test])
    if prediction == main.find_real_class(*test):
        found += 1
        greens.append(test)
    else:
        reds.append(test)
plt.scatter([x for x, _ in greens], [x for _, x in greens], color='g', marker='s')
plt.scatter([x for x, _ in reds], [x for _, x in reds], color='r', marker='s')

print(f"Correctly found percent = {found*100.0/len(X_test)}%")

def pick_best_neighbour():
    max = 0
    k = 0
    for i in range(1, 15):
        knn = nb.KNeighborsClassifier(n_neighbors=i)
        rate = main.full_leave_one_out(knn, X_test)
        print(f"Prediction rate for {i} neighbours is {rate}")
        if rate > max:
            max = rate
            k = i
    print(f"The best neighbour is {k}")

plt.show()

pick_best_neighbour()