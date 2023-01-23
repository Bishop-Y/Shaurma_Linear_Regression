import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

data = pd.read_csv("Regression_1.csv", sep=';')

y = np.array(data["qsh"])
x = np.array([data["psh"]]).reshape((-1, 1))

# количество наблюдений
n = len(data)

# создание модели
model = LinearRegression()
model.fit(x, y)

#################################
print("a)")
a = model.intercept_
b = model.coef_[0]
print(f"\ta = {a:.4f}")
print(f"\tb = {b:.4f}")
print()

#################################
print("b)")
print(f"\tqsh_i = {a:.4f}{b:.4f} * psh_i")
print()

#################################
print("c)")
TSS = np.cov(y) * (n - 1)
print(f"\tTSS = {TSS:.4f}")

y_pred = np.array(a + x * b).reshape((1, -1))[0]
RSS = np.sum(np.square(y_pred - y))
print(f"\tRSS = {RSS:.4f}")

ESS = TSS - RSS
print(f"\tESS = {ESS:.4f}")
print()

#################################
print("d)")
r_sq = model.score(x, y)
print(f"\tr_sq = {r_sq:.4f}")

k = 1  # одна независимая переменная в модели
r_sq_adj = 1 - (1 - r_sq) * ((n - 1) / (n - k - 1))
print(f"\tr_sq_adj = {r_sq_adj:.4f}")
