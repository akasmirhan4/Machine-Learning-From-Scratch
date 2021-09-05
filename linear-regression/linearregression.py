import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def getBestFitLine(xs, ys):
    #   getBestFitLine return m, b
    #   m = (x_mean * y_mean - xy_mean) / (x_mean^2 - x_sq_mean)
    #   b = y_mean - m * x_mean

    numerator = (np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)
    denominator = np.mean(xs)**2 - np.mean(xs**2)
    m = numerator/denominator
    b = np.mean(ys) - m * np.mean(xs)
    return m, b

def sqError(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def getRsq(origLine, regLine):
    meanLine = [np.mean(origLine) for y in origLine]
    sqErrorRegLine = sqError(origLine, regLine)
    sqErrorMeanLine = sqError(origLine, meanLine)
    return 1 - (sqErrorRegLine / sqErrorMeanLine)

m, b = getBestFitLine(xs, ys)

regLine = [(m*x) + b for x in xs]

x_predict = 8
y_predict = (m*x_predict) + b

r_squared = getRsq(ys, regLine)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(x_predict, y_predict, color="g")
plt.plot(xs, regLine)
plt.show()
