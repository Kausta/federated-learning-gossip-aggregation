from math import sqrt
from random import randint


def rmse(expected, values):
    total = 0
    for val in values:
        total += (val - expected) ** 2
    return sqrt(total / len(values))


def main():
    val = [1., 2., 3., 4., 5., 6.]
    alpha = [2., 2., 2., 1., 1., 1.]

    def iter(x, y):
        alpha[x] /= 2
        total = alpha[x] + alpha[y]
        val[y] = (alpha[y] * val[y] + alpha[x] * val[x]) / total
        alpha[y] = total

    average = 0
    total_w = 0
    for x, w in zip(val, alpha):
        average += x * w
        total_w += w
    average = average / total_w
    print("0 RMSE:", rmse(average, val), val)
    for i in range(100):
        for j in range(6):
            iter(j, randint(0, 5))
        print(i, "RMSE:", rmse(average, val), val)


if __name__ == '__main__':
    main()
