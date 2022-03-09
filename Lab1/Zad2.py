import math
import random
import statistics

if __name__ == '__main__':

    x = [3, 8, 9, 10, 12]
    y = [8, 7, 7, 5, 6]
    z = x.copy()
    n = 0

    for i in range(len(x)):
        z[i] = x[i] + y[i]
    print(z)

    for i in range(len(x)):
        z[i] = x[i] * y[i]
    print(z)

    for i in range(len(x)):
        n += x[i] * y[i]
    print(n)

    n = 0
    for i in range(len(x)):
        n += x[i] * x[i]
    print(math.sqrt(n))

    n = 0
    for i in range(len(y)):
        n += y[i] * y[i]
    print(math.sqrt(n))

    z = []
    for i in range(50):
        z.append(random.random()*99 + 1)
    print(z)

    print(statistics.mean(z))
    print(min(z))
    print(max(z))
    print(statistics.stdev(z))

    minz = min(z)
    maxz = max(z)
    z2 = z.copy()
    for i in range(len(z)):
        z2[i] = (z[i] - minz)/(maxz - minz)
    print(z2)
    print(maxz)
    print(z.index(maxz))
    print(z2[z.index(maxz)])

    m = statistics.mean(z)
    d = statistics.stdev(z)
    z2 = z.copy()
    for i in range(len(z)):
        z2[i] = (z[i] - m)/d
    print(z2)
    print(statistics.mean(z2))
    print(statistics.stdev(z2))

    z2 = z.copy()
    for i in range(len(z)):
        for j in range(0, 100, 10):
            if (z[i] >= j) and (z[i] < j + 10):
                z2[i] = "[" + str(j) + ", " + str(j+10) + ")"
    print(z2)
