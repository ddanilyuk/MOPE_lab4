import math
import numpy as np
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial
from random import randint
from prettytable import PrettyTable


# *********** Змінні за варіантом ***********
m = 3
N = 8

X1max = 40
X1min = 10
X2max = 35
X2min = -15
X3max = 5
X3min = -15

Xmax_average = (X1max + X2max + X3max) / 3
Xmin_average = (X1min + X2min + X3min) / 3

y_max = round(200 + Xmax_average)
y_min = round(200 + Xmin_average)



def plan_matrix(n, m):
    """
    Функція для знаходження матриці планування
    """
    y = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            y[i][j] = randint(y_min, y_max)

    array = [[1, -1, -1, -1],
            [1, -1,  1,  1],
            [1,  1, -1,  1],
            [1,  1,  1, -1],
            [1, -1, -1,  1],
            [1, -1,  1, -1],
            [1,  1, -1, -1],
            [1,  1,  1,  1]]
    x_norm = np.array(array)

    for row in array:
        row.append(row[1]*row[2])
        row.append(row[1]*row[3])
        row.append(row[2]*row[3])

    x_norm_vzaemodia = np.array(array)

    # print(x_norm_vzaemodia)

    # x_norm = x_norm_vzaemodia
    print(x_norm)
                
    # x_norm = x_norm[:len(y)]

    x_range = [(X1min, X1max), (X1min, X1max), (X1min, X1max)]
    x = np.ones(shape=(len(x_norm), len(x_norm[0])))

    for i in range(len(x_norm)):
        for j in range(1, len(x_norm[i])):
            if x_norm[i][j] == -1:
                x[i][j] = x_range[j-1][0]
            else:
                x[i][j] = x_range[j-1][1]

    x_vzaemodia = np.ones(shape=(len(x_norm), len(x_norm_vzaemodia[0])))

    for i in range(len(x_norm_vzaemodia)):
        x_vzaemodia[i][0] = x[i][1] * x[i][2]
        x_vzaemodia[i][1] = x[i][1] * x[i][3]
        x_vzaemodia[i][2] = x[i][2] * x[i][3]
        x_vzaemodia[i][3] = x[i][1] * x[i][2] * x[i][3]

    print('\nМатриця планування звичайна')
    matrix = np.concatenate((x,y),axis=1)
    table = PrettyTable()
    
    yFieldNames = []
    yFieldNames += (f'Y{i+1}' for i in range(m))

    fieldNames = ["X0c", "X1c", "X2c", "X3c"] + yFieldNames
    table.field_names = fieldNames

    for i in range(len(matrix)):
        table.add_row(matrix[i])

    print(table)

    print('\nМатриця планування з ефектом взаємодії')
    matrix = np.concatenate((x, x_norm_vzaemodia ,y),axis=1)
    table = PrettyTable()
    
    yFieldNames = []
    yFieldNames += (f'Y{i+1}' for i in range(m))

    fieldNames = ["X0c", "X1c", "X2c", "X3c", "x12, x13, x23, x123"] + yFieldNames
    table.field_names = fieldNames

    for i in range(len(matrix)):
        table.add_row(matrix[i])

    print(table)

    return x, y


def getRandomY():
    y1, y2, y3 = [], [], []

    for _ in range(0, 8):
        y1.append(randint(y_min, y_max))
        y2.append(randint(y_min, y_max))
        y3.append(randint(y_min, y_max))

    return y1, y2, y3


def getYRows(y1, y2, y3):
    y_rows = []
    for i in range(8):
        y_rows.append([y1[i], y2[i], y3[i]])
    
    return y_rows


def getYAverage(y_rows) -> list:
    y_average_all = []

    for i in range(len(y_rows)):
        y_average_all.append(np.average(y_rows[i]))

    return y_average_all

def main(m):
    N = 8
    fisher = partial(f.ppf, q=1-0.05)
    student = partial(t.ppf, q=1-0.025)

    # y
    y1, y2, y3 = getRandomY()

    # y_rows
    y_rows = getYRows(y1, y2, y3)

    # середнє значення y_rows
    y_average = getYAverage(y_rows)

    disp_list = []

    list_bi = []

    
    x, y = plan_matrix(N, m)
    # list_bi = find_coefficient(x, y_average, N)
    # disp_list = getDispersion(y)





if __name__ == "__main__":
    main(m)
