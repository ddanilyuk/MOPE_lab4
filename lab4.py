import math
import numpy as np
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial
from random import randint
from prettytable import PrettyTable


def cohren(f1, f2, q=0.05):
    q1 = q / f1
    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fisher_value / (fisher_value + f1 - 1)


fisher = partial(f.ppf, q=1-0.05)
student = partial(t.ppf, q=1-0.025)


X1max = -5
X1min = -25
X2max = -10
X2min = -70
X3max = -5
X3min = -25

Xmax_average = (X1max + X2max + X3max)/3
Xmin_average = (X1min + X2min + X3min)/3

y_max = round(200 + Xmax_average)
y_min = round(200 + Xmin_average)

x0_factor = [1, 1, 1, 1, 1, 1, 1, 1]
x1_factor = [-1, -1, 1, 1, -1, -1, 1, 1]
x2_factor = [-1, 1, -1, 1, -1, 1, -1, 1]
x3_factor = [-1, 1, 1, -1, 1, -1, -1, 1]
x1x2_factor = [a*b for a, b in zip(x1_factor, x2_factor)]
x1x3_factor = [a*b for a, b in zip(x1_factor, x3_factor)]
x2x3_factor = [a*b for a, b in zip(x2_factor, x3_factor)]
x1x2x3_factor = [a*b*c for a, b, c in zip(x1_factor, x2_factor, x3_factor)]

m = 3

y1, y2, y3 = [], [], []
for i in range(0, 8):
    y1.append(randint(y_min, y_max))
    y2.append(randint(y_min, y_max))
    y3.append(randint(y_min, y_max))

Y_row1 = [y1[0], y2[0], y3[0]]
Y_row2 = [y1[1], y2[1], y3[1]]
Y_row3 = [y1[2], y2[2], y3[2]]
Y_row4 = [y1[3], y2[3], y3[3]]
Y_row5 = [y1[4], y2[4], y3[4]]
Y_row6 = [y1[5], y2[5], y3[5]]
Y_row7 = [y1[6], y2[6], y3[6]]
Y_row8 = [y1[7], y2[7], y3[7]]

Y_average1 = np.average(Y_row1)
Y_average2 = np.average(Y_row2)
Y_average3 = np.average(Y_row3)
Y_average4 = np.average(Y_row4)
Y_average5 = np.average(Y_row5)
Y_average6 = np.average(Y_row6)
Y_average7 = np.average(Y_row7)
Y_average8 = np.average(Y_row8)
Y_average = [round(Y_average1, 3), round(Y_average2, 3), round(Y_average3, 3), round(Y_average4, 3),
             round(Y_average5, 3), round(Y_average6, 3), round(Y_average7, 3), round(Y_average8, 3)]

x0 = [1, 1, 1, 1, 1, 1, 1, 1]
x1 = [-25, -25, -5, -5, -25, -25, -5, -5]
x2 = [-70, -10, -70, -10, -70, -10, -70, -10]
x3 = [-25, -5, -5, -25, -5, -25, -25, -5]
x1x2 = [a*b for a, b in zip(x1, x2)]
x1x3 = [a*b for a, b in zip(x1, x3)]
x2x3 = [a*b for a, b in zip(x2, x3)]
x1x2x3 = [a*b*c for a, b, c in zip(x1, x2, x3)]

list_for_solve_b = [x0_factor, x1_factor, x2_factor, x3_factor, x1x2_factor, x1x3_factor, x2x3_factor, x1x2x3_factor]
list_for_solve_a = list(zip(x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3))

N = 8
list_bi = []
for k in range(N):
    S = 0
    for i in range(N):
        S += (list_for_solve_b[k][i]*Y_average[i])/N
    list_bi.append(round(S, 5))

Disp1 = 0
Disp2 = 0
Disp3 = 0
Disp4 = 0
Disp5 = 0
Disp6 = 0
Disp7 = 0
Disp8 = 0
for i in range(m):
    Disp1 += ((Y_row1[i] - np.average(Y_row1))**2)/m
    Disp2 += ((Y_row2[i] - np.average(Y_row2))**2)/m
    Disp3 += ((Y_row3[i] - np.average(Y_row3))**2)/m
    Disp4 += ((Y_row4[i] - np.average(Y_row4))**2)/m
    Disp5 += ((Y_row5[i] - np.average(Y_row5))**2)/m
    Disp6 += ((Y_row6[i] - np.average(Y_row6))**2)/m
    Disp7 += ((Y_row7[i] - np.average(Y_row7))**2)/m
    Disp8 += ((Y_row8[i] - np.average(Y_row8))**2)/m
sum_dispersion = Disp1 + Disp2 + Disp3 + Disp4 + Disp5 + Disp6 + Disp7 + Disp8
disp_list = [round(Disp1, 3), round(Disp2, 3), round(Disp3, 3), round(Disp4, 3), round(Disp5, 3),  round(Disp6, 3),
             round(Disp7, 3), round(Disp8, 3)]

pt1 = PrettyTable()
column_names1 = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3", "Y", "S^2"]
pt1.add_column(column_names1[0], x0_factor)
pt1.add_column(column_names1[1], x1_factor)
pt1.add_column(column_names1[2], x2_factor)
pt1.add_column(column_names1[3], x3_factor)
pt1.add_column(column_names1[4], x1x2_factor)
pt1.add_column(column_names1[5], x1x3_factor)
pt1.add_column(column_names1[6], x2x3_factor)
pt1.add_column(column_names1[7], x1x2x3_factor)
pt1.add_column(column_names1[8], y1)
pt1.add_column(column_names1[9], y2)
pt1.add_column(column_names1[10], y3)
pt1.add_column(column_names1[11], Y_average)
pt1.add_column(column_names1[12], disp_list)
print(pt1, "\n")

print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 \n".format(list_bi[0], list_bi[1],
                                                                                           list_bi[2], list_bi[3],
                                                                                           list_bi[4], list_bi[5],
                                                                                           list_bi[6], list_bi[7]))

pt2 = PrettyTable()
pt2.add_column(column_names1[0], x0)
pt2.add_column(column_names1[1], x1)
pt2.add_column(column_names1[2], x2)
pt2.add_column(column_names1[3], x3)
pt2.add_column(column_names1[4], x1x2)
pt2.add_column(column_names1[5], x1x3)
pt2.add_column(column_names1[6], x2x3)
pt2.add_column(column_names1[7], x1x2x3)
pt2.add_column(column_names1[8], y1)
pt2.add_column(column_names1[9], y2)
pt2.add_column(column_names1[10], y3)
pt2.add_column(column_names1[11], Y_average)
pt2.add_column(column_names1[12], disp_list)
print(pt2, '\n')

list_ai = [round(i, 5) for i in solve(list_for_solve_a, Y_average)]
print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(list_ai[0], list_ai[1],
                                                                                        list_ai[2], list_ai[3],
                                                                                        list_ai[4], list_ai[5],
                                                                                        list_ai[6], list_ai[7]))

Gp = max(Disp1, Disp2, Disp3, Disp4, Disp5, Disp6, Disp7, Disp8) / sum_dispersion
F1 = m-1
N = len(y1)
F2 = N
Gt = cohren(F1, F2)
print("\nGp = ", Gp, " Gt = ", Gt)
if Gp < Gt:
    print("Дисперсія однорідна!\n")

    Dispersion_B = sum_dispersion / N
    Dispersion_beta = Dispersion_B / (m * N)
    S_beta = math.sqrt(abs(Dispersion_beta))

    beta0 = 0
    beta1 = 0
    beta2 = 0
    beta3 = 0
    beta4 = 0
    beta5 = 0
    beta6 = 0
    beta7 = 0
    for i in range(len(x0_factor)):
        beta0 += (Y_average[i] * x0_factor[i]) / N
        beta1 += (Y_average[i] * x1_factor[i]) / N
        beta2 += (Y_average[i] * x2_factor[i]) / N
        beta3 += (Y_average[i] * x3_factor[i]) / N
        beta4 += (Y_average[i] * x1x2_factor[i]) / N
        beta5 += (Y_average[i] * x1x3_factor[i]) / N
        beta6 += (Y_average[i] * x2x3_factor[i]) / N
        beta7 += (Y_average[i] * x1x2x3_factor[i]) / N
    beta_list = [beta0, beta1, beta2, beta3, beta4, beta5, beta6, beta7]

    t0 = abs(beta0) / S_beta
    t1 = abs(beta1) / S_beta
    t2 = abs(beta2) / S_beta
    t3 = abs(beta3) / S_beta
    t4 = abs(beta4) / S_beta
    t5 = abs(beta5) / S_beta
    t6 = abs(beta6) / S_beta
    t7 = abs(beta7) / S_beta
    t_list = [t0, t1, t2, t3, t4, t5, t6, t7]

    F3 = F1 * F2
    d = 0
    T = student(df=F3)
    print("t табличне = ", T)
    for i in range(len(t_list)):
        if t_list[i] < T:
            beta_list[i] = 0
            print("Гіпотеза підтверджена, beta{} = 0".format(i))
        else:
            print("Гіпотеза не підтверджена.\nbeta{} = {}".format(i, beta_list[i]))
            d += 1

    y_1 = beta_list[0] + beta_list[1] * x1[0] + beta_list[2] * x2[0] + beta_list[3] * x3[0] + beta_list[4] * x1x2[0] \
          + beta_list[5]*x1x3[0] + beta_list[6]*x2x3[0] + beta_list[7]*x1x2x3[0]
    y_2 = beta_list[0] + beta_list[1] * x1[1] + beta_list[2] * x2[1] + beta_list[3] * x3[1] + beta_list[4] * x1x2[1] \
          + beta_list[5] * x1x3[1] + beta_list[6] * x2x3[1] + beta_list[7] * x1x2x3[1]
    y_3 = beta_list[0] + beta_list[1] * x1[2] + beta_list[2] * x2[2] + beta_list[3] * x3[2] + beta_list[4] * x1x2[2] \
          + beta_list[5] * x1x3[2] + beta_list[6] * x2x3[2] + beta_list[7] * x1x2x3[2]
    y_4 = beta_list[0] + beta_list[1] * x1[3] + beta_list[2] * x2[3] + beta_list[3] * x3[3] + beta_list[4] * x1x2[3] \
          + beta_list[5] * x1x3[3] + beta_list[6] * x2x3[3] + beta_list[7] * x1x2x3[3]
    y_5 = beta_list[0] + beta_list[1] * x1[4] + beta_list[2] * x2[4] + beta_list[3] * x3[4] + beta_list[4] * x1x2[4] \
          + beta_list[5] * x1x3[4] + beta_list[6] * x2x3[4] + beta_list[7] * x1x2x3[4]
    y_6 = beta_list[0] + beta_list[1] * x1[5] + beta_list[2] * x2[5] + beta_list[3] * x3[5] + beta_list[4] * x1x2[5] \
          + beta_list[5] * x1x3[5] + beta_list[6] * x2x3[5] + beta_list[7] * x1x2x3[5]
    y_7 = beta_list[0] + beta_list[1] * x1[6] + beta_list[2] * x2[6] + beta_list[3] * x3[6] + beta_list[4] * x1x2[6] \
          + beta_list[5] * x1x3[6] + beta_list[6] * x2x3[6] + beta_list[7] * x1x2x3[6]
    y_8 = beta_list[0] + beta_list[1] * x1[7] + beta_list[2] * x2[7] + beta_list[3] * x3[7] + beta_list[4] * x1x2[7] \
          + beta_list[5] * x1x3[7] + beta_list[6] * x2x3[7] + beta_list[7] * x1x2x3[7]
    Y_counted_for_Student = [y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8]

    F4 = N - d
    Dispersion_ad = 0
    for i in range(len(Y_counted_for_Student)):
        Dispersion_ad += ((Y_counted_for_Student[i] - Y_average[i]) ** 2) * m / (N - d)
    Fp = Dispersion_ad / Dispersion_beta
    Ft = fisher(dfn=F4, dfd=F3)
    if Fp > Ft:
        print("Рівняння регресії неадекватне.")
    else:
        print("Рівняння регресії адекватне!")

else:
    print("Дисперсія неоднорідна. Спробуйте ще раз.")
