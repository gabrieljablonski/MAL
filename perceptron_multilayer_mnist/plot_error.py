import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


file = open('error_logs.txt')
lines = file.readlines()
file.close()

cycles = []
errors = []

for line in lines:
    cycle, error = line.split('\t')
    cycles.append(int(cycle))
    errors.append(float(error))


plt.plot(cycles, errors, 'r')  # 'ro', markersize=3)
plt.show()
