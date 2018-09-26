from adaline import Adaline
from scipy.stats import linregress
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def main():
    xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
          3.0, 3.5, 4.0, 4.5, 5.0]
    ys = [2.26, 3.8, 4.43, 5.91, 6.18, 7.26,
          8.15, 9.14, 10.87, 11.58, 12.55]

    a_regression, b_regression, correlation_coefficient, _, regression_standard_error = linregress(xs, ys)
    regression_equation = f"y={a_regression:.5f}*x+{b_regression:.5f}"
    ys_regression = [a_regression * x + b_regression for x in xs]

    determination_coefficient = correlation_coefficient ** 2

    print(f"Regression equation: {regression_equation}")
    print(f"r = {correlation_coefficient:.5f}")
    print(f"r² = {determination_coefficient:.5f}")
    print(f"σ = {regression_standard_error:.5f}\n")

    learning_rate = 0.0015
    list_max_cycles = [100, 200, 500, 1000]
    random_starting_weights = False
    weight_adjustment_tolerance = None
    square_error_tolerance = None

    network = Adaline(learning_rate=learning_rate)
    for source, target in zip(xs, ys):
        network.add_sources([source], target)

    adaline_plots = []

    for max_cycles in list_max_cycles:
        print(f"Max cycles: {max_cycles}\n--------------------------")
        network.train(random_starting_weights=random_starting_weights, max_cycles=max_cycles,
                      weight_adjustment_tolerance=weight_adjustment_tolerance,
                      square_error_tolerance=square_error_tolerance,
                      verbose=False)

        a_adaline, b_adaline = network.neuron.weights[0], network.neuron.bias
        adaline_equation = f"y={a_adaline:.5f}*x+{b_adaline:.5f}"
        ys_adaline = [a_adaline*x + b_adaline for x in xs]

        total_square_error = sum([(y - y_line)**2 for y, y_line in zip(ys, ys_adaline)])

        adaline_standard_error = (total_square_error/len(ys))**0.5

        print(f"Adaline equation: {adaline_equation}\n")

        print(f"Difference for a coefficient: {abs(a_adaline - a_regression):.5f}")
        print(f"Difference for b coefficient: {abs(b_adaline - b_regression):.5f}")
        print(f"σ = {adaline_standard_error}\n-----------------------\n")

        adaline_plots.append(plt.plot(xs, ys_adaline, linestyle='--',
                             linewidth=3, label=f"Cycles: {max_cycles}", zorder=1)[0])

    regression_plot, = plt.plot(xs, ys_regression, color='blue', linestyle='-',
                                linewidth=5, label=f"Regression: {regression_equation}", zorder=0)

    scatter_plot = plt.scatter(xs, ys, color='black', marker='x', s=80, label='Source points', zorder=2)
    plt.legend(handles=[scatter_plot, *adaline_plots, regression_plot])

    plt.show()


if __name__ == '__main__':
    main()
