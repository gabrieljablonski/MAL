from adaline import Adaline
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def activation_function(input_):
    output = input_
    return output


def main():
    sources_list = [[ 1,  1],
                    [-1,  1],
                    [ 1, -1],
                    [-1, -1]]
    targets = [1, 1, 1, -1]

    square_errors = {}

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 0.35, 0.4]
    weight_adjustment_tolerance = None
    square_error_tolerance = None
    max_cycles = 50

    network = Adaline(activation_function=activation_function)
    for sources, target in zip(sources_list, targets):
        network.add_sources(sources, target)

    for learning_rate in learning_rates:
        network.learning_rate = learning_rate

        network.train(random_starting_weights=False, max_cycles=max_cycles,
                      weight_adjustment_tolerance=weight_adjustment_tolerance,
                      square_error_tolerance=square_error_tolerance)

        print(
            f">>Learning rate: {learning_rate}\n\n"
            f"Final weights:\n"
            f"{[float(f'{weigth:.5f}') for weigth in network.neuron.weights]}\n"
            f"Final bias:\n"
            f"{network.neuron.bias:.5f}\n\n"
            f"Cycles: {network.cycles}\n"
            f"Final square error: {network.total_square_error_by_cycle[-1]:.5f}\n\n\n"
        )

        square_errors[learning_rate] = network.total_square_error_by_cycle

    curves = []
    for learning_rate, square_error in square_errors.items():
        curves.append(plt.plot(range(len(square_error)), square_error, '--', linewidth=2, label=str(learning_rate))[0])
    plt.ylim([-0.1, 4])
    plt.legend(handles=curves)
    plt.show()


if __name__ == '__main__':
    main()
