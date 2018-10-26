from random import uniform
from numpy import exp, sin, linspace, pi
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def binary_sigmmoid(x: float) -> float:
    y = 1. / (1 + exp(-x))

    return y


def d_binary_sigmoid(x: float) -> float:
    _x = binary_sigmmoid(x)
    y = _x * (1 - _x)

    return y


def bipolar_sigmoid(x: float) -> float:
    y = 2. / (1 + exp(-x)) - 1

    return y


def d_bipolar_sigmoid(x: float) -> float:
    _x = bipolar_sigmoid(x)
    y = 0.5 * (1 - _x ** 2)

    return y


def f(x):
    return sin(x) * sin(2 * x)


xs = linspace(-pi, pi, 50)
ys = f(xs)

sources = [[x] for x in xs]
targets = ys

N_INPUTS = 1
N_HIDDEN_NEURONS = 10
LEARNING_RATE = 0.25
MAX_CYCLES = 2000
SQUARE_ERROR_TOLERANCE = 0.00

DEFAULT_ACTIVATION = 1
RANDOM_WEIGHT_INIT = True
WEIGHT_INIT_RANGE = (-.2, .2)

ACTIVATION_FUNCTION = bipolar_sigmoid
D_ACTIVATION_FUNCTION = d_bipolar_sigmoid

output_weights = [uniform(*WEIGHT_INIT_RANGE) if RANDOM_WEIGHT_INIT else 0
                  for _ in range(N_HIDDEN_NEURONS)]
output_bias = uniform(*WEIGHT_INIT_RANGE) if RANDOM_WEIGHT_INIT else 0

hidden_weights = [
    [uniform(*WEIGHT_INIT_RANGE) for _ in range(N_INPUTS)] if RANDOM_WEIGHT_INIT else [0, 0]
    for _ in range(N_HIDDEN_NEURONS)
]
hidden_biases = [
    uniform(*WEIGHT_INIT_RANGE) if RANDOM_WEIGHT_INIT else 0
    for _ in range(N_HIDDEN_NEURONS)
]

total_square_error_by_cycle = []

current_cycle = 0
while True:
    total_square_error = 0
    current_cycle += 1

    for source, target in zip(sources, targets):
        hidden_net_values = []
        hidden_outputs = []

        for neuron_i, bias in zip(range(N_HIDDEN_NEURONS), hidden_biases):
            net_value = bias
            for s, weight in zip(source, hidden_weights[neuron_i]):
                net_value += s * weight
            output = ACTIVATION_FUNCTION(net_value)

            hidden_net_values.append(net_value)
            hidden_outputs.append(output)

        net_value = output_bias
        for h_output, weight in zip(hidden_outputs, output_weights):
            net_value += h_output * weight
        output = ACTIVATION_FUNCTION(net_value)

        total_square_error += 0.5 * (output - target) ** 2

        output_layer_net_value = net_value
        output_layer_output = output

        output_error = (target - output_layer_output) * D_ACTIVATION_FUNCTION(output_layer_net_value)

        output_weight_adjustments = []
        for weight_i in range(N_HIDDEN_NEURONS):
            adj = LEARNING_RATE * output_error * hidden_outputs[weight_i]
            output_weight_adjustments.append(adj)

        output_bias_adjustment = LEARNING_RATE * output_error * DEFAULT_ACTIVATION

        hidden_errors = []
        for neuron_i, weight, net_value in zip(range(N_HIDDEN_NEURONS), output_weights, hidden_net_values):
            error = output_error * weight * D_ACTIVATION_FUNCTION(net_value)
            hidden_errors.append(error)

        hidden_weight_adjustments = []
        hidden_bias_adjustments = []

        for error in hidden_errors:
            hidden_weight_adjustments.append([])
            for s in source:
                adj = LEARNING_RATE * error * s
                hidden_weight_adjustments[-1].append(adj)

            adj = LEARNING_RATE * error * DEFAULT_ACTIVATION
            hidden_bias_adjustments.append(adj)

        new_output_weights = []
        for weight_i, old_weight in enumerate(output_weights):
            new_weight = old_weight + output_weight_adjustments[weight_i]
            new_output_weights.append(new_weight)
        new_output_bias = output_bias + output_bias_adjustment

        new_hidden_weights = []
        new_hidden_biases = []
        for neuron_i, old_weights in enumerate(hidden_weights):
            new_hidden_weights.append([])
            for weight_i, old_weight in enumerate(old_weights):
                new_weight = old_weight + hidden_weight_adjustments[neuron_i][weight_i]
                new_hidden_weights[-1].append(new_weight)
            new_bias = hidden_biases[neuron_i] + hidden_bias_adjustments[neuron_i]
            new_hidden_biases.append(new_bias)

        output_weights = new_output_weights
        output_bias = new_output_bias
        hidden_weights = new_hidden_weights
        hidden_biases = new_hidden_biases

    total_square_error /= len(sources)
    total_square_error_by_cycle.append(total_square_error)

    print(
        f"Cycle: {current_cycle}\n"
        f"Total square error: {total_square_error}\n"
        f"---------------------------------------------"
    )

    if current_cycle >= MAX_CYCLES:
        print(f"Max cycles reached: {current_cycle}")
        break

    if total_square_error < SQUARE_ERROR_TOLERANCE:
        print(f"Square error tolerance reached: {total_square_error}")
        break

outputs = []
for source, target in zip(sources, targets):
    hidden_net_values = []
    hidden_outputs = []
    for neuron_i, bias in zip(range(N_HIDDEN_NEURONS), hidden_biases):
        net_value = bias
        for s, weight in zip(source, hidden_weights[neuron_i]):
            net_value += s * weight
        output = ACTIVATION_FUNCTION(net_value)

        hidden_net_values.append(net_value)
        hidden_outputs.append(output)

    net_value = output_bias
    for h_output, weight in zip(hidden_outputs, output_weights):
        net_value += h_output * weight
    output = ACTIVATION_FUNCTION(net_value)

    print(
        f"\nSource: {source}\n"
        f"Expected: {target}\n"
        f"Output: {output}\n"
    )
    outputs.append(output)

plt.figure(1)
plt.plot(range(1, current_cycle + 1), total_square_error_by_cycle, 'r*')

plt.figure(2)
plt.plot(xs, ys, 'b--')
plt.plot(xs, outputs, 'r-')
plt.show()
