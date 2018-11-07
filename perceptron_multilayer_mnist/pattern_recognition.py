from random import uniform
from numpy import exp, sin, linspace, pi
import numpy as np
from PIL import Image
import pickle
from random import choice
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


def bipolar_sigmoid(x: float) -> float:
    y = 2. / (1 + exp(-x)) - 1

    return y


def d_bipolar_sigmoid(x: float) -> float:
    _x = bipolar_sigmoid(x)
    y = 0.5 * (1 - _x ** 2)

    return y


def save(what, where):
    f = open(where, 'wb')
    pickle.dump(what, f)
    f.close()


def load(where):
    f = open(where, 'rb')
    data = pickle.load(f)
    f.close()
    return data


sources, targets = load('training_samples_1000.pkl')

PICKLE_PATH = 'weights_with_1000_samples'
LOG_PATH = 'error_logs.txt'
SAVE_WEIGHTS = True
TRAIN = True
SAVE_INTERVAL = 1

N_INPUTS = len(sources[0])
N_HIDDEN_NEURONS = 200
N_OUTPUT_NEURONS = 10
LEARNING_RATE = 0.0001
MAX_CYCLES = 20000
SQUARE_ERROR_TOLERANCE = 0.00

ACTIVATED = 1
NOT_ACTIVATED = -1
DEFAULT_ACTIVATION = ACTIVATED
RANDOM_WEIGHT_INIT = True
WEIGHT_INIT_RANGE = (-.1, .1)

ACTIVATION_FUNCTION = bipolar_sigmoid
D_ACTIVATION_FUNCTION = d_bipolar_sigmoid

output_weights = [
    [uniform(*WEIGHT_INIT_RANGE) if RANDOM_WEIGHT_INIT else 0 for _ in range(N_HIDDEN_NEURONS)]
    for __ in range(N_OUTPUT_NEURONS)
]

output_biases = [
    uniform(*WEIGHT_INIT_RANGE) if RANDOM_WEIGHT_INIT else 0
    for _ in range(N_OUTPUT_NEURONS)
]

hidden_weights = [
    [uniform(*WEIGHT_INIT_RANGE) for _ in range(N_INPUTS)] if RANDOM_WEIGHT_INIT else [0, 0]
    for _ in range(N_HIDDEN_NEURONS)
]

hidden_biases = [
    uniform(*WEIGHT_INIT_RANGE) if RANDOM_WEIGHT_INIT else 0
    for _ in range(N_HIDDEN_NEURONS)
]

total_square_error_by_cycle = []


def train(reload=False, path=None):
    global hidden_weights, hidden_biases, output_weights, output_biases
    current_cycle = 0

    if reload:
        if path is None:
            print('Must provide path for reloading weights.')
            return
        hidden_weights, hidden_biases, output_weights, output_biases = load(path)

    try:
        while TRAIN:
            total_square_error = 0
            current_cycle += 1

            for i, (source, target) in enumerate(zip(sources, targets)):
                hidden_net_values = []
                hidden_outputs = []
                output_layer_net_values = []
                output_layer_outputs = []

                for neuron_i, bias in zip(range(N_HIDDEN_NEURONS), hidden_biases):
                    net_value = bias
                    for s, weight in zip(source, hidden_weights[neuron_i]):
                        net_value += s * weight
                    output = ACTIVATION_FUNCTION(net_value)

                    hidden_net_values.append(net_value)
                    hidden_outputs.append(output)

                for neuron_weights, bias in zip(output_weights, output_biases):
                    net_value = bias
                    for h_output, weight in zip(hidden_outputs, neuron_weights):
                        net_value += h_output * weight
                    output = ACTIVATION_FUNCTION(net_value)

                    output_layer_net_values.append(net_value)
                    output_layer_outputs.append(output)

                output_errors = []
                output_weight_adjustments = []
                output_bias_adjustments = []

                for neuron_i, (output, net_value) in enumerate(zip(output_layer_outputs, output_layer_net_values)):
                    output_weight_adjustments.append([])

                    t = ACTIVATED if neuron_i == int(target) else NOT_ACTIVATED
                    total_square_error += 0.5 * (output - t) ** 2

                    error = (t - output) * D_ACTIVATION_FUNCTION(net_value)
                    output_errors.append(error)

                    for weight_i in range(N_HIDDEN_NEURONS):
                        adj = LEARNING_RATE*error*hidden_outputs[weight_i]
                        output_weight_adjustments[-1].append(adj)
                    adj = LEARNING_RATE*error*DEFAULT_ACTIVATION
                    output_bias_adjustments.append(adj)

                hidden_errors = []
                for hidden_neuron in range(N_HIDDEN_NEURONS):
                    error = 0
                    for output_neuron, delta_k in enumerate(output_errors):
                        error += delta_k*output_weights[output_neuron][hidden_neuron]
                    error *= D_ACTIVATION_FUNCTION(hidden_net_values[hidden_neuron])
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
                for old_weights, adjs in zip(output_weights, output_weight_adjustments):
                    new_output_weights.append([])
                    for old_weight, adj in zip(old_weights, adjs):
                        new_weight = old_weight + adj
                        new_output_weights[-1].append(new_weight)

                new_output_biases = []
                for old_bias, adj in zip(output_biases, output_bias_adjustments):
                    new_bias = old_bias + adj
                    new_output_biases.append(new_bias)

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
                output_biases = new_output_biases
                hidden_weights = new_hidden_weights
                hidden_biases = new_hidden_biases

                print(f"source: {i+1}/{len(sources)}")

            total_square_error /= len(sources)
            total_square_error_by_cycle.append(total_square_error)

            print(
                f"Cycle: {current_cycle}\n"
                f"Total square error: {total_square_error}\n"
                f"---------------------------------------------"
            )

            path = LOG_PATH
            file = open(path, 'a')
            file.write(f"{current_cycle}\t{total_square_error}\n")
            file.close()

            if not current_cycle % SAVE_INTERVAL and SAVE_WEIGHTS:
                path = f"{PICKLE_PATH}_{current_cycle}.pkl"
                save((hidden_weights, hidden_biases, output_weights, output_biases), path)

            if current_cycle >= MAX_CYCLES:
                print(f"Max cycles reached: {current_cycle}")
                break

            if total_square_error < SQUARE_ERROR_TOLERANCE:
                print(f"Square error tolerance reached: {total_square_error}")
                break

    except KeyboardInterrupt:
        path = f"{PICKLE_PATH}_{current_cycle}.pkl"
        save((hidden_weights, hidden_biases, output_weights, output_biases), path)
        plt.plot(total_square_error_by_cycle, 'r*')
        plt.show()

    path = f"{PICKLE_PATH}_{current_cycle}.pkl"
    save((hidden_weights, hidden_biases, output_weights, output_biases), path)


def test(path, all=False, plot_wrong=False, from_input=False):
    global hidden_weights, hidden_biases, output_weights, output_biases
    hidden_weights, hidden_biases, output_weights, output_biases = load(path)

    if from_input:
        while True:
            image_path = input('Path to input image:\n>>')
            try:
                test_image = Image.open(image_path).convert(mode='L')
            except Exception as e:
                print(e)
                continue

            if test_image is None:
                print("Invalid path.")
                continue

            if np.array(test_image).shape != (28, 28):
                print("Invalid image shape.")
                continue

            source = np.array(test_image).flatten()

            hidden_net_values = []
            hidden_outputs = []
            outputs = []
            for neuron_i, bias in zip(range(N_HIDDEN_NEURONS), hidden_biases):
                net_value = bias
                for s, weight in zip(source, hidden_weights[neuron_i]):
                    net_value += s * weight
                output = ACTIVATION_FUNCTION(net_value)

                hidden_net_values.append(net_value)
                hidden_outputs.append(output)

            for output_neuron in range(N_OUTPUT_NEURONS):
                net_value = output_biases[output_neuron]
                for h_output, weight in zip(hidden_outputs, output_weights[output_neuron]):
                    net_value += h_output * weight
                output = ACTIVATION_FUNCTION(net_value)
                outputs.append(output)

            classification = outputs.index(max(outputs))

            outputs = [f"\t{index}: {(output+1)*50:.1f}%" for index, output in enumerate(outputs)]
            outputs = '\n'.join(outputs)

            print(f"Outputs:\n{outputs}")
            print(f"Classification: {classification}")
            plt.imshow(source.reshape(28, 28), cmap='gray')
            plt.show()

    sources, targets = load('testing_samples_1000.pkl')
    t_s = {i: [] for i in range(10)}

    for source, target in zip(sources, targets):
        t_s[int(target)].append(source)

    if all:
        got_right = [0 for _ in range(10)]
        file = open('acc_logs.txt', 'w')
        for index, (source, target) in enumerate(zip(sources, targets)):
            hidden_net_values = []
            hidden_outputs = []
            outputs = []
            for neuron_i, bias in zip(range(N_HIDDEN_NEURONS), hidden_biases):
                net_value = bias
                for s, weight in zip(source, hidden_weights[neuron_i]):
                    net_value += s * weight
                output = ACTIVATION_FUNCTION(net_value)

                hidden_net_values.append(net_value)
                hidden_outputs.append(output)

            for output_neuron in range(N_OUTPUT_NEURONS):
                net_value = output_biases[output_neuron]
                for h_output, weight in zip(hidden_outputs, output_weights[output_neuron]):
                    net_value += h_output * weight
                output = ACTIVATION_FUNCTION(net_value)
                outputs.append(output)

            classification = outputs.index(max(outputs))

            outputs = [f"\t{index}: {(output+1)*50:.1f}%" for index, output in enumerate(outputs)]
            outputs = '\n'.join(outputs)

            print(f"--{index+1}:")
            if classification == int(target):
                got_right[target] += 1
                m = f"Right! {classification}.\n"
                print(m)
                file.write(f"{index}\t{m}")
            else:
                m = f"Wrong! {classification} instead of {target}. \n{outputs}\n"
                print(m)
                file.write(f"{index}\t{m}")
                if plot_wrong:
                    plt.imshow(source.reshape(28, 28), cmap='gray')
                    plt.show()

        accuracy = [f"\t{digit}: {100.*n/len(s):.3f}% ({n}/{len(s)})" for digit, (n, s) in enumerate(zip(got_right, t_s.values()))]
        accuracy = '\n'.join(accuracy)
        m = f"Total accuracy: {100.*sum(got_right)/len(sources):.3f}% ({sum(got_right)}/{len(sources)})\n"  \
            f"Per digit:\n{accuracy}\n"
        print(m)
        file.write(m)
        file.close()

    else:
        while True:
            digit = input('\nWhich digit to test against:\n>>')

            target = int(digit)

            if target in t_s:
                source = choice(t_s[target])
            else:
                print('Invalid digit.\n')
                continue

            hidden_net_values = []
            hidden_outputs = []
            outputs = []
            for neuron_i, bias in zip(range(N_HIDDEN_NEURONS), hidden_biases):
                net_value = bias
                for s, weight in zip(source, hidden_weights[neuron_i]):
                    net_value += s * weight
                output = ACTIVATION_FUNCTION(net_value)

                hidden_net_values.append(net_value)
                hidden_outputs.append(output)

            for output_neuron in range(N_OUTPUT_NEURONS):
                net_value = output_biases[output_neuron]
                for h_output, weight in zip(hidden_outputs, output_weights[output_neuron]):
                    net_value += h_output * weight
                output = ACTIVATION_FUNCTION(net_value)
                outputs.append(output)

            classification = outputs.index(max(outputs))

            outputs = [f"\t{index}: {(output+1)*50:.1f}%" for index, output in enumerate(outputs)]
            outputs = '\n'.join(outputs)
            print(
                f"Expected: {target}\n"
                f"Output:\n{outputs}\n\n"
                f"The network believes that to be a '{classification}'!\n"
            )
            plt.imshow(source.reshape(28, 28), cmap='gray')
            plt.show()


def main():
    path = f"{PICKLE_PATH}_169.pkl"
    # test(path, all=False)
    test(path, all=True, plot_wrong=True)
    # test(path, from_input=True)

    # path = f"{PICKLE_PATH}_165.pkl"
    # train(reload=True, path=path)


if __name__ == '__main__':
    main()
