import numpy as np
from perceptron import Perceptron, ACTIVATED, NOT_ACTIVATED

GRID_HEIGHT = 5
GRID_WIDTH = 5
GRID_SHAPE = (GRID_WIDTH, GRID_HEIGHT)
LEARNING_RATE = 0.1
THRESHOLD = 0


def activation_function(input_):
    if input_ >= THRESHOLD:
        return ACTIVATED
    else:
        return NOT_ACTIVATED


def main():
    x_letter = np.full(GRID_SHAPE, NOT_ACTIVATED, dtype=np.int8)
    np.fill_diagonal(x_letter, ACTIVATED)
    x_letter = np.fliplr(x_letter)
    np.fill_diagonal(x_letter, ACTIVATED)
    x_flat = x_letter.flatten()

    t_letter = np.full(GRID_SHAPE, NOT_ACTIVATED, dtype=np.int8)
    t_letter[0, :] = [ACTIVATED]*GRID_WIDTH
    mid_col = GRID_SHAPE[1] // 2
    t_letter[:, mid_col] = [ACTIVATED]*GRID_HEIGHT
    t_flat = t_letter.flatten()

    targets = {"x": ACTIVATED, "t": NOT_ACTIVATED}

    network = Perceptron(LEARNING_RATE, activation_function)
    
    network.add_sources(x_flat, targets["x"])
    network.add_sources(t_flat, targets["t"])

    network.train(random_starting_weights=True)

    print(
        f'\n>>Perceptron results:\n\n'
        f'Starting weights:'
        f'\n{np.array2string(np.array(network.starting_weights), precision=5, max_line_width=10*GRID_WIDTH)}\n'
        f'Starting bias: {network.starting_bias:.5f}\n\n'
        f'Final weights:'
        f'\n{np.array2string(np.array(network.weights), precision=5, max_line_width=10*GRID_WIDTH)}\n'
        f'Final bias: {network.bias:.5f}\n\n'
        f'Total cycles: {network.cycles}\n'
    )

    x_output = network.output(x_flat)
    t_output = network.output(t_flat)

    x_activation = "Activated" if x_output == ACTIVATED else "Not activated"
    t_activation = "Activated" if t_output == ACTIVATED else "Not activated"

    print(
        f'Output for letter X: {x_activation}\n'
        f'Output for letter T: {t_activation}\n'
    )


if __name__ == '__main__':
    main()
