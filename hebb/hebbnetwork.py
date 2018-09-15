ACTIVATED = 1
NOT_ACTIVATED = -1
STANDARD_INPUT = ACTIVATED

NUMBER_OF_INPUTS = 4


class IO:
    def __init__(self, input1, input2, target):
        self.input1 = input1
        self.input2 = input2
        self.target = target


class HebbNetwork:
    def __init__(self):
        self.weight1 = 0
        self.weight2 = 0
        self.threshold = 0
        self.bias = 0

        self.target_table = [
            IO(input1=NOT_ACTIVATED, input2=NOT_ACTIVATED, target=NOT_ACTIVATED),
            IO(input1=NOT_ACTIVATED, input2=ACTIVATED, target=NOT_ACTIVATED),
            IO(input1=ACTIVATED, input2=NOT_ACTIVATED, target=NOT_ACTIVATED),
            IO(input1=ACTIVATED, input2=ACTIVATED, target=NOT_ACTIVATED)
        ]

    def set_targets(self, targets):
        for io, target in zip(self.target_table, targets):
            io.target = target

    def reset_weights(self):
        self.weight1 = 0
        self.weight2 = 0
        self.bias = 0

    def train(self):
        self.reset_weights()
        for io in self.target_table:
            self.weight1 += io.input1 * io.target
            self.weight2 += io.input2 * io.target
            self.bias += STANDARD_INPUT * io.target

    def get_weights(self):
        return self.weight1, self.weight2, self.bias

    def output(self, input1, input2):
        """ response = w1 * x1 + w2 * x2 + b """
        response = self.weight1 * input1 + self.weight2 * input2 + self.bias

        if response >= self.threshold:
            return ACTIVATED
        else:
            return NOT_ACTIVATED
