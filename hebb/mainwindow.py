from PyQt5 import QtWidgets
import functools

from hebbwindow import HebbWindow
from hebbnetwork import HebbNetwork, ACTIVATED, NOT_ACTIVATED


class MainWindow(HebbWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.network = HebbNetwork()

        # Define line_edits and respective associated push_buttons
        self.line_buttons_targets = {
            self.line_target1: self.btn_toggle_target1,
            self.line_target2: self.btn_toggle_target2,
            self.line_target3: self.btn_toggle_target3,
            self.line_target4: self.btn_toggle_target4
        }

        self.line_buttons_inputs = {
            self.line_input1: self.btn_toggle_input1,
            self.line_input2: self.btn_toggle_input2
        }

        self.all_line_buttons = {**self.line_buttons_targets, **self.line_buttons_inputs}

        self.line_targets = [
            self.line_target1,
            self.line_target2,
            self.line_target3,
            self.line_target4
        ]

        # All lines with buttons associated should have a connection to toggle_value()
        for line, button in self.all_line_buttons.items():
            button.clicked.connect(functools.partial(self.toggle_value, line, button))

        # Target line buttons (training) should connect train_network()
        for button in self.line_buttons_targets.values():
            button.clicked.connect(self.train_network)

        # Input line buttons (testing) should connect update_output()
        for button in self.line_buttons_inputs.values():
            button.clicked.connect(functools.partial(self.update_output))

        self.line_output.textChanged.connect(lambda: self.handle_line_color(self.line_output))

    def toggle_value(self, line, button):
        current_value = int(line.text())
        new_value = self.toggle_activation(current_value)

        line.setText(str(new_value))
        self.handle_line_color(line)
        self.handle_button_symbol(button, new_value)

    @staticmethod
    def toggle_activation(value):
        if value is ACTIVATED:
            return NOT_ACTIVATED
        elif value is NOT_ACTIVATED:
            return ACTIVATED

    @staticmethod
    def handle_line_color(line):
        value = int(line.text())
        if value == ACTIVATED:
            color = 'green'
        elif value == NOT_ACTIVATED:
            color = 'red'
        line.setStyleSheet(f'QLineEdit{{background: {color}}}')

    @staticmethod
    def handle_button_symbol(button, value):
        if value == ACTIVATED:
            symbol = '-'
        elif value == NOT_ACTIVATED:
            symbol = '+'
        button.setText(symbol)

    def train_network(self):
        targets = [int(line.text()) for line in self.line_targets]

        self.network.set_targets(targets)
        self.network.train()

        self.handle_weight_change()
        self.update_output()

    def handle_weight_change(self):
        weight1, weight2, bias = self.network.get_weights()

        self.line_weight1.setText(str(weight1))
        self.line_weight2.setText(str(weight2))
        self.line_bias.setText(str(bias))

    def update_output(self):
        input1, input2 = int(self.line_input1.text()), int(self.line_input2.text())
        output = self.network.output(input1, input2)

        self.line_output.setText(str(output))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
