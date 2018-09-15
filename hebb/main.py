if __name__ == "__main__":
    from mainwindow import MainWindow
    from PyQt5 import QtWidgets
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
