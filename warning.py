from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread

class WarningDialog(QtWidgets.QWidget):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle("WARNING")
        self.setGeometry(100, 100, 400, 200)

        self.message = message

        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel(self.message)
        layout.addWidget(label)

        close_button = QtWidgets.QPushButton("Kapat")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)

class WarningThread(QThread):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self):
        app = QtWidgets.QApplication([])
        warning_dialog = WarningDialog(self.message)
        warning_dialog.show()
        app.exec_()
