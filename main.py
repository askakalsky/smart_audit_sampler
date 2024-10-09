import sys
from PySide6 import QtWidgets
from app.sampling_app import SamplingApp


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SamplingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
