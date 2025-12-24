import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from ml.classifier import PlastiTraceClassifier
from ui.main_window_new import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set global font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    # Create classifier
    classifier = PlastiTraceClassifier("models/plastitrace.pth")
    
    # Create and show main window
    window = MainWindow(classifier)
    window.show()
    
    sys.exit(app.exec_())
