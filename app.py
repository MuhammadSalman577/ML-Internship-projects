
import sys
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit,
    QPushButton, QVBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from dataPreprocessing import SpamClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class SpamClassifierUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spam Email Classifier")
        self.showMaximized()  

        self.classifier = SpamClassifier("spam_dataset.csv")
        self.X_test = None
        self.y_test = None
        self.y_pred = None

        self.init_ui()
        self.train_model()

    def init_ui(self):
        self.title = QLabel("📧 Spam Email Detection")
        self.title.setAlignment(Qt.AlignCenter)

        self.info = QLabel("Training model...")
        self.info.setAlignment(Qt.AlignCenter)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter email text here...")

        self.predict_button = QPushButton("Classify Email")
        self.predict_button.clicked.connect(self.classify_email)
        self.predict_button.setFixedSize(200, 50)
        self.result = QLabel("")
        self.result.setAlignment(Qt.AlignCenter)

        self.performance_button = QPushButton("Show Model Performance")
        self.performance_button.clicked.connect(self.show_performance)
        self.performance_button.setFixedSize(200, 50) 


        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.info)
        layout.addWidget(self.text_input)
        layout.addWidget(self.predict_button, alignment=Qt.AlignHCenter)
        layout.addWidget(self.result)
        layout.addWidget(self.performance_button, alignment=Qt.AlignHCenter)
        layout.addStretch()

        self.setLayout(layout)
        self.apply_styles()

    def train_model(self):
        try:
            accuracy = self.classifier.train()
            self.info.setText(f"Model Accuracy: {accuracy:.2f}")

            df = self.classifier.load_and_prepare_data()
            X = self.classifier.vectorizer.transform(df['clean_text'])
            y = df['label_num']
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.y_pred = self.classifier.model.predict(self.X_test)

        except Exception as e:
            self.info.setText("Training failed")
            self.result.setText(str(e))

    def classify_email(self):
        email = self.text_input.toPlainText().strip()

        if not email:
            self.result.setText("⚠️ Please enter an email.")
            self.result.setStyleSheet("color: orange;")
            return

        try:
            prediction = self.classifier.predict(email)

            if prediction == 1:
                self.result.setText("🚨 Spam Email Detected")
                self.result.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.result.setText("✅ Not Spam")
                self.result.setStyleSheet("color: green; font-weight: bold;")

        except Exception as e:
            self.result.setText(f"Error: {e}")
            self.result.setStyleSheet("color: red;")

    def show_performance(self):
        if self.y_test is None or self.y_pred is None:
            self.result.setText("No performance data available")
            return

        cm = confusion_matrix(self.y_test, self.y_pred)
        fig = Figure(figsize=(5,4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Spam", "Spam"])
        disp.plot(ax=ax, cmap=plt.cm.Blues)

        self.layout().addWidget(canvas)
        canvas.draw()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
                font-family: Arial;
            }
            QLabel {
                font-size: 16px;
                color: black;
            }
            QTextEdit {
                font-size: 14px;
                border-radius: 8px;
                padding: 6px;
                color: black;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpamClassifierUI()
    window.show()
    sys.exit(app.exec())
