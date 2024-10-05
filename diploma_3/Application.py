from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QSpinBox, QRadioButton, QFileDialog, \
    QComboBox
from PyQt6.QtGui import QAction, QIcon, QFont
from PyQt6.QtCore import Qt
from GettingData import get_mels, dataset_amplitude_to_db, data_normalization
from PreparingData import prep_data_for_kan, prep_data_for_cnn
import numpy as np
from kan import KAN
from CNNModel import CNN
import torch


class MainWindow(QMainWindow):
    data = None
    kan = KAN(width=[16384, 10], grid=5, k=3)
    cnn = CNN()

    def __init__(self):
        super().__init__()
        self.setFixedSize(1500, 800)
        self.setWindowTitle('Дипломная работа Зинченко С.К.')
        self.setWindowIcon(QIcon(''))
        self.setObjectName("MainWindow")
        self.setStyleSheet("MainWindow{border-image:url(GUI/bg.png)}")

        self.button_data = QPushButton(QIcon(''), 'Загрузить данные', self)
        self.button_data.setGeometry(390, 97, 692, 50)
        self.button_data.clicked.connect(self.get_data)

        self.label_data = QLabel('Аудио записть не выбрана', self, alignment=Qt.AlignmentFlag.AlignCenter)
        self.label_data.setGeometry(390, 148, 692, 37)

        self.button_result = QPushButton(QIcon(''), 'Получить результат', self)
        self.button_result.setGeometry(390, 240, 692, 98)
        self.button_result.clicked.connect(self.get_result)

        self.label_res_cnn = QLabel('CNN: ', self)
        self.label_res_cnn.setGeometry(300, 605, 986, 37)

        self.label_res_kan = QLabel('KAN: ', self)
        self.label_res_kan.setGeometry(300, 644, 986, 37)

        self.kan.load_ckpt(f'KAN.pth', 'models')
        self.cnn.load_state_dict(torch.load(f'models/CNN.pth'))
        self.cnn.eval()

    def get_data(self):
        name = QFileDialog.getOpenFileName(None, "Выбор аудио файла", './test', "WAV Files (*.wav)")[0]
        self.label_data.setText(name)
        self.data = get_mels(name)
        self.data = dataset_amplitude_to_db(self.data, 477509.28)
        self.data = data_normalization(np.array(self.data), -213.57964, 0.0)

    def get_result(self):
        if self.data is not None:
            match_dict = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                          5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
            cnn_text = 'CNN: '
            cnn_res = torch.softmax(self.cnn(torch.from_numpy(np.array([self.data]))), dim=1).detach().numpy()[0]
            cnn_res = sorted(enumerate(cnn_res), key=lambda x: x[1], reverse=True)
            for i in cnn_res[:3]:
                cnn_text += match_dict[i[0]] + ' (' + str(int(i[1] * 100)) + '%), '

            kan_text = 'KAN: '
            kan_res = torch.softmax(self.kan(torch.flatten(torch.from_numpy(np.array(self.data)), start_dim=1)), dim=1).detach().numpy()[0]
            kan_res = sorted(enumerate(kan_res), key=lambda x: x[1], reverse=True)
            for i in kan_res[:3]:
                kan_text += match_dict[i[0]] + ' (' + str(int(i[1] * 100)) + '%), '

            self.label_res_cnn.setText(cnn_text)
            self.label_res_kan.setText(kan_text)


app = QApplication([])
window = MainWindow()

window.show()
app.exec()
