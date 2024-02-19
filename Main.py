from PIL import Image
from pathlib import Path
from PyQt5.QtWidgets import *
from torchvision import datasets
from PyQt5.QtCore import QSize, Qt
from torch.utils.data import DataLoader
from PyQt5 import QtCore, QtGui, QtWidgets
from facenet_pytorch import MTCNN, InceptionResnetV1
from PyQt5.QtGui import QBrush, QFont, QPixmap, QImage, QPalette, QIcon
from IPython.utils.path import ensure_dir_exists
import sys, os, torch, cv2, numpy, face_recognition, traceback, shutil

class CustomToggleButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = 0
        self.setCheckable(True)
        self.setStyleSheet("border: none;")
        self.setIconSize(QSize(65,  65))

        self.iconOn = QIcon('./img/ON.png')
        self.iconOff = QIcon('./img/OFF.png')

        self.setIcon(self.iconOff)
        self.clicked.connect(self.toggle_state)

    def toggle_state(self):
        self.setIcon(self.iconOn if self.isChecked() else self.iconOff)
        self.state = 1 if self.isChecked() else 0

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super(Window, self). __init__()
        self.inputDirectory = ""
        self.inputTargetFaceDirectory = ""
        self.outputDirectory = ""
        self.ui = Ui_MainWindow()

        self.modelMoveToCache()
        self.ui.setupUi(self)
        self.ui.Filter.clicked.connect(self.filtre)

        self.buton1 = QPushButton('FOTOĞRAFLAR', self)
        self.buton1.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 20px;
                background-color: lightblue;
                background-color: rgba(62, 66, 148, 220);
                font-weight: bold;
            }

            QPushButton::pressed {
                background-color: rgba(62, 66, 148, 100);
            }
        """)
        self.buton1.setFont(QFont('Arial', 16))
        self.buton1.clicked.connect(self.pushInputFolder)
        self.buton1.setGeometry(QtCore.QRect(150, 180, 200, 75))
        layout1 = QVBoxLayout()
        layout1.addWidget(self.buton1)
        self.setLayout(layout1)

        self.buton1_5 = QPushButton('Hedef Yüzler', self)
        self.buton1_5.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 25px;
                background-color: lightblue;
                background-color: rgba(62, 66, 148, 220);
                font-weight: bold;
            }

            QPushButton::pressed {
                background-color: rgba(62, 66, 148, 100);
            }
        """)
        self.buton1_5.setFont(QFont('Arial', 15))
        self.buton1_5.clicked.connect(self.pushTargetFaceFolder)
        self.buton1_5.setGeometry(QtCore.QRect(180, 295, 130, 50))
        layout2 = QVBoxLayout()
        layout2.addWidget(self.buton1_5)
        self.setLayout(layout2)

        self.buton2 = QPushButton('HEDEF KLASÖR', self)
        self.buton2.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 20px;
                background-color: rgba(200, 50, 50, 220);
                font-weight: bold;
            }

            QPushButton::pressed {
                background-color: rgba(200, 50, 50, 100);
            }
        """)
        self.buton2.clicked.connect(self.pushOutputFolder)
        self.buton2.setFont(QFont('Arial', 16))
        self.buton2.setGeometry(QtCore.QRect(1325, 200, 200, 75))
        layout3 = QVBoxLayout()
        layout3.addWidget(self.buton2)
        self.setLayout(layout3)

        self.ui.addPrompt("1) Tüm fotoğrafların olduğu klasörü seçiniz.\n")
        self.ui.addPrompt("2) İstediğiniz filtrelemeleri seçebilirsiniz.\n")
        self.ui.addPrompt("3) Sınıflandırmanın çıkarılacağı klasörü seçiniz.\n\n")
        self.ui.addPrompt("Fotoğraflar ölçeklendirilir ve yeniden formatlanır.\n")
        self.ui.addPrompt("Kişiler tüm fotoğraflarda tespit edilir.\n")
        self.ui.addPrompt("Tespit edilen kişiler birbiriyle karşılaştırılır.\n")
        self.ui.addPrompt("Yüz hatlarının matematiksel modeli oluşturulur.\n")
        self.ui.addPrompt("Yüz hatları benzer olanlar tespit edilir, klasörlenir.\n")

        self.setWindowTitle('Folder Selector')
        oImage = QImage("./img/image.png")
        sImage = oImage.scaled(QSize(1920, 1080))
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)
        self.show()

    def modelMoveToCache(self):
        path = "~\\.cache\\torch\\checkpoints\\"
        expanded_folder_path = os.path.expanduser(path)
        os.makedirs(expanded_folder_path, exist_ok=True)
        isHere = Path(expanded_folder_path + "20180402-114759-vggface2.pt")

        if not isHere.exists():
            isHere2 = Path("./20180402-114759-vggface2.pt")
            if isHere2.exists():
                os.system(f"cp ./20180402-114759-vggface2.pt {expanded_folder_path}")
            else:
                self.showErrorPopup("Model bulunamadi, Lütfen modeli çalıştırılabilir uzantıya indiriniz!\n Model Adı: 20180402-114759-vggface2.pt")

    def showPopup(self, msg):
        result = QMessageBox.question(self, 'UYARI', msg,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        return 1 if result == 16384 else 0

    def showErrorPopup(self, msg):
        QMessageBox.warning(self, 'UYARI', msg)
        sys.exit("Program basarıyla tamamland.")

    def pushInputFolder(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if directory:
            self.inputDirectory = directory.replace('/', '\\')
            self.ui.addImages(directory)

    def pushTargetFaceFolder(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Target Face Directory')
        if directory:
            self.inputTargetFaceDirectory = directory.replace('/', '\\')
            self.ui.addImages(directory)

    def pushOutputFolder(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if directory:
            self.outputDirectory = directory.replace('/', '\\')

    def filtre(self):
        if self.ui.button.state:
            dosya_yolu = Path(self.inputTargetFaceDirectory.replace('\\', '/'))
            if self.inputTargetFaceDirectory != "" and dosya_yolu.is_dir():
                resizedImages2(self.inputTargetFaceDirectory, 960)
                self.inputTargetFaceDirectory = f"{self.inputTargetFaceDirectory}\_TARGET_RESIZED_{str(960)}"
                startFaceProcessor2(self.inputTargetFaceDirectory, self)
            else:
                QMessageBox.warning(self, 'UYARI', "Lütfen belirttiğiniz kişilerin olduğu klasörü doğru seçiniz!")

        resizedImages(self.inputDirectory, 960)
        if "" in [self.inputDirectory, self.outputDirectory]:
            print("Lütfen girdi ve cikti dosyasini giriniz!")
            return

        self.inputDirectory = f"{self.inputDirectory}\_RESIZED_{str(960)}"

        items = self.ui.gridLayoutWidget.findChildren(QtWidgets.QRadioButton)
        faceDistance = 0.80
        for i in items:
            if i.isChecked():
                faceDistance = float(str(i.text()).strip(' <'))

        items2 = self.ui.gridLayoutWidget_2.findChildren(QtWidgets.QRadioButton)
        personCount = -1
        for j in items2:
            if j.isChecked():
                personCount = int(str(j.text()).strip('+'))

        items3 = self.ui.gridLayoutWidget_3.findChildren(QtWidgets.QRadioButton)
        self.category = "Hepsi"
        for k in items3:
            if k.isChecked():
                self.category = str(k.text()).split(' ')[0]

        startFaceProcessor(self.inputDirectory, self.category, self)
        if self.ui.button.state:
            dosya_yolu = Path(self.inputTargetFaceDirectory.replace('\\', '/'))
            if self.inputTargetFaceDirectory != "" and dosya_yolu.is_dir():
                compareStart2(self.inputTargetFaceDirectory, self.inputDirectory, self.outputDirectory, personCount, faceDistance, self)
            else:
                QMessageBox.warning(self, 'UYARI', "Lütfen belirttiğiniz kişilerin olduğu klasörü doğru seçiniz!")
        else:
            compareStart(self.inputDirectory, self.outputDirectory, personCount, faceDistance, self)
        try:
            self.ui.finishImages(self.outputDirectory + "\\1")
        except:
            print("Klasör bulunamadi!")

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1769, 998)
        
        font = QtGui.QFont()
        font.setPointSize(9)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.button = CustomToggleButton("", self.centralwidget)
        self.button.move(205,  335)
        self.centralwidget.setObjectName("centralwidget")
        self.Filter = QtWidgets.QPushButton(self.centralwidget)
        self.Filter.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 20px;
                background-color: rgba(25, 80, 25, 150);
                font-weight: bold;
            }

            QPushButton::pressed {
                background-color: rgba(25, 80, 25, 100);
            }
        """)
        self.Filter.setGeometry(QtCore.QRect(615, 440, 420, 75))
        self.Filter.setCheckable(False)
        self.Filter.setAutoRepeat(False)
        self.Filter.setAutoExclusive(False)
        self.Filter.setObjectName("Filter")

        self.scrollArea = QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(50, 540, 700, 440))
        self.scrollArea.setObjectName("scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setStyleSheet("""
            QScrollArea {
                background-color: rgba(255, 19, 2, 150);
            }
        """)
        
        self.areaText = ""
        self.scrollArea2 = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea2.setGeometry(QtCore.QRect(875, 540, 700, 440))
        self.scrollArea2.setObjectName("scrollArea2")
        self.scrollArea2.setWidgetResizable(True)
        
        self.scrollArea2.setStyleSheet("""
            QScrollArea {
                background-color: rgba(81, 95, 237, 150);
            }
        """)

        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 500, 370))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")

        self.threadtable = QtWidgets.QTableWidget(self.scrollAreaWidgetContents_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_2)
        self.horizontalLayout.addWidget(self.threadtable)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.threadtable.setObjectName("threadtable")
        self.threadtable.setColumnCount(0)
        self.threadtable.setRowCount(0)

        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 500, 370))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.threadtable3 = QtWidgets.QTableWidget(self.scrollAreaWidgetContents_3)
        self.horizontalLayout3 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_3)
        self.horizontalLayout3.addWidget(self.threadtable3)
        self.horizontalLayout3.setObjectName("horizontalLayout3")
        self.threadtable3.setObjectName("threadtable3")
        self.threadtable3.setColumnCount(0)
        self.threadtable3.setRowCount(0)
        
        self.groupRows = QtWidgets.QGroupBox(self.centralwidget)
        self.groupRows.setGeometry(QtCore.QRect(720, 40, 200, 390))
        self.groupRows.setObjectName("groupRows")
        self.groupRows.setStyleSheet("""
            QGroupBox {
                border: none;
                border-radius: 20px;
                background-image: url('./img/blue.jpg');
                background-repeat: no-repeat;
                background-position: center;
                font-weight: bold;
            }

            QGroupBox::title {
                background: transparent;
                border: none;
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)

        self.groupRows2 = QtWidgets.QGroupBox(self.centralwidget)        
        self.groupRows2.setGeometry(QtCore.QRect(470, 40, 200, 390))
        self.groupRows2.setObjectName("groupRows2")
        self.groupRows2.setStyleSheet("""
            QGroupBox {
                border: none;
                border-radius: 20px;
                background-image: url('./img/orange.jpg');
                background-repeat: no-repeat;
                background-position: center;
                font-weight: bold;
            }

            QGroupBox::title {
                background: transparent;
                border: none;
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)

        self.groupRows3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupRows3.setGeometry(QtCore.QRect(970, 40, 200, 390))
        self.groupRows3.setObjectName("groupRows3")
        self.groupRows3.setStyleSheet("""
            QGroupBox {
                border: none;
                border-radius: 20px;
                background-image: url('./img/green.jpg');
                background-repeat: no-repeat;
                background-position: center;
                font-weight: bold;
            }

            QGroupBox::title {
                background: transparent;
                border: none;
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)

        self.gridLayoutWidget = QtWidgets.QWidget(self.groupRows)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(40, 60, 160, 330))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupRows2)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(50, 60, 160, 330))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")

        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupRows3)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(40, 60, 160, 330))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")

        self.radioButton = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_5 = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.radioButton_5.setObjectName("radioButton_5")
        self.radioButton_6 = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.radioButton_6.setObjectName("radioButton_6")
        self.radioButton_7 = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.radioButton_7.setObjectName("radioButton_7")

        self.radioButton_8 = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioButton_8.setObjectName("radioButton_8")
        self.radioButton_9 = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioButton_9.setObjectName("radioButton_9")
        self.radioButton_10 = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioButton_10.setObjectName("radioButton_10")
        self.radioButton_11 = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioButton_11.setObjectName("radioButton_11")      
        self.radioButton_12 = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioButton_12.setObjectName("radioButton_12")
        self.radioButton_13 = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioButton_13.setObjectName("radioButton_13")
        self.radioButton_14 = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioButton_14.setObjectName("radioButton_14")

        self.radioButton_15 = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.radioButton_15.setObjectName("radioButton_15")
        self.radioButton_16 = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.radioButton_16.setObjectName("radioButton_16")
        self.radioButton_17 = QtWidgets.QRadioButton(self.gridLayoutWidget_3)
        self.radioButton_17.setObjectName("radioButton_17")

        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.addWidget(self.radioButton, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.radioButton_2, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.radioButton_3, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.radioButton_4, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.radioButton_5, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.radioButton_6, 5, 0, 1, 1)
        self.gridLayout.addWidget(self.radioButton_7, 6, 0, 1, 1)

        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.addWidget(self.radioButton_8, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.radioButton_9, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.radioButton_10, 2, 0, 1, 1)
        self.gridLayout_2.addWidget(self.radioButton_11, 3, 0, 1, 1)
        self.gridLayout_2.addWidget(self.radioButton_12, 4, 0, 1, 1)
        self.gridLayout_2.addWidget(self.radioButton_13, 5, 0, 1, 1)
        self.gridLayout_2.addWidget(self.radioButton_14, 6, 0, 1, 1)

        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.addWidget(self.radioButton_15, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.radioButton_16, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.radioButton_17, 2, 0, 1, 1)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        
        self.retranslateUi(MainWindow)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setStatusBar(self.statusbar)

        self.label = QLabel(self.scrollArea2)
        self.label.setText(self.areaText)
        self.label.setStyleSheet("background-color: rgba(255, 255, 255, 40);")
        
        font = QFont()
        font.setPointSize(20)
        self.label.setFont(font)

        self.label.setAlignment(Qt.AlignCenter) 

    def addPrompt(self, msg):
        self.areaText += msg
        self.label.setText(self.areaText)
        self.scrollArea2.setWidget(self.label)

    def addImages(self, paths):
        imgPath = paths
        filenames = [i for i in os.listdir(imgPath) if '.' in i]
        labels = []
        for filename in filenames[:5]:
            if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                imageLabel = QLabel()
                image = QImage(os.path.join(imgPath, filename)).scaled(500, 500, aspectRatioMode=Qt.KeepAspectRatio)
                pixmap = QPixmap.fromImage(image)
                imageLabel.setPixmap(pixmap)
                
                hbox = QHBoxLayout()
                hbox.addStretch(1)
                hbox.addWidget(imageLabel)
                hbox.addStretch(1)
                
                labels.append(hbox)

        layout = QVBoxLayout()
        for label in labels:
            layout.addLayout(label)


        widget = QWidget()
        widget.setLayout(layout)
        self.scrollArea.setWidget(widget)

    def finishImages(self, paths):
        imgPath = paths
        filenames = os.listdir(imgPath)
        labels = []
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                imageLabel = QLabel()
                image = QImage(os.path.join(imgPath, filename)).scaled(500, 500, aspectRatioMode=Qt.KeepAspectRatio)
                pixmap = QPixmap.fromImage(image)
                imageLabel.setPixmap(pixmap)

                hbox = QHBoxLayout()
                hbox.addStretch(1)
                hbox.addWidget(imageLabel)
                hbox.addStretch(1)

                labels.append(hbox)

        layout = QVBoxLayout()
        for label in labels:
            layout.addLayout(label)


        widget = QWidget()
        widget.setLayout(layout)
        self.scrollArea2.setWidget(widget)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Filter.setText(_translate("MainWindow", "ÇALIŞTIR"))
        self.Filter.setFont(QFont('Arial', 16))
        
        headerFont = QFont('Arial', 10)
        headerFont.setBold(True)
        
        self.groupRows2.setTitle(_translate("MainWindow", "FİLTRELEME SAYISI(100+)"))
        self.groupRows2.setFont(headerFont)
        self.radioButton_8.setText(_translate("MainWindow", "1"))
        self.radioButton_9.setText(_translate("MainWindow", "3"))
        self.radioButton_10.setText(_translate("MainWindow", "5"))
        self.radioButton_11.setText(_translate("MainWindow", "10"))
        self.radioButton_12.setText(_translate("MainWindow", "25"))
        self.radioButton_13.setText(_translate("MainWindow", "50"))
        self.radioButton_14.setText(_translate("MainWindow", "100+"))

        self.groupRows.setTitle(_translate("MainWindow", "BENZERLİK MESAFESİ (0.90)"))
        self.groupRows.setFont(headerFont)
        self.radioButton.setText(_translate("MainWindow", "0.50 <"))
        self.radioButton_2.setText(_translate("MainWindow", "0.60 <"))
        self.radioButton_3.setText(_translate("MainWindow", "0.70 <"))
        self.radioButton_4.setText(_translate("MainWindow", "0.80 <"))
        self.radioButton_5.setText(_translate("MainWindow", "0.90 <"))
        self.radioButton_6.setText(_translate("MainWindow", "1.00 <"))
        self.radioButton_7.setText(_translate("MainWindow", "1.10 <"))

        self.groupRows3.setTitle(_translate("MainWindow", "İSTENEN TÜR (Herkes)"))
        self.groupRows3.setFont(headerFont)
        self.radioButton_15.setText(_translate("MainWindow", "Bireysel Fotoğraflar"))
        self.radioButton_16.setText(_translate("MainWindow", "Toplu Fotoğraflar"))
        self.radioButton_17.setText(_translate("MainWindow", "Hepsi"))

class faceCompareProcessor2:
    def __init__(self, inputTargetFaceDirectory, inputPath, outputPath, personCount, faceDistance, MainUi):
        self.MainUi = MainUi
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.personCount = personCount
        self.faceDistance = faceDistance
        self.inputTargetFaceDirectory = inputTargetFaceDirectory

        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self.device}')
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                           device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    def load_dataset(self, imagePath, folder):
        faceDataset = datasets.ImageFolder("\\".join(imagePath.split('\\')[:-1]) + f"\\{folder}")
        faceDataset.idx_to_class = {i: c for c, i in faceDataset.class_to_idx.items()}
        loader = DataLoader(faceDataset, collate_fn=self.collate_fn, num_workers=self.workers)
        return loader, faceDataset
    def collate_fn(self, x):
        return x[0]
    def process_faces(self, loader, faceDataset, faceDetectionRate=0.9999):
        names = []
        aligned = []

        for x, y in loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                print(f'Face detected with probability {faceDataset.idx_to_class[y]}: {prob:8f}')
                aligned.append(x_aligned)
                names.append(faceDataset.idx_to_class[y])

        aligned = torch.stack(aligned).to(self.device)
        embeddings = self.resnet(aligned).detach().cpu()
        return embeddings, names
    def arrayControl(self, new_folder, arr):
        for i in new_folder:
            if i in arr:
                return 1
        return 0
    def calculate_distances(self, embeddings1, names1, embeddings2, names2):
        human_images = []
        return_images = []

        for i, e2 in enumerate(embeddings2):
            new_folder = []
            for j, e1 in enumerate(embeddings1):
                if (e2 - e1).norm().item() < self.faceDistance:
                    new_folder.append(names1[j])
            if len(new_folder) > 0:
                human_images.append(new_folder)

        muteIdx = []
        acceptIdx = []
        for idx, i in enumerate(human_images):
            flag = 1
            for j in i:
                if idx + 1 != len(human_images):
                    for idx2, k in enumerate(human_images[idx + 1:]):
                        if j in k:
                            if len(k) >= len(i):
                                flag = 0
                            else:
                                muteIdx.append(idx + idx2 + 1)
            if flag == 1:
                acceptIdx.append(idx)
        
        for idx, element in enumerate(human_images):
            if (idx in acceptIdx) and (idx not in muteIdx):
                return_images.append(element)
                # print(element)

        sorted_array = sorted(return_images, key=len, reverse=True)
        if self.personCount != -1: return sorted_array[:self.personCount]
        else: return sorted_array
    def organize_folders(self, human_images):
        for i in range(len(human_images)):
            folder_name = f'{self.outputPath}\\{i + 1}'
            isHere = Path(folder_name)
            if isHere.exists():
                os.system(f"rm -rf {folder_name}")
            
            ensure_dir_exists(folder_name)
            print(human_images)
            beforeImgName = human_images[i][-1].split('_')[1]
            selectPath = "\\".join(self.inputPath.split('\\')[:-1])

            shutil.copy((selectPath + '\\allFaces\\' + human_images[i][-1] + '\\' + beforeImgName + '.jpg').replace('\\', '/'), folder_name.replace('\\', '/'))
            shutil.move((folder_name + '/' + human_images[i][-1].split('_')[1] + '.jpg').replace('\\', '/'), (folder_name + '/' + 'Crop.jpg').replace('\\', '/'))

            for j in human_images[i]:
                fileName = f'{selectPath}/' + j.split('_')[1] + '.jpg'
                shutil.copy(fileName.replace('\\', '/'), folder_name.replace('\\', '/'))
        if Window.showPopup(self.MainUi, "Fotoğraflar başarılı bir şekilde klasörlendi, Uygulama kapatılsın mı?") == 1:
            Window.showErrorPopup(self.MainUi, "Kapatmak için tıklayınız.")
def compareStart2(inputTargetFaceDirectory, inputPath, outputPath, personCount, faceDistance, MainUi):
    face_processor = faceCompareProcessor2(inputTargetFaceDirectory, inputPath, outputPath, personCount, faceDistance, MainUi)

    data_loader1, faceDataset1 = face_processor.load_dataset(face_processor.inputPath, "allFaces")
    embeddings1, names1 = face_processor.process_faces(data_loader1, faceDataset1)
    
    data_loader2, faceDataset2 = face_processor.load_dataset(face_processor.inputTargetFaceDirectory, "targetAllFaces")
    embeddings2, names2 = face_processor.process_faces(data_loader2, faceDataset2)

    human_images = face_processor.calculate_distances(embeddings1, names1, embeddings2, names2)
    face_processor.organize_folders(human_images)

class faceCompareProcessor:
    def __init__(self, inputPath, outputPath, personCount, faceDistance, MainUi):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.personCount = personCount
        self.faceDistance = faceDistance
        self.MainUi = MainUi

        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self.device}')
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                           device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    def load_dataset(self):
        faceDataset = datasets.ImageFolder("\\".join(self.inputPath.split('\\')[:-1]) + "\\allFaces")
        faceDataset.idx_to_class = {i: c for c, i in faceDataset.class_to_idx.items()}
        loader = DataLoader(faceDataset, collate_fn=self.collate_fn, num_workers=self.workers)
        return loader, faceDataset
    def collate_fn(self, x):
        return x[0]
    def process_faces(self, loader, faceDataset, faceDetectionRate=0.9999):
        names = []
        aligned = []

        for x, y in loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                print(f'Face detected with probability {faceDataset.idx_to_class[y]}: {prob:8f}')
                aligned.append(x_aligned)
                names.append(faceDataset.idx_to_class[y])

        aligned = torch.stack(aligned).to(self.device)
        embeddings = self.resnet(aligned).detach().cpu()
        return embeddings, names
    def arrayControl(self, new_folder, arr):
        for i in new_folder:
            if i in arr:
                return 1
        return 0
    def calculate_distances(self, embeddings, names):
        human_images = []
        return_images = []

        for i, e1 in enumerate(embeddings):
            new_folder = []
            for j, e2 in enumerate(embeddings):
                if (e1 - e2).norm().item() < self.faceDistance:
                    new_folder.append(names[j])

            human_images.append(new_folder)


        muteIdx = []
        acceptIdx = []
        for idx, i in enumerate(human_images):
            flag = 1
            for j in i:
                if idx + 1 != len(human_images):
                    for idx2, k in enumerate(human_images[idx + 1:]):
                        if j in k:
                            if len(k) >= len(i):
                                flag = 0
                            else:
                                muteIdx.append(idx + idx2 + 1)
            if flag == 1:
                acceptIdx.append(idx)
        
        for idx, element in enumerate(human_images):
            if (idx in acceptIdx) and (idx not in muteIdx):
                return_images.append(element)
                print(element)



        sorted_array = sorted(return_images, key=len, reverse=True)
        if self.personCount != -1: return sorted_array[:self.personCount]
        else: return sorted_array
    def organize_folders(self, human_images):
        for i in range(len(human_images)):
            folder_name = f'{self.outputPath}\\{i + 1}'
            isHere = Path(folder_name)
            if isHere.exists():
                os.system(f"rm -rf {folder_name}")
            
            ensure_dir_exists(folder_name)
            # os.system(f'mkdir {folder_name}')
            beforeImgName = human_images[i][-1].split('_')[1]
            selectPath = "\\".join(self.inputPath.split('\\')[:-1])

            shutil.copy((selectPath + '\\allFaces\\' + human_images[i][-1] + '\\' + beforeImgName + '.jpg').replace('\\', '/'), folder_name.replace('\\', '/'))
            shutil.move((folder_name + '/' + human_images[i][-1].split('_')[1] + '.jpg').replace('\\', '/'), (folder_name + '/' + 'Crop.jpg').replace('\\', '/'))

            for j in human_images[i]:
                fileName = f'{selectPath}/' + j.split('_')[1] + '.jpg'
                shutil.copy(fileName.replace('\\', '/'), folder_name.replace('\\', '/'))
        if Window.showPopup(self.MainUi, "Fotoğraflar başarılı bir şekilde klasörlendi, Uygulama kapatılsın mı?") == 1:
            Window.showErrorPopup(self.MainUi, "Kapatmak için tıklayınız.")
def compareStart(inputPath, outputPath, personCount, faceDistance, MainUi):
    face_processor = faceCompareProcessor(inputPath, outputPath, personCount, faceDistance, MainUi)
    data_loader, faceDataset = face_processor.load_dataset()
    embeddings, names = face_processor.process_faces(data_loader, faceDataset)
    human_images = face_processor.calculate_distances(embeddings, names)
    face_processor.organize_folders(human_images)

def preprocess_images(input_folder, output_folder, size, target_size=(960, 960)):
    target_size = (size, size)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)

        new_image_file = os.path.splitext(image_file)[0] + '.jpg'
        output_path = os.path.join(output_folder, new_image_file)

        img = cv2.imread(input_path)
        height, width, _ = img.shape
        max_size = max(height, width)

        square_img = numpy.zeros((max_size, max_size, 3), dtype=numpy.uint8)
        square_img[:height, :width, :] = img

        resized_img = cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)

class multiFaceProcessor:
    def __init__(self, inputPath, category, MainUi, inputImages="allFaces"):
        folder = inputPath.split("\\")[-1]
        inputPath = "\\".join(inputPath.split("\\")[:-1])
        self.category = category
        self.MainUi = MainUi

        self.faces_folder_path = os.path.join(inputPath, inputImages)

        folder_path = Path(self.faces_folder_path)
        if folder_path.exists():
            if Window.showPopup(self.MainUi, "Eski allFaces verilerini silmek ister misiniz?") == 1:
                patth = (self.faces_folder_path).replace('\\', '/')
                os.system(f"rm -rf {patth}")
            else:
                Window.showErrorPopup(self.MainUi, "Eski allFaces dosyanizi yedekleyiniz!")
        
        self.inputPath = os.path.join(inputPath, folder)

        try:
            ensure_dir_exists(self.faces_folder_path)
        except:
            Window.showErrorPopup(self.MainUi, "Klasör oluşturulamadi!")

    def process_image(self, img_name):
        image_path = os.path.join(self.inputPath, img_name)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if self.category == "Bireysel" and len(face_locations) != 1: return
        elif self.category == "Toplu" and len(face_locations) <= 1: return

        print(f"\nI found {len(face_locations)} face(s) in {img_name}.")

        for idx, face_location in enumerate(face_locations, start=1):
            top, right, bottom, left = face_location

            face_path = os.path.join(self.faces_folder_path, f"{idx}_{img_name.split('.')[0]}")
            
            ensure_dir_exists(face_path)

            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save(os.path.join(face_path, img_name))

            print("Image saved ->", face_path)
def startFaceProcessor(ImgsPath, category, MainUi):
    face_processor = multiFaceProcessor(ImgsPath, category, MainUi)
    images_name = os.listdir(face_processor.inputPath)
    for img_name in images_name:
        face_processor.process_image(img_name)
def resizedImages(path, size):
    output_folder_path = path + "/_RESIZED_" + str(size)
    preprocess_images(path, output_folder_path, size)

class multiFaceProcessor2:
    def __init__(self, inputPath, MainUi, inputImages="targetAllFaces"):
        folder = inputPath.split("\\")[-1]
        self.inputPath = "\\".join(inputPath.split("\\")[:-1])
        self.MainUi = MainUi

        self.faces_folder_path = os.path.join(self.inputPath, inputImages)

        folder_path = Path(self.faces_folder_path)
        if folder_path.exists():
            if Window.showPopup(self.MainUi, "Eski targetAllFaces verilerini silmek ister misiniz?") == 1:
                patth = (self.faces_folder_path).replace('\\', '/')
                os.system(f"rm -rf {patth}")
            else:
                Window.showErrorPopup(self.MainUi, "Eski targetAllFaces dosyanizi yedekleyiniz!")

        self.inputPath = os.path.join(self.inputPath, folder)

        try:
            ensure_dir_exists(self.faces_folder_path)
        except:
            Window.showErrorPopup(self.MainUi, "Klasör oluşturulamadi!")
    def process_image2(self, img_name):
        image_path = os.path.join(self.inputPath, img_name)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        print(f"\nI found {len(face_locations)} face(s) in {img_name}.")

        for idx, face_location in enumerate(face_locations, start=1):
            top, right, bottom, left = face_location

            face_path = os.path.join(self.faces_folder_path, f"{idx}_{img_name.split('.')[0]}")
            
            ensure_dir_exists(face_path)

            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save(os.path.join(face_path, img_name))

            print("Image saved ->", face_path)
def startFaceProcessor2(targetImgsPath, MainUi):
    face_processor = multiFaceProcessor2(targetImgsPath, MainUi)
    images_name = os.listdir(face_processor.inputPath)
    for img_name in images_name:
        face_processor.process_image2(img_name)
def resizedImages2(path, size):
    output_folder_path = path + "/_TARGET_RESIZED_" + str(size)
    preprocess_images(path, output_folder_path, size)

def excepthook(exc_type, exc_value, exc_traceback):
    with open("error.log", "w") as f:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

if __name__=="__main__":
    sys.excepthook = excepthook
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
