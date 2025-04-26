import sys
import torch
import cv2
import pc_video
from PySide6.QtWidgets import QMainWindow, QApplication, QApplication, QFileDialog, QMessageBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer

from main_window_ui import Ui_MainWindow


def convert2QImage(img):
   height, width, channel = img.shape
   return QImage(img, width, height, width * channel, QImage.Format_RGB888)

#https://www.bilibili.com/video/BV1KhBMYuEXN

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        #程序运行前，将页面切换到第一页
        self.stackedWidget.setCurrentIndex(0)

        #信号与槽函数
        self.pushButton_image.clicked.connect(self.display_detect_image)

        #导入图像检测的模型
        self.model = torch.hub.load("./","custom",path = "./yolov5s.pt",source = "local")

        #初始化一个摄像头,并初始化它的编号
        self.cap = cv2.VideoCapture()
        self.camera_id = 0

        #信号槽绑定
        self.pushButton_open.clicked.connect(self.open_camera)
        self.pushButton_close.clicked.connect(self.close_camera)

        #初始化摄像头采集数据所使用的定时器
        self.camera_timer = QTimer()

        #设置定时器的触发时间
        self.camera_timer.setInterval(20)

        #将定时器计数一次产生的timeout信号与对应的函数进行绑定
        self.camera_timer.timeout.connect(self.show_camera_img)

        #模式初始化
        self.mode_init()

        #模型选择信号与槽函数
        self.pushButton_model.clicked.connect(self.choose_detect_model)

        #视频检测信号与槽函数绑定
        self.pushButton_video.clicked.connect(self.video_choose)

        #初始化视频
        self.video = None

        #视频的播放也需要定时器来一张张图的显示
        self.video_timer = QTimer()

        #设置视频定时器间隔以及信号与槽函数
        self.video_timer.setInterval(30)
        self.video_timer.timeout.connect(self.video_image_display)

        #视频获取的信号与槽函数绑定
        self.pushButton_get_video.clicked.connect(self.get_bilibili_video)

    #模式选择函数
    def mode_init(self):
        #使用combobox控件显示程序的可选模式
        self.comboBox.addItem("图像检测")
        self.comboBox.addItem("摄像头实时检测")
        self.comboBox.addItem("视频检测")
        self.comboBox.addItem("网页视频获取")

        #将comboBox页面变化的信号进行槽函数绑定
        self.comboBox.currentIndexChanged.connect(self.mode_choose)

    #通过comboBox的页面变化，进行程序模式的切换
    def mode_choose(self):
        print("stackwidget index = ", self.stackedWidget.currentIndex())
        print("combox index = ", self.comboBox.currentIndex())

        #comboBox在第几页，stackwidget就在第几页
        mode_page = self.comboBox.currentIndex()
        self.stackedWidget.setCurrentIndex(mode_page)

        #如果摄像头处于开启状态下直接调用了模式切换函数，就关闭摄像头
        if self.camera_timer.isActive():
            self.close_camera()

        #如果视频显示使用的定时器未关闭就切换模式，那就关闭视频
        if self.video_timer.isActive():
            self.label_video.clear()
            self.label_video_detect.clear()
            self.video_timer.stop()

    #图像选择函数,用于获取待检测图像的路径，并在ui界面显示原始图像
    def display_detect_image(self):
        print("图像检测")
        path = QFileDialog.getOpenFileName(filter="*.jpg;*.png")   #返回所有文件路径相关的内容

        #只有选到一个存在的元素，path的返回值才不是空元组
        if path[0]:
            image_path = path[0]
            #根据图像的路径，将图像转化为QImage格式
            pic_qimage = QImage(image_path)
            #将QImage的数据转换为QPixmap
            pic_qpixmap = QPixmap.fromImage(pic_qimage)
            #将qpixmap格式的图像显示在label上
            self.label_image.setPixmap(pic_qpixmap)
            #调用图像检测函数，并传入所选择的图像
            self.detect_image(image_path, self.label_image_detect)

    #图像显示函数，显示图像检测后的结果
    def detect_image(self, img, place):
        result = self.model(img)
        #获取result检测图像的三维矩阵,直接使用render生成的是列表，需要将列表元素进行索引
        result_array = result.render()[0]
        result_qimage = convert2QImage(result_array)
        #参照图像显示的函数，将qimage转换为qpixmap，并显示在label上
        result_qpixmap = QPixmap.fromImage(result_qimage)
        place.setPixmap(result_qpixmap)

        #切分检测结果
        self.plot_detect_information(result)

    #开启摄像头
    def open_camera(self):
        print("开启摄像头")
        self.cap.open(self.camera_id,cv2.CAP_DSHOW)
        self.camera_timer.start()   #摄像头开启后，计时器开启，数据开始采集

    #关闭摄像头
    def close_camera(self):
        print("关闭摄像头")
        self.cap.release()
        self.camera_timer.stop()   #摄像头关闭，计时器停止，数据停止采集
        self.label_cap.clear()
        self.label_cap_detect.clear()

    #摄像头图像采集函数
    def show_camera_img(self):
        print("采集数据")
        ret, image_array = self.cap.read()
        if ret:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            #矩阵转qimage
            pic_qimage = convert2QImage(image_array)
            
            #qimage转qpixmap
            pic_qpixmap = QPixmap.fromImage(pic_qimage)
            self.label_cap.setPixmap(pic_qpixmap)

            #将image_array赋值给图像检测函数
            self.detect_image(image_array, self.label_cap_detect)

    #模型选择函数
    def choose_detect_model(self):
        print("模型选择")

        #使用文件操作获取所选模型文件的路径,并指定选择文件的类型为pt
        model_path = QFileDialog.getOpenFileName(filter="*.pt")

        #只有选择了模型文件，model_path这个元组的第一个成员才不为空
        if model_path[0]:
            path = model_path[0]

            #将path赋值给model成员
            self.model = torch.hub.load("./", "custom", path = path, source = "local")
    
    #视频选择函数
    def video_choose(self):
        print("视频检测")

        #获取视频路径
        path = QFileDialog.getOpenFileName(dir="./datasets", filter="*.mp4")
        if path[0]:
            video_path = path[0]
            
            #将所选视频进行传参，使视频成功被获取
            self.video = cv2.VideoCapture(video_path)
            self.video_timer.start()

    #视频图像显示以及检测
    def video_image_display(self):
        print("视频图像显示")
        ret, frame = self.video.read()

        #如果视频播放完毕，ret为假，清空显示屏
        if not ret:
            self.label_video.clear()
            self.label_video_detect.clear()
            self.video_timer.stop()
        else:
            #修改图像颜色为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #矩阵转qimage
            video_qimage = convert2QImage(frame)

            #qimage转qpixmap
            video_qpixmap = QPixmap.fromImage(video_qimage)
            self.label_video.setPixmap(video_qpixmap)

            #检测视频图像
            self.detect_image(frame, self.label_video_detect)

    #获取网页视频
    def get_bilibili_video(self):
        url = self.lineEdit_url.text()
        video_info = pc_video.get_video_info(url)
        pc_video.save_video(video_info)

        #创建一个弹窗对象
        box = QMessageBox()
        box.setWindowTitle("消息")
        box.setText("视频已保存")
        box.exec()
        self.lineEdit_url.clear()

    #获取检测结果的文本信息并进行切分
    def plot_detect_information(self, result):
        #获取检测结果的表格形式
        detect_sum = result.pandas().xywh[0]

        #列索引表格中names的这一列
        obj_sum = detect_sum['name']

        #进一步提取列信息，对检测目标进行计数归类
        obj_value = obj_sum.value_counts()

        #提取准确的检测目标信息以及其数量
        size = len(obj_value)
        for i in range(size):
            obj = obj_value.index[i]
            obj_num = obj_value[i]
            data = obj + "num is" + str(obj_num)
            print(data)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    
    app.exec()