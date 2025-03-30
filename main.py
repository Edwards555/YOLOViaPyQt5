import ctypes  # 用于设置应用程序的图标
import sys  # 提供与Python解释器交互的功能
import time  # 提供时间相关的功能

import cv2  # OpenCV库，用于图像处理
import numpy as np  # 数值计算库
import qdarkstyle  # PyQt5的暗黑主题样式
from PIL import Image  # 图像处理库Pillow
from PyQt5 import QtCore, QtGui, QtWidgets  # PyQt5核心模块
from PyQt5.Qt import QThread  # PyQt5线程模块
from PyQt5.QtCore import *  # PyQt5核心功能
from PyQt5.QtGui import *  # PyQt5图形界面功能
from PyQt5.QtWidgets import *  # PyQt5小部件功能

from custom.graphicsView import GraphicsView  # 自定义的图形视图类
from custom.listWidgets import *  # 自定义的列表小部件
from custom.stackedWidget import *  # 自定义的堆叠小部件
from custom.treeView import FileSystemTreeView  # 自定义的文件系统树视图
from yolo import YOLO  # YOLO目标检测类
from Database import Database  # 数据库操作类
from Sign_Up import SignWindow  # 注册窗口类


# 设置应用程序的任务栏图标
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")

# 多线程实时检测类
class DetectThread(QThread):
    Send_signal = pyqtSignal(np.ndarray, int)  # 自定义信号，发送检测结果和警告状态

    def __init__(self, fileName):
        super(DetectThread, self).__init__()
        self.capture = cv2.VideoCapture(fileName)  # 打开视频文件或摄像头
        self.count = 0  # 检测到未佩戴口罩的计数
        self.warn = False  # 是否发送警告信号

    def run(self):
        # 循环读取视频帧并进行检测
        ret, self.frame = self.capture.read()
        while ret:
            ret, self.frame = self.capture.read()
            self.detectCall()

    def detectCall(self):
        fps = 0.0  # 初始化帧率
        t1 = time.time()  # 记录开始时间
        frame = self.frame  # 当前帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        frame = Image.fromarray(np.uint8(frame))  # 转换为Pillow图像
        frame_new, predicted_class = yolo.detect_image(frame)  # 使用YOLO进行检测
        frame = np.array(frame_new)  # 转换为NumPy数组
        if predicted_class == "face":  # 如果检测到人脸
            self.count += 1
        else:
            self.count = 0
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换回BGR格式
        fps = (fps + (1. / (time.time() - t1))) / 2  # 计算帧率
        #print("fps= %.2f" % (fps))  # 打印帧率
        frame = cv2.putText(frame, "fps= %.2f" % (
            fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 在帧上显示帧率
        if self.count > 30:  # 如果连续检测到未佩戴口罩超过30帧
            self.count = 0
            self.warn = True
        else:
            self.warn = False
        self.Send_signal.emit(frame, self.warn)  # 发送信号

# 主窗口类
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()

        self.cap = cv2.VideoCapture()  # 视频捕获对象
        self.CAM_NUM = 0  # 摄像头编号
        self.thread_status = False  # 判断识别线程是否开启
        self.tool_bar = self.addToolBar('工具栏')  # 添加工具栏
        self.action_right_rotate = QAction(
            QIcon("icons/右旋转.png"), "向右旋转90", self)  # 向右旋转按钮
        self.action_left_rotate = QAction(
            QIcon("icons/左旋转.png"), "向左旋转90°", self)  # 向左旋转按钮
        self.action_opencam = QAction(QIcon("icons/摄像头.png"), "开启摄像头", self)  # 开启摄像头按钮
        self.action_video = QAction(QIcon("icons/video.png"), "加载视频", self)  # 加载视频按钮
        self.action_image = QAction(QIcon("icons/图片.png"), "加载图片", self)  # 加载图片按钮
        self.action_right_rotate.triggered.connect(self.right_rotate)  # 绑定右旋转事件
        self.action_left_rotate.triggered.connect(self.left_rotate)  # 绑定左旋转事件
        self.action_opencam.triggered.connect(self.opencam)  # 绑定开启摄像头事件
        self.action_video.triggered.connect(self.openvideo)  # 绑定加载视频事件
        self.action_image.triggered.connect(self.openimage)  # 绑定加载图片事件
        self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate,
                                  self.action_opencam, self.action_video, self.action_image))  # 添加按钮到工具栏
        self.stackedWidget = StackedWidget(self)  # 堆叠小部件
        self.fileSystemTreeView = FileSystemTreeView(self)  # 文件系统树视图
        self.graphicsView = GraphicsView(self)  # 图形视图
        self.dock_file = QDockWidget(self)  # 文件停靠窗口
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('目录'))  # 设置标题
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)  # 禁用停靠功能

        self.dock_attr = QDockWidget(self)  # 属性停靠窗口
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('上报数据'))  # 设置标题
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)  # 禁用停靠功能

        self.setCentralWidget(self.graphicsView)  # 设置中心窗口
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)  # 添加左侧停靠窗口
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)  # 添加右侧停靠窗口

        self.setWindowTitle('穴位检测')  # 设置窗口标题
        self.setWindowIcon(QIcon('icons/mask.png'))  # 设置窗口图标
        self.src_img = None  # 原始图像
        self.cur_img = None  # 当前图像
    # 更新图像
    def update_image(self):
        if self.src_img is None:
            return
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.update_image(img)

    # 更改图像
    def change_image(self, img):
        self.src_img = img
        img = self.process_image()
        self.cur_img = img
        self.graphicsView.change_image(img)

    # 处理图像
    def process_image(self):
        img = self.src_img.copy()
        for i in range(self.useListWidget.count()):
            img = self.useListWidget.item(i)(img)
        return img

    # 向右旋转
    def right_rotate(self):
        self.graphicsView.rotate(90)

    # 向左旋转
    def left_rotate(self):
        self.graphicsView.rotate(-90)

    # 添加检测到的项目
    def add_item(self, image):
        # 总Widget
        wight = QWidget()
        # 总体横向布局
        layout_main = QHBoxLayout()
        map_l = QLabel()  # 图片显示
        map_l.setFixedSize(60, 40)
        map_l.setPixmap(image.scaled(60, 40))
        # 右边的纵向布局
        layout_right = QVBoxLayout()
        # 右下的的横向布局
        layout_right_down = QHBoxLayout()  # 右下的横向布局
        layout_right_down.addWidget(
            QLabel(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        # 按照从左到右, 从上到下布局添加
        layout_main.addWidget(map_l)  # 最左边的图片
        layout_right.addWidget(QLabel('警告！检测到未佩戴口罩'))  # 右边的纵向布局
        layout_right.addLayout(layout_right_down)  # 右下角横向布局
        layout_main.addLayout(layout_right)  # 右边的布局
        wight.setLayout(layout_main)  # 布局给wight
        item = QListWidgetItem()  # 创建QListWidgetItem对象
        item.setSizeHint(QSize(300, 80))  # 设置QListWidgetItem大小
        self.stackedWidget.addItem(item)  # 添加item
        self.stackedWidget.setItemWidget(item, wight)  # 为item设置widget

    # 打开视频
    def openvideo(self):
        print(self.thread_status)
        if self.thread_status == False:

            fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择视频", "D:/", "*.mp4;;*.flv;;All Files(*)")

            flag = self.cap.open(fileName)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请选择视频文件",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.detectThread = DetectThread(fileName)
                self.detectThread.Send_signal.connect(self.Display)
                self.detectThread.start()
                self.action_video.setText('关闭视频')
                self.thread_status = True
        elif self.thread_status == True:
            self.detectThread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            self.action_video.setText('打开视频')
            self.thread_status = False

    # 打开图片
    def openimage(self):
        if self.thread_status == False:
            fileName, filetype = QFileDialog.getOpenFileName(
                self, "选择图片", "D:/", "*.jpg;;*.png;;All Files(*)")
            if fileName != '':
                src_img = Image.open(fileName)
                r_image, predicted_class = yolo.detect_image(src_img)
                r_image = np.array(r_image)
                showImage = QtGui.QImage(
                    r_image.data, r_image.shape[1], r_image.shape[0], QtGui.QImage.Format_RGB888)
                self.graphicsView.set_image(QtGui.QPixmap.fromImage(showImage))

    # 打开摄像头
    def opencam(self):
        if self.thread_status == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"警告", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.detectThread = DetectThread(self.CAM_NUM)
                self.detectThread.Send_signal.connect(self.Display)
                self.detectThread.start()
                self.action_video.setText('关闭视频')
                self.thread_status = True
        else:
            self.detectThread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            self.action_video.setText('打开视频')
            self.thread_status = False

    # 显示检测结果
    def Display(self, frame, warn):

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(
            im.data, im.shape[1], im.shape[0], QtGui.QImage.Format_RGB888)
        self.graphicsView.set_image(QtGui.QPixmap.fromImage(showImage))

    # 关闭事件
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Warning, u"关闭", u"确定退出？")
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.thread_status == True:
                self.detectThread.terminate()
            if self.cap.isOpened():
                self.cap.release()
            event.accept()



# 主程序入口
if __name__ == "__main__":
    yolo = YOLO()  # 初始化YOLO模型
    app = QApplication(sys.argv)  # 创建应用程序
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())  # 设置暗黑主题
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))  # 设置暗黑主题
    window = MyApp()  # 创建主窗口
    window.show()  # 显示主窗口
    sys.exit(app.exec_())  # 运行应用程序
