深度学习基于YOLOv8 pyqt5的交通标志检测系统
@[toc]
以下文字及代码仅供参考。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d3def4d079b46678658d2f93acb46dd.png)

同学呀，先训练5000张交通标志数据集 

下面的类别别搞错了：
类别['绿灯', '红灯', '限速10', '限速100', '限速110', '限速120', '限速20', '限速30', '限速40', '限速50', '限速60', '限速70', '限速80', '限速90', '停止']，15类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/744438daa2ec4207bc2e76545bf18a6f.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/17e79529895c43d29e19766a674bd2a5.png)
基于 **YOLOv8** 和 **PyQt5** 的交通标志检测系统的实现。系统能够检测并分类交通标志，包含 15 类标签（如绿灯、红灯、限速标志等）。

仅供参考。同学

---

## **项目结构**

```
traffic_sign_detection/
├── dataset/               # 数据集
│   ├── images/            # 图片数据
│   └── labels/            # 标注文件
├── models/                # 模型定义
│   └── train_model.py     # 训练脚本
├── utils/                 # 工具函数
│   └── data_utils.py      # 数据处理工具
├── app.py                 # PyQt5 主程序
└── requirements.txt       # 依赖库
```

---

## **一、安装依赖**

在终端中运行以下命令安装所需库：

```bash
pip install ultralytics opencv-python pyqt5 pandas matplotlib
```

- `ultralytics`：YOLOv8 的官方库。
- `opencv-python`：图像处理。
- `pyqt5`：构建图形用户界面。
- `pandas`：数据加载与处理。
- `matplotlib`：可视化工具。

---

## **二、准备数据集**

### **1. 数据集格式**
假设你的数据集已经按照 YOLO 格式组织好：

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### **2. 创建 `data.yaml` 文件**

创建一个 `data.yaml` 文件，内容如下：

```yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 15                   # 类别数量（15类）
names: [
    '绿灯', '红灯', '限速10', '限速100', '限速110',
    '限速120', '限速20', '限速30', '限速40', '限速50',
    '限速60', '限速70', '限速80', '限速90', '停止'
]
```

---

## **三、训练 YOLOv8 模型**

### **1. 训练脚本 (`models/train_model.py`)**

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 使用 YOLOv8 Nano 模型

# 训练模型
if __name__ == "__main__":
    model.train(
        data='dataset/data.yaml',  # 数据配置文件路径
        epochs=50,                 # 训练轮数
        imgsz=640,                 # 输入图片大小
        batch=16,                  # 批量大小
        device='0' if torch.cuda.is_available() else 'cpu',  # 使用 GPU 或 CPU
        name='traffic_sign_detection'  # 实验名称
    )
```

运行以下命令开始训练：

```bash
cd models
python train_model.py
```

训练完成后，生成的模型权重文件将保存在 `runs/detect/traffic_sign_detection/weights/best.pt`。

---

## **四、构建 PyQt5 界面**

### **1. PyQt5 主程序 (`app.py`)**

```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO

class TrafficSignDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        # 加载 YOLOv8 模型
        self.model = YOLO('runs/detect/traffic_sign_detection/weights/best.pt')

    def initUI(self):
        self.setWindowTitle("交通标志检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setText("选择图片以检测交通标志")
        self.label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("选择图片", self)
        self.button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            image = cv2.imread(file_path)
            results = self.model(image)
            annotated_image = results[0].plot()  # 绘制检测结果

            # 将 OpenCV 图像转换为 PyQt 图像
            height, width, channel = annotated_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            self.label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficSignDetectionApp()
    window.show()
    sys.exit(app.exec_())
```

---

## **五、运行系统**

1. **训练模型**：
   - 确保数据集已准备好，并运行 `models/train_model.py` 进行模型训练。

2. **启动 PyQt5 应用**：
   - 在终端中运行以下命令启动 PyQt5 应用程序：

```bash
python app.py
```

---

## **六、总结**

通过以上步骤，你可以完成一个基于 **YOLOv8** 和 **PyQt5** 的交通标志检测系统，包括：

1. **模型训练**：使用 YOLOv8 对 15 类交通标志进行分类和检测。
2. **桌面应用**：使用 PyQt5 构建独立的桌面应用程序，支持图片上传和检测结果展示。

仅供参考，我的同学。
