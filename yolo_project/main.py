#transfer learning
from ultralytics import YOLO


def train():
    model = YOLO("yolov8n.pt")
    model.train(data="data.yaml", epochs=10, imgsz=640, batch=8)
    #1955/8

# Pythondaki main method.
# python main.py => Çalış
# başka bir dosyadan import edildiğinde çalışmaz.
if __name__ == "__main__":
    train()