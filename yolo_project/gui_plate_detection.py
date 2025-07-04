import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

# YOLOv8 modelini yükle (.pt dosyanın yolu)
model = YOLO(r"C:\Users\ZühalÖzdemir\PycharmProjects\yolo_project\runs\detect\train\weights\best.pt")


# Görselde plakayı bulur ve çerçeveler
def detect_and_draw(image_path):
    image = cv2.imread(image_path)
    results = model(image_path)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 7)  # yeşil kutu
        cv2.putText(image, "plaka", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    # BGR -> RGB (Tkinter için)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# Görseli seçip GUI'de göster
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        processed_image = detect_and_draw(file_path)
        processed_image = processed_image.resize((800, 800))  # Pencereye sığdır
        photo = ImageTk.PhotoImage(processed_image)
        image_label.config(image=photo)
        image_label.image = photo

# GUI Arayüzü
root = tk.Tk()
root.title("YOLOv8 Plaka Çerçeveleme")

frame = tk.Frame(root)
frame.pack(pady=10)

btn = tk.Button(frame, text="Görsel Seç", command=browse_image)
btn.pack()

image_label = tk.Label(root)
image_label.pack()

root.mainloop()
