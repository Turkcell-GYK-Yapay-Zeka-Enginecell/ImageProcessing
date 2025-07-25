import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

folder_path = "images"

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

data = []
labels = []

for file in os.listdir(folder_path):
    if file.endswith(".jpg"):
        path = os.path.join(folder_path, file)
        label =  "_".join(file.split("_")[:-1])   # 'cat.jpg' -> 'cat'
        # if label in labels:
        #     # Görüntüyü oku ve boyutlandır
        #     img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        #     img_array = np.array(img) / 255.0
        #     data.append(img_array)
        # else:
        #Görüntüyü oku ve boyutlandır
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        data.append(img_array)
        labels.append(label)
        # print(label)
        # print(img_array)

# Diziye çevir
data = np.array(data)
labels = np.array(labels) #37 sınıf


# Etiketleri sayıya çevir
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)


# Eğitim/test verisini ayır
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) # Bu kısımda verisetini TensorDataset'e çeviriyoruz.
train_ds = train_ds.shuffle(10000) # Burada veri setini rastgele karıştırıyoruz.
train_ds = train_ds.batch(BATCH_SIZE) # Burada batchlere böl. 32/32/32
train_ds = train_ds.prefetch(AUTOTUNE) # Pre-Fetch ile X adet veriyi önden çekiyoruz.


test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    include_top = False,
    weights = "imagenet",
    input_shape = (IMG_SIZE,IMG_SIZE,3), #224x224'ten küçük olmalı
    # input_tensor= ,
    #pooling = ,
    #classes = ,
    #classifier_activation = ,
)

base_model.trainable =False #base modelin tekrar eğitilmesini durduruyoruz

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(37, activation='softmax') #kaç sınıf varsa o kadar
])

model.summary()
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_ds,epochs=10,validation_data=test_ds, callbacks=[TqdmCallback(verbose=1)])
model.save("pet_classification.h5")
