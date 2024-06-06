import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def load_spectrograms(spectrogram_folder, label):
    spectrograms = []
    labels = []
    for file_name in os.listdir(spectrogram_folder):
        if file_name.endswith(".png"):
            file_path = os.path.join(spectrogram_folder, file_name)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (224, 224))  # Ajuste para o tamanho de entrada do MobileNetV2
            img = preprocess_input(img)  # Pré-processamento conforme MobileNetV2
            spectrograms.append(img)
            labels.append(label)
    return np.array(spectrograms), np.array(labels)

# Carregar e preparar dados
X_whistle, y_whistle = load_spectrograms("spectrogram_folder", 1)
X_no_whistle, y_no_whistle = load_spectrograms("non_whistle_spectrograms", 0)

# Oversampling para balanceamento de classes
X_no_whistle_upsampled, y_no_whistle_upsampled = resample(X_no_whistle, y_no_whistle,
                                                          replace=True, 
                                                          n_samples=len(X_whistle),
                                                          random_state=123)

# Combinar dados balanceados
X = np.concatenate((X_whistle, X_no_whistle_upsampled), axis=0)
y = np.concatenate((y_whistle, y_no_whistle_upsampled), axis=0)

# Embaralhar os dados
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# Divisão de dados para treinamento e teste
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Configuração do modelo usando MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar o modelo base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                              height_shift_range=0.1, shear_range=0.1, 
                              zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

# Pesos das classes para lidar com o desequilíbrio
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = {i: weights[i] for i in range(len(weights))}

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# Treinamento do modelo
history = model.fit(data_gen.flow(X_train, y_train, batch_size=32),
                    epochs=20,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, model_checkpoint],
                    class_weight=class_weight)

# Avaliação do modelo
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy of the model: {:.2f}%".format(accuracy * 100))

# Plotar gráficos de acurácia e perda
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r:', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r:', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


plot_history(history)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy of the model: {:.2f}%".format(accuracy * 100))

# Save the model
model.save('final_model.h5')