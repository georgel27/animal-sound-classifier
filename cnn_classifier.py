import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # To save/load label encoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset_path = 'C:\\Users\\George\\Desktop\\animalsoundclassification-main\\animalsoundclassification-main\\Dataset'  
model_path = "C:\\Users\\George\\Desktop\\animalsoundclassification-main\\animalsoundclassification-main\\dog_sound_classifier.h5"
label_encoder_path = "C:\\Users\\George\\Desktop\\animalsoundclassification-main\\animalsoundclassification-main\\label_encoder.pkl"

# Classes of animal sound
classes = ["dog_bark", "dog_growl", "dog_grunt", "dog_whine", "dog_howl", "dog_moan", "dog_whimper","dog_snarl", "dog_yip", "dog_sniff", "dog_bay", "dog_pant"]

def extract_features(file_path, max_pad_len=100):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("Error processing file", file_path, e)
        return None

# Load dataset
def load_dataset(split="train"):
    features, labels = [], []
    for category in classes:
        folder_path = os.path.join(dataset_path, f"{category}_{split}")
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(category)
    return np.array(features), np.array(labels)

# Initialize label encoder
label_encoder = LabelEncoder()

# Check if model exists
if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    print("Loading existing model and label encoder...")
    model = keras.models.load_model(model_path)
    label_encoder = joblib.load(label_encoder_path)
else:
    print("Training new model...")
    # Load train and test sets
    X_train, y_train = load_dataset("train")
    X_test, y_test = load_dataset("test")

    # Encode labels
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Save label encoder
    joblib.dump(label_encoder, label_encoder_path)

    # Reshape for CNN input
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Build CNN Model
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(40, 100, 1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(classes), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}")

    # Classification report and confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification report as text
    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix as heatmap
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    df_cm = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    plt.figure(figsize=(6,5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Save the model
    model.save(model_path)

def predict_sound(file_path):
    if not os.path.exists(label_encoder_path):
        return "Error: Label encoder not found. Train the model first."
    label_encoder = joblib.load(label_encoder_path)

    feature = extract_features(file_path)
    if feature is None:
        return "Error: Could not process file"
    feature = feature[np.newaxis, ..., np.newaxis]
    prediction = model.predict(feature)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Testing a wave file from a random source
print(predict_sound('C:\\Users\\George\\Desktop\\animalsoundclassification-main\\animalsoundclassification-main\\Testing\\dog_bark_test.wav'))
