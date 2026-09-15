import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Dataset location
DATASET_PATH = "dataset"


# Emotion mapping
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# Lists to store data
X = []
y = []


print("Reading audio files...")


# Read all WAV files
for filename in os.listdir(DATASET_PATH):

    if filename.endswith(".wav"):

        file_path = os.path.join(DATASET_PATH, filename)

        # Get emotion code from filename
        emotion_code = filename.split("-")[2]

        # Check whether emotion exists
        if emotion_code in emotion_map:

            emotion = emotion_map[emotion_code]

            try:

                # Load audio
                audio, sample_rate = librosa.load(
                    file_path,
                    duration=3,
                    offset=0.5
                )

                # Extract MFCC
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sample_rate,
                    n_mfcc=40
                )

                # Average MFCC values
                mfcc_scaled = np.mean(mfcc.T, axis=0)

                X.append(mfcc_scaled)
                y.append(emotion)

            except Exception as e:

                print("Error:", filename)
                print(e)


print("Finished reading audio files.")

print("Total samples:", len(X))
from collections import Counter

print("Emotion counts:")
print(Counter(y))


# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)


# Encode emotion labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)


print("Feature shape:", X.shape)
print("Number of emotions:", len(label_encoder.classes_))
print("Emotions:", label_encoder.classes_)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
from collections import Counter

print("\nEmotion counts:")
print(Counter(y))

# Build neural network
model = Sequential([

    Dense(128, activation="relu", input_shape=(40,)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(len(label_encoder.classes_), activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Display model
model.summary()


# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# Evaluate
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)


print()
print("Test Accuracy:", test_accuracy)


# Create models folder
os.makedirs("models", exist_ok=True)


# Save model
model.save("models/emotion_model.keras")


# Save emotion labels
np.save(
    "models/emotion_labels.npy",
    label_encoder.classes_
)


print()
print("Model saved successfully!")
print("Location: models/emotion_model.keras")