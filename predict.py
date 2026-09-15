import numpy as np
import librosa
import tensorflow as tf


MODEL_PATH = "models/emotion_model.keras"

emotion_labels = np.load(
    "models/emotion_labels.npy",
    allow_pickle=True
)

model = tf.keras.models.load_model(MODEL_PATH)


def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_scaled = np.mean(mfcc.T, axis=0)

    return mfcc_scaled


file_path = "test_audio/test.wav"

features = extract_features(file_path)

features = features.reshape(1, 40)

prediction = model.predict(features)

predicted_index = np.argmax(prediction)

predicted_emotion = emotion_labels[predicted_index]
emoji_map = {
    "angry": "😠",
    "calm": "😌",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}

predicted_emoji = emoji_map.get(predicted_emotion, "🙂")

confidence = prediction[0][predicted_index] * 100

print()
print("Predicted Emotion:", predicted_emotion,predicted_emoji)
print("Confidence:", round(confidence, 2), "%")