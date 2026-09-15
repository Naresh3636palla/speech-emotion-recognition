from features import extract_features


file_path = "dataset/03-01-01-01-01-01-01.wav"

features = extract_features(file_path)

print("Feature shape:", features.shape)
print("Features:", features)