from pydub import AudioSegment

input_file = "test_audio/Recording.m4a"
output_file = "test_audio/test.wav"

audio = AudioSegment.from_file(input_file, format="m4a")

audio.export(output_file, format="wav")

print("Conversion completed!")
print("Saved as:", output_file)