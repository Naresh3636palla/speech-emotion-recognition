import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
from scipy.io.wavfile import write
import os
import threading
import time
import winsound


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "models/emotion_model.keras"

emotion_labels = np.load(
    "models/emotion_labels.npy",
    allow_pickle=True
)

model = tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# EMOJI
# =========================================================

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


# =========================================================
# VARIABLES
# =========================================================

selected_file = ""
recording = False
audio_data = None

SAMPLE_RATE = 44100
MAX_RECORD_SECONDS = 10

record_start_time = 0


# =========================================================
# FEATURE EXTRACTION
# =========================================================

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

    mfcc_scaled = np.mean(
        mfcc.T,
        axis=0
    )

    return mfcc_scaled


# =========================================================
# UPLOAD AUDIO
# =========================================================

def choose_audio():

    global selected_file

    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[
            ("WAV files", "*.wav"),
            ("Audio files", "*.wav *.mp3 *.m4a"),
            ("All files", "*.*")
        ]
    )

    if file_path:

        selected_file = file_path

        file_name = os.path.basename(
            file_path
        )

        file_label.config(
            text="🎵 " + file_name,
            fg="#FFFFFF"
        )

        status_label.config(
            text="Audio selected",
            fg="#86EFAC"
        )

        emotion_label.config(
            text="READY",
            fg="#D8B4FE"
        )

        confidence_label.config(
            text="Confidence: --"
        )


# =========================================================
# START RECORDING
# =========================================================

def start_recording():

    global recording
    global audio_data
    global record_start_time

    if recording:
        return

    recording = True

    record_start_time = time.time()

    record_button.config(
        text="🔴 RECORDING",
        bg="#DC2626"
    )

    status_label.config(
        text="🎙️ Speak now...",
        fg="#FCA5A5"
    )

    timer_label.config(
        text="Recording: 00:00"
    )

    def record_audio():

        global audio_data
        global recording

        try:

            audio_data = sd.rec(
                int(
                    MAX_RECORD_SECONDS *
                    SAMPLE_RATE
                ),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

        except Exception as e:

            recording = False

            window.after(
                0,
                lambda: status_label.config(
                    text="Recording error",
                    fg="#FCA5A5"
                )
            )

    threading.Thread(
        target=record_audio,
        daemon=True
    ).start()

    update_timer()


# =========================================================
# RECORDING TIMER
# =========================================================

def update_timer():

    if recording:

        elapsed = int(
            time.time() -
            record_start_time
        )

        if elapsed >= MAX_RECORD_SECONDS:

            stop_recording()

            return

        timer_label.config(
            text=f"Recording: 00:{elapsed:02d}"
        )

        window.after(
            500,
            update_timer
        )


# =========================================================
# STOP RECORDING
# =========================================================

def stop_recording():

    global recording
    global selected_file

    if not recording:
        return

    recording = False

    try:
        sd.stop()
    except:
        pass

    record_button.config(
        text="🎤 RECORD",
        bg="#B91C1C"
    )

    timer_label.config(
        text="Recording: Finished"
    )

    status_label.config(
        text="✅ Recording completed",
        fg="#86EFAC"
    )

    if audio_data is not None:

        os.makedirs(
            "test_audio",
            exist_ok=True
        )

        selected_file = (
            "test_audio/recorded_voice.wav"
        )

        try:

            write(
                selected_file,
                SAMPLE_RATE,
                audio_data
            )

            file_label.config(
                text="🎙️ recorded_voice.wav",
                fg="#FFFFFF"
            )

        except Exception as e:

            status_label.config(
                text="Could not save recording",
                fg="#FCA5A5"
            )


# =========================================================
# PLAY AUDIO
# =========================================================

def play_audio():

    if not selected_file:

        messagebox.showwarning(
            "No Audio",
            "Please record or upload an audio file first."
        )

        return

    try:

        if selected_file.lower().endswith(
            ".wav"
        ):

            winsound.PlaySound(
                selected_file,
                winsound.SND_FILENAME |
                winsound.SND_ASYNC
            )

        else:

            messagebox.showinfo(
                "Playback",
                "For playback, please use a WAV file."
            )

    except Exception as e:

        messagebox.showerror(
            "Playback Error",
            str(e)
        )


# =========================================================
# STOP AUDIO
# =========================================================

def stop_audio():

    try:

        winsound.PlaySound(
            None,
            winsound.SND_PURGE
        )

    except:
        pass


# =========================================================
# PREDICT EMOTION
# =========================================================

def predict_emotion():

    if not selected_file:

        emotion_label.config(
            text="SELECT AUDIO",
            fg="#FCA5A5"
        )

        return

    try:

        status_label.config(
            text="🤖 Analyzing...",
            fg="#FDE68A"
        )

        window.update()

        features = extract_features(
            selected_file
        )

        features = features.reshape(
            1,
            40
        )

        prediction = model.predict(
            features,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction
        )

        predicted_emotion = str(
            emotion_labels[predicted_index]
        )

        confidence = (
            prediction[0][predicted_index]
            * 100
        )

        emoji = emoji_map.get(
            predicted_emotion,
            "🙂"
        )

        # Result

        result_label.config(
            text=emoji
        )

        emotion_label.config(
            text=predicted_emotion.upper(),
            fg="#F0ABFC"
        )

        confidence_label.config(
            text=f"✨ Confidence: {confidence:.2f}%",
            fg="#86EFAC"
        )

        status_label.config(
            text="✅ Analysis completed",
            fg="#86EFAC"
        )

        # Chart

        update_chart(
            prediction[0]
        )

        # History

        add_history(
            predicted_emotion,
            confidence
        )

    except Exception as e:

        emotion_label.config(
            text="ERROR",
            fg="#FCA5A5"
        )

        status_label.config(
            text=str(e),
            fg="#FCA5A5"
        )


# =========================================================
# PROBABILITY BARS
# =========================================================

def update_chart(probabilities):

    for widget in chart_frame.winfo_children():
        widget.destroy()

    for i, emotion in enumerate(
        emotion_labels
    ):

        percentage = float(
            probabilities[i] * 100
        )

        row = tk.Frame(
            chart_frame,
            bg="#241333"
        )

        row.pack(
            fill="x",
            pady=1
        )

        emoji = emoji_map.get(
            str(emotion),
            "🙂"
        )

        # Emotion name

        name = tk.Label(
            row,
            text=f"{emoji} {str(emotion).capitalize()}",
            font=("Segoe UI", 8, "bold"),
            width=11,
            anchor="w",
            bg="#241333",
            fg="#E2E8F0"
        )

        name.pack(
            side="left"
        )

        # Background bar

        bar_bg = tk.Frame(
            row,
            bg="#374151",
            width=230,
            height=9
        )

        bar_bg.pack(
            side="left",
            padx=4
        )

        bar_bg.pack_propagate(
            False
        )

        # Filled bar

        bar_width = int(
            230 *
            percentage /
            100
        )

        bar = tk.Frame(
            bar_bg,
            bg="#A855F7",
            width=bar_width,
            height=9
        )

        bar.pack(
            side="left"
        )

        # Percentage

        percent = tk.Label(
            row,
            text=f"{percentage:.1f}%",
            font=("Segoe UI", 8, "bold"),
            bg="#241333",
            fg="#CBD5E1"
        )

        percent.pack(
            side="left"
        )


# =========================================================
# HISTORY
# =========================================================

def add_history(
    emotion,
    confidence
):

    emoji = emoji_map.get(
        emotion,
        "🙂"
    )

    current_time = time.strftime(
        "%H:%M"
    )

    history_list.insert(
        0,
        f"{current_time}  {emoji} "
        f"{emotion.upper()}  "
        f"{confidence:.1f}%"
    )


# =========================================================
# RESET
# =========================================================

def reset_app():

    global selected_file

    selected_file = ""

    result_label.config(
        text="🎤"
    )

    emotion_label.config(
        text="READY",
        fg="#D8B4FE"
    )

    confidence_label.config(
        text="Confidence: --",
        fg="#CBD5E1"
    )

    file_label.config(
        text="🎵 No audio selected",
        fg="#CBD5E1"
    )

    status_label.config(
        text="Waiting for audio...",
        fg="#94A3B8"
    )

    timer_label.config(
        text="Recording: --"
    )

    for widget in chart_frame.winfo_children():
        widget.destroy()


# =========================================================
# MAIN WINDOW
# =========================================================

window = tk.Tk()

window.title(
    "🎤 Emotion AI - Speech Emotion Recognition"
)

# SMALL LAPTOP-FRIENDLY SIZE

window.geometry(
    "650x720"
)

window.resizable(
    False,
    False
)

window.configure(
    bg="#0B1020"
)


# =========================================================
# OUTER BORDER
# =========================================================

outer_border = tk.Frame(
    window,
    bg="#A855F7",
    padx=2,
    pady=2
)

outer_border.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


main_frame = tk.Frame(
    outer_border,
    bg="#111827"
)

main_frame.pack(
    fill="both",
    expand=True
)


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    main_frame,
    bg="#4C1D95",
    height=90
)

header.pack(
    fill="x"
)

header.pack_propagate(
    False
)


tk.Label(
    header,
    text="🎤  EMOTION AI",
    font=("Segoe UI", 22, "bold"),
    bg="#4C1D95",
    fg="#FFFFFF"
).pack(
    pady=(14, 0)
)


tk.Label(
    header,
    text="Speech Emotion Recognition System",
    font=("Segoe UI", 9),
    bg="#4C1D95",
    fg="#DDD6FE"
).pack()


# =========================================================
# CONTENT
# =========================================================

content = tk.Frame(
    main_frame,
    bg="#111827"
)

content.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=8
)


# =========================================================
# AUDIO INPUT CARD
# =========================================================

audio_border = tk.Frame(
    content,
    bg="#38BDF8",
    padx=2,
    pady=2
)

audio_border.pack(
    fill="x"
)


audio_card = tk.Frame(
    audio_border,
    bg="#172033",
    padx=10,
    pady=7
)

audio_card.pack(
    fill="x"
)


tk.Label(
    audio_card,
    text="🎵 AUDIO INPUT",
    font=("Segoe UI", 9, "bold"),
    bg="#172033",
    fg="#38BDF8"
).pack()


file_label = tk.Label(
    audio_card,
    text="🎵 No audio selected",
    font=("Segoe UI", 9, "bold"),
    bg="#172033",
    fg="#CBD5E1"
)

file_label.pack(
    pady=3
)


# =========================================================
# RECORD / UPLOAD BUTTONS
# =========================================================

button_frame = tk.Frame(
    audio_card,
    bg="#172033"
)

button_frame.pack()


record_button = tk.Button(
    button_frame,
    text="🎤 RECORD",
    font=("Segoe UI", 9, "bold"),
    bg="#B91C1C",
    fg="white",
    activebackground="#DC2626",
    relief="flat",
    padx=12,
    pady=6,
    command=start_recording
)

record_button.grid(
    row=0,
    column=0,
    padx=3
)


tk.Button(
    button_frame,
    text="⏹ STOP",
    font=("Segoe UI", 9, "bold"),
    bg="#374151",
    fg="white",
    activebackground="#4B5563",
    relief="flat",
    padx=12,
    pady=6,
    command=stop_recording
).grid(
    row=0,
    column=1,
    padx=3
)


tk.Button(
    button_frame,
    text="📁 UPLOAD",
    font=("Segoe UI", 9, "bold"),
    bg="#0369A1",
    fg="white",
    activebackground="#0284C7",
    relief="flat",
    padx=12,
    pady=6,
    command=choose_audio
).grid(
    row=0,
    column=2,
    padx=3
)


timer_label = tk.Label(
    audio_card,
    text="Recording: --",
    font=("Segoe UI", 8),
    bg="#172033",
    fg="#94A3B8"
)

timer_label.pack(
    pady=2
)


status_label = tk.Label(
    audio_card,
    text="Waiting for audio...",
    font=("Segoe UI", 8),
    bg="#172033",
    fg="#94A3B8"
)

status_label.pack()


# =========================================================
# PLAY / STOP AUDIO
# =========================================================

play_frame = tk.Frame(
    content,
    bg="#111827"
)

play_frame.pack(
    pady=5
)


tk.Button(
    play_frame,
    text="▶ PLAY",
    font=("Segoe UI", 8, "bold"),
    bg="#059669",
    fg="white",
    relief="flat",
    padx=15,
    pady=5,
    command=play_audio
).grid(
    row=0,
    column=0,
    padx=3
)


tk.Button(
    play_frame,
    text="⏹ STOP AUDIO",
    font=("Segoe UI", 8, "bold"),
    bg="#4B5563",
    fg="white",
    relief="flat",
    padx=15,
    pady=5,
    command=stop_audio
).grid(
    row=0,
    column=1,
    padx=3
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

tk.Button(
    content,
    text="🔮  ANALYZE EMOTION",
    font=("Segoe UI", 10, "bold"),
    bg="#9333EA",
    fg="white",
    activebackground="#A855F7",
    relief="flat",
    padx=30,
    pady=7,
    command=predict_emotion
).pack(
    pady=4
)


# =========================================================
# RESULT CARD
# =========================================================

result_border = tk.Frame(
    content,
    bg="#EC4899",
    padx=2,
    pady=2
)

result_border.pack(
    fill="x"
)


result_card = tk.Frame(
    result_border,
    bg="#241333",
    padx=10,
    pady=5
)

result_card.pack(
    fill="x"
)


tk.Label(
    result_card,
    text="✨ DETECTED EMOTION ✨",
    font=("Segoe UI", 9, "bold"),
    bg="#241333",
    fg="#F9A8D4"
).pack()


result_label = tk.Label(
    result_card,
    text="🎤",
    font=("Segoe UI Emoji", 28),
    bg="#241333"
)

result_label.pack(
    pady=0
)


emotion_label = tk.Label(
    result_card,
    text="READY",
    font=("Segoe UI", 16, "bold"),
    bg="#241333",
    fg="#D8B4FE"
)

emotion_label.pack()


confidence_label = tk.Label(
    result_card,
    text="Confidence: --",
    font=("Segoe UI", 9, "bold"),
    bg="#241333",
    fg="#CBD5E1"
)

confidence_label.pack(
    pady=2
)


# =========================================================
# CHART
# =========================================================

tk.Label(
    result_card,
    text="📊 EMOTION ANALYSIS",
    font=("Segoe UI", 8, "bold"),
    bg="#241333",
    fg="#F9A8D4"
).pack(
    pady=(2, 1)
)


chart_frame = tk.Frame(
    result_card,
    bg="#241333"
)

chart_frame.pack(
    fill="x"
)


# =========================================================
# HISTORY
# =========================================================

tk.Label(
    content,
    text="🕒 PREDICTION HISTORY",
    font=("Segoe UI", 8, "bold"),
    bg="#111827",
    fg="#C4B5FD"
).pack(
    pady=(4, 2)
)


history_list = tk.Listbox(
    content,
    height=2,
    font=("Consolas", 8),
    bg="#172033",
    fg="#CBD5E1",
    bd=0,
    highlightthickness=1,
    highlightbackground="#374151"
)

history_list.pack(
    fill="x"
)


# =========================================================
# RESET
# =========================================================

tk.Button(
    content,
    text="🔄 RESET",
    font=("Segoe UI", 8, "bold"),
    bg="#374151",
    fg="white",
    activebackground="#4B5563",
    relief="flat",
    padx=18,
    pady=5,
    command=reset_app
).pack(
    pady=5
)


# =========================================================
# FOOTER
# =========================================================

tk.Label(
    main_frame,
    text="🤖 AI-powered • Speech Emotion Recognition • Student Project",
    font=("Segoe UI", 7),
    bg="#111827",
    fg="#64748B"
).pack(
    pady=(0, 5)
)


# =========================================================
# START APPLICATION
# =========================================================

window.mainloop()