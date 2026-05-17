from flask import Flask, request, send_file, render_template
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def apply_effect(y, sr, effect):

    y = np.array(y, dtype=np.float32)
    y = np.nan_to_num(y)

    if effect == "male":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-5)

    elif effect == "female":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=5)

    elif effect == "child":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=7)

    elif effect == "deep":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-8)

    elif effect == "fast":
        y = librosa.effects.time_stretch(y, rate=3)

    elif effect == "slow":
        y = librosa.effects.time_stretch(y, rate=0.3)


    elif effect == "echo":
        echo = np.zeros(len(y))
        delay = int(0.25 * sr)
        echo[delay:] = y[:-delay]
        y = y + 0.6 * echo

    elif effect == "radio":

        
        radio_bg, _ = librosa.load("assets/Radio Tuning Sound Effect [ HD ].mp3", sr=sr)

      
        if len(radio_bg) > len(y):
            radio_bg = radio_bg[:len(y)]
        else:
            radio_bg = np.pad(radio_bg, (0, len(y) - len(radio_bg)))

      
        b, a = signal.butter(4, [300/(sr/2), 3000/(sr/2)], btype='band')
        y_filtered = signal.lfilter(b, a, y)

      
        y_filtered = np.tanh(y_filtered * 2)

    
        y = 0.6 * y_filtered + 0.4 * radio_bg

     
        y = y / (np.max(np.abs(y)) + 1e-6)

    elif effect == "phone":
        ringtone, _ = librosa.load("assets/phone.mp3", sr=sr)

        if len(ringtone) > len(y):
            ringtone = ringtone[:len(y)]
        else:
            ringtone = np.pad(ringtone, (0, len(y) - len(ringtone)))

        b, a = signal.butter(4, [300/(sr/2), 3400/(sr/2)], btype='band')
        y = signal.lfilter(b, a, y)

        y = 0.7 * y + 0.3 * ringtone

    elif effect == "cave":
        echo = np.zeros(len(y))
        delay = int(0.6 * sr)
        echo[delay:] = y[:-delay]
        y = y + 0.8 * echo

    elif effect == "bassboost":
        y = np.clip(y * 2.0, -1, 1)

    elif effect == "reverb":
        reverb = np.zeros(len(y))
        delay = int(0.1 * sr)
        reverb[delay:] = y[:-delay]
        y = y + reverb

  
    elif effect == "robot":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        y = np.sign(y) * np.sqrt(np.abs(y))

    elif effect == "alien":
        alien_bg, _ = librosa.load("assets/neuralmatrix-alien.mp3", sr=sr)

        if len(alien_bg) > len(y):
            alien_bg = alien_bg[:len(y)]
        else:
            alien_bg = np.pad(alien_bg, (0, len(y) - len(alien_bg)))

        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=10)
        y_distorted = np.sin(y_shifted * 5)

        b, a = signal.butter(4, [500/(sr/2), 6000/(sr/2)], btype='band')
        y = signal.lfilter(b, a, y_distorted)

        y = 0.7 * y + 0.3 * alien_bg

    elif effect == "oldman":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-7)
        y = librosa.effects.time_stretch(y, rate=0.8)

    elif effect == "ghost":
        ghost_bg, _ = librosa.load("assets/viacheslavstarostin-ghost-mystery.mp3", sr=sr)

        if len(ghost_bg) > len(y):
            ghost_bg = ghost_bg[:len(y)]
        else:
            ghost_bg = np.pad(ghost_bg, (0, len(y) - len(ghost_bg)))

        echo = np.zeros(len(y))
        delay = int(0.35 * sr)
        echo[delay:] = y[:-delay]

        y = y + 0.6 * echo
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)

        b, a = signal.butter(4, [400/(sr/2), 5000/(sr/2)], btype='band')
        y = signal.lfilter(b, a, y)

        y = 0.7 * y + 0.3 * ghost_bg

    elif effect == "monster":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-10)
        y = np.clip(y * 2, -0.6, 0.6)

    elif effect == "chipmunk":
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=6)
        y = librosa.effects.time_stretch(y, rate=1.15)
        y = np.tanh(y * 1.5)
        y = y * 1.2

    return y



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/main")
def main():
    return render_template("main.html")



@app.route("/process", methods=["POST"])
def process():

    file = request.files["file"]
    effect = request.form.get("effect", "male")

    pitch = float(request.form.get("pitch", 0))
    speed = float(request.form.get("speed", 1))
    cutoff = float(request.form.get("cutoff", 4000))
    reverb_amount = float(request.form.get("reverb", 0.3))

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, "output.wav")

    file.save(input_path)

    y, sr = librosa.load(input_path, sr=None)

    y = apply_effect(y, sr, effect)


    pitch = max(-24, min(24, pitch))
    speed = max(0.5, min(2.0, speed))
    cutoff = max(100, min(cutoff, sr * 0.95))

    if pitch != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)

    if speed != 1:
        y = librosa.effects.time_stretch(y, rate=speed)


    nyquist = sr / 2
    cutoff_norm = min(cutoff / nyquist, 0.99)

    b, a = signal.butter(4, cutoff_norm, btype='low')
    y = signal.lfilter(b, a, y)

  
    delay = int(0.08 * sr)
    reverb = np.zeros(len(y))
    reverb[delay:] = y[:-delay]
    y = y + reverb_amount * reverb

    y = np.nan_to_num(y)

    sf.write(output_path, y, sr)

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
