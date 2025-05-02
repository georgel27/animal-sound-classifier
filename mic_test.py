import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

duration = 3  # seconds
sample_rate = 22050

print("Recording for 3 seconds...")
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording complete!")

# Save to file
wav.write("mic_test.wav", sample_rate, (recording * 32767).astype(np.int16))
print("Saved as mic_test.wav")

# Play back
print("Playing back...")
sd.play(recording, samplerate=sample_rate)
sd.wait()
print("Mic test complete!")
