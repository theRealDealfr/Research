"""
Generate sample speech-like audio signals (16 kHz, 16-bit mono PCM .wav)
for testing inference and model serving pipelines.
"""
import math
import struct
import wave
import os

def create_sample_wav(filename="sample_child_speech.wav", duration_sec=3.0, sample_rate=16000):
    num_samples = int(duration_sec * sample_rate)
    # Simulate multi-harmonic formant frequencies typical of speech (~250Hz fundamental with harmonics)
    f0 = 260.0  # Typical child voice fundamental frequency
    harmonics = [(f0, 0.4), (f0 * 2, 0.25), (f0 * 3, 0.15), (f0 * 4, 0.1)]
    
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Add smooth amplitude envelope (fade in and fade out)
        env = min(1.0, t * 5.0) * min(1.0, (duration_sec - t) * 5.0)
        
        # Formant sum
        val = sum(amp * math.sin(2.0 * math.pi * freq * t) for freq, amp in harmonics)
        # Modulation to mimic syllables (at 3 Hz)
        syllable = 0.6 + 0.4 * math.sin(2.0 * math.pi * 3.0 * t)
        val = val * syllable * env
        
        # Clamp to 16-bit signed integer range
        int_val = int(max(-32767, min(32767, val * 24000.0)))
        samples.append(int_val)
        
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)        # 16-bit PCM (2 bytes)
        wf.setframerate(sample_rate)
        packed_data = struct.pack("<" + "h" * len(samples), *samples)
        wf.writeframes(packed_data)
        
    print(f"Generated sample audio file: {filename} ({duration_sec}s, {sample_rate}Hz mono)")
    return filename

if __name__ == "__main__":
    create_sample_wav("sample_child_speech.wav", duration_sec=3.5)
    create_sample_wav("sample_child_speech_alt.wav", duration_sec=4.0)
