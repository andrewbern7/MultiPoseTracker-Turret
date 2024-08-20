import sounddevice as sd
import vosk
import json
import threading
from config import SAFEWORD, vosk_model_path
class SafewordListener:
    def __init__(self, safeword=SAFEWORD, model_path=vosk_model_path):
        self.safeword = safeword
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.shutdown_triggered = False
        self.listener_thread = threading.Thread(target=self.listen_for_safeword)

    def start(self):
        """Start the safeword listener in a separate thread."""
        self.listener_thread.start()

    def listen_for_safeword(self):
        with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                               channels=1, callback=self.callback):
            print("Listening for safeword...")
            while not self.shutdown_triggered:
                pass  # Keep the stream open

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recognizer.AcceptWaveform(bytes(indata)):
            result = self.recognizer.Result()
            result_json = json.loads(result)
            text = result_json.get('text', '')
            if text:
                print(f"Detected speech: {text}")
            if self.safeword in text.lower():
                print("Safeword detected! Shutting down the turret...")
                self.shutdown_triggered = True

    def is_shutdown_triggered(self):
        return self.shutdown_triggered
