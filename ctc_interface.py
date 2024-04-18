import argparse
import wave

import torch
import torchaudio
import pyaudio
import numpy as np
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=kwargs["nfilt"],
            window_fn=self.torch_windows[kwargs["window"]],
            mel_scale=mel_scale,
            norm=kwargs["mel_norm"],
            n_fft=kwargs["n_fft"],
            f_max=kwargs.get("highfreq", None),
            f_min=kwargs.get("lowfreq", 0),
            wkwargs=wkwargs,
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
            mel_scale=mel_scale, **kwargs,
        )



class SpeechRecognizer:
    def __init__(self):
        self.chunk = 2048
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        self.threshold = self.calibrate_microphone()
        #self.threshold = 100
        self.gigaAM = "data/ctc_model_weights.ckpt"
        self.config = "data/ctc_model_config.yaml"
        self.device = "cpu"
        self.model = EncDecCTCModel.from_config_file(self.config)
        self.ckpt = torch.load(self.gigaAM, map_location="cpu")
        self.model.load_state_dict(self.ckpt, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()


    def calibrate_microphone(self):
        print("Calibrating... Be quiet!")
        frames = [self.stream.read(self.chunk) for _ in range(0, int(self.rate / self.chunk * 2))]
        data_int = np.frombuffer(b''.join(frames), dtype=np.int16)
        new_threshold = np.max(data_int)
        print(f"Calibration complete. New threshold is {new_threshold}")
        return new_threshold

    def record_audio(self, frames):

        filename = "speech.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        return filename


    def recognize_speech(self):
        print("Listening...")
        frames = []
        while True:
            data = self.stream.read(self.chunk)
            data_int = np.frombuffer(data, dtype=np.int16)
            if np.max(data_int) > self.threshold:
                frames.append(data)
            else:
                if frames:
                    #noisy_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                    #reduced_noise = nr.reduce_noise(y=noisy_data, sr=self.rate)
                    #frames = [reduced_noise.tobytes()]
                    self.record_audio(frames)
                    print("Wait...")
                    transcription = self.model.transcribe(["speech.wav"])[0]
                    print("Recognized:", transcription)
                    frames = []
                    print("Listening...")

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


recognizer = SpeechRecognizer()
try:
    recognizer.recognize_speech()
except KeyboardInterrupt:
    recognizer.close()
