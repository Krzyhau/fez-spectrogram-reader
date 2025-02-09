import librosa
import numpy as np
import soundfile

class TrackData:
    def __init__(self, track_data: dict, config: dict) -> None:
        self.index: int = track_data["index"]
        self.name : str = track_data["track"]
        self.start: float = track_data["start"]
        self.end: float = track_data["end"]
        self.min_freq: float = track_data["min_freq"]
        self.max_freq: float = track_data["max_freq"]

        self.puzzlified_audio_path: str = config["puzzlified_path_pattern"] % self.name
        self.unpuzzlified_audio_path: str = config["unpuzzlified_path_pattern"] % self.name

        self.db_delta: float = track_data["db_delta"] if "db_delta" in track_data else 0.0

    def load_spectrogram_audio(self, path: str):
        self.spectrogram_audio, self.sample_rate = librosa.load(path, sr=None, mono=True)

    def process_spectrogram_audio(self):
        self.load_audio_tracks()
        self.correct_unpuzzled_audio_volume()
        self.match_tracks_lenghts()
        self.isolate_and_cut_spectrogram_audio()

        return self.spectrogram_audio, self.sample_rate
    
    def load_audio_tracks(self):
        self.puzzlified_audio, self.sample_rate = librosa.load(self.puzzlified_audio_path, sr=None, mono=True)
        self.unpuzzlified_audio, _ = librosa.load(self.unpuzzlified_audio_path, sr=self.sample_rate, mono=True)
    
    def correct_unpuzzled_audio_volume(self):
        # Some tracks have been toned down in parts where spectrogram is mixed. 
        # It has to be corrected for the phase cancellation to work.
        self.unpuzzlified_audio *= librosa.db_to_amplitude(self.db_delta)
    
    def match_tracks_lenghts(self):
        # Other tracks have been prolonged by spectrogram being mixed after the end of track.
        # To compensate, unpuzzlified audio is prolonged to match the length of puzzlified audio.
        length_diff = len(self.puzzlified_audio) - len(self.unpuzzlified_audio)
        if length_diff != 0:
            self.unpuzzlified_audio = np.pad(self.unpuzzlified_audio, (0, length_diff), 'constant')

    def isolate_and_cut_spectrogram_audio(self):
        phase_cancelled_audio = self.puzzlified_audio - self.unpuzzlified_audio

        start_sample_time = int(self.start * self.sample_rate)
        end_sample_time = int(self.end * self.sample_rate)
        self.spectrogram_audio = phase_cancelled_audio[start_sample_time:end_sample_time]
    
    def export_spectrogram_audio(self, path):
        soundfile.write(path, self.spectrogram_audio, self.sample_rate)