import os
import librosa
import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
from trackdata import TrackData

tracks_count = 0

FONT_PATH = os.path.dirname(__file__) + "/04b03.ttf"

def setup():
    assert os.path.exists(FONT_PATH)
    font_manager.fontManager.addfont(FONT_PATH)

    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': '04b03',
        'text.color': 'white',
        'axes.labelcolor': 'white', 
        'axes.edgecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    })

def generate_spectrogram_image(trackdata: TrackData, path: str):
    fft_size = 4096
    window_size = 2048

    hop_size = window_size // 4

    stft = librosa.stft(
        trackdata.spectrogram_audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=window_size,
        center=False,
        window="blackman",
    )
    spectrogram = np.abs(stft)

    freqs = librosa.fft_frequencies(sr=trackdata.sample_rate, n_fft=fft_size)
    multiplier = np.linspace(1, 4, len(freqs))
    spectrogram = spectrogram * multiplier[:, np.newaxis]
    spectrogram = np.clip(spectrogram, 0, np.max(spectrogram) * 0.6)

    prepare_plotting()

    librosa.display.specshow(
        spectrogram,
        y_axis="log",
        x_axis="time",
        sr=trackdata.sample_rate,
        hop_length=hop_size,
        cmap="gray",
    )

    finalize_plotting(trackdata)
    plt.savefig(path)
    plt.close()

def prepare_plotting():
    # this gives us a 1280x1280 full image with 1024x1024 spectrogram
    plt.figure(figsize=(10, 10), dpi=128)
    plt.gca().set_position([0.1, 0.1, 0.8, 0.8])

def finalize_plotting(trackdata: TrackData):
    setup_xaxis(trackdata.start, trackdata.end)
    setup_yaxis(trackdata.min_freq, trackdata.max_freq)

    plt.xlabel('')
    plt.ylabel('')

    global tracks_count
    index_identifier = f"{trackdata.index}/{tracks_count}"
    main_title = f"FEZ Original Soundtrack Spectrograms - {trackdata.name} ({index_identifier})"

    plt.title(main_title, size=16, pad=20, loc='left')
    plt.title("v3", size=16, pad=20, loc='right', color='gray')

def setup_xaxis(start_time, end_time):
    plt.xticks(
        plt.gca().get_xlim(), 
        [f"{int(start_time // 60):02}:{int(start_time % 60):02}:{round((start_time % 1) * 1000):03}", 
         f"{int(end_time // 60):02}:{int(end_time % 60):02}:{round((end_time % 1) * 1000):03}"]
    )
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    plt.gca().xaxis.get_majorticklabels()[0].set_ha('left')
    plt.gca().xaxis.get_majorticklabels()[-1].set_ha('right')
    plt.gca().tick_params(axis='x', labelsize=16, pad=10)
    

def setup_yaxis(min_freq, max_freq):
    plt.ylim(min_freq, max_freq)
    ysteps = construct_ysteps(min_freq, max_freq)
    plt.yticks(ysteps, [f"{y}Hz" for y in ysteps])
    plt.gca().yaxis.set_minor_locator(plt.NullLocator())
    plt.gca().yaxis.get_majorticklabels()[0].set_va('bottom')
    plt.gca().yaxis.get_majorticklabels()[-1].set_va('top')

def construct_ysteps(min_freq, max_freq):
    ysteps = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ysteps = [y for y in ysteps if min_freq * 1.5 <= y <= max_freq * 0.75]
    ysteps = [min_freq] + ysteps + [max_freq]
    return ysteps