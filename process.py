import os
import sys
import json
import librosa
import soundfile as sf
import numpy as np
from matplotlib import font_manager
from PIL import Image
import matplotlib.pyplot as plt

CONFIG_PATH = os.path.dirname(__file__) + "/config.json"
OUT_IMG_PATH = os.path.dirname(__file__) + "/output/img/"
OUT_WAV_PATH = os.path.dirname(__file__) + "/output/wav/"

def load_config():
    with open(CONFIG_PATH, 'r') as data_file:
        global config
        config = json.load(data_file)
    for index, image_data in enumerate(config["images"]):
        image_data["index"] = index+1

def process_all_images(filter_name = None):
    force_reload_audio = (filter_name != None)
    for image_data in config["images"]:
        if filter_name != None and not image_data["track"].startswith(filter_name):
            continue
        print(f"Processing {image_data["track"]}...")
        process_image_data(image_data, force_reload_audio)

def process_image_data(image_data, force_reload_audio):
    spectrogram_audio, sample_rate = load_or_create_spectrogram_cut_audio(image_data, force_reload_audio)
    generate_spectrogram_image(image_data, spectrogram_audio, sample_rate)

def load_or_create_spectrogram_cut_audio(image_data, force_create):
    target_path = f"{OUT_WAV_PATH}{image_data["track"]}.wav"
    if os.path.exists(target_path) and not force_create:
        return sf.read(target_path)
    else:
        spectrogram_cut_audio, sample_rate = create_spectrogram_cut_audio(image_data)
        sf.write(target_path, spectrogram_cut_audio, sample_rate)
        return spectrogram_cut_audio, sample_rate

def create_spectrogram_cut_audio(image_data):
    puzzlified_audio_path = config["puzzlified_path_pattern"] % image_data["track"]
    unpuzzlified_audio_path = config["unpuzzlified_path_pattern"] % image_data["track"]

    puzzlified_audio, sample_rate = librosa.load(puzzlified_audio_path, sr=None, mono=True)
    unpuzzlified_audio, _ = librosa.load(unpuzzlified_audio_path, sr=sample_rate, mono=True)
    
    if "db_delta" in image_data:
        unpuzzlified_audio = librosa.db_to_amplitude(-image_data["db_delta"]) * unpuzzlified_audio

    length_diff = len(puzzlified_audio) - len(unpuzzlified_audio)
    if length_diff != 0:
        unpuzzlified_audio = np.pad(unpuzzlified_audio, (0, length_diff), 'constant')

    phase_cancelled_spectrogram_audio = puzzlified_audio - unpuzzlified_audio
    start_sample_time = int(image_data["start"] * sample_rate)
    end_sample_time = int(image_data["end"] * sample_rate)
    spectrogram_cut_audio = phase_cancelled_spectrogram_audio[start_sample_time:end_sample_time]
    return spectrogram_cut_audio, sample_rate

def generate_spectrogram_image(image_data, audio_signal, sample_rate):
    fft_size = 4096
    window_size = 2048

    hop_size = window_size // 4

    stft = librosa.stft(
        audio_signal,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=window_size,
        center=False,
        window="blackman",
    )
    spectrogram = np.abs(stft)

    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_size)
    multiplier = np.linspace(1, 4, len(freqs))
    spectrogram = spectrogram * multiplier[:, np.newaxis]
    spectrogram = np.clip(spectrogram, 0, np.max(spectrogram) * 0.6)

    plot_prepare()

    librosa.display.specshow(
        spectrogram,
        y_axis="log",
        x_axis="time",
        sr=sample_rate,
        hop_length=hop_size,
        cmap="gray",
    )

    plot_finalize(image_data)
    plt.savefig(f"{OUT_IMG_PATH}{image_data["track"]}.png")
    plt.close()

def plot_prepare():
    font_path = os.path.join(os.path.dirname(__file__), '04b03.ttf')
    assert os.path.exists(font_path)
    font_manager.fontManager.addfont(font_path)

    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': '04b03',
        'text.color': 'white',
        'axes.labelcolor': 'white', 
        'axes.edgecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    })

    # this gives us a 1280x1280 full image with 1024x1024 spectrogram
    plt.figure(figsize=(10, 10), dpi=128)
    plt.gca().set_position([0.1, 0.1, 0.8, 0.8])

def plot_finalize(image_data):
    setup_xaxis(image_data["start"], image_data["end"])
    setup_yaxis(image_data["min_freq"], image_data["max_freq"])

    plt.xlabel('')
    plt.ylabel('')

    index_identifier = f"{image_data["index"]}/{len(config["images"])}"
    main_title = f"FEZ Original Soundtrack Spectrograms - {image_data["track"]} ({index_identifier})"

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

def create_mosaic_image():
    images = []
    for image_data in config["images"]:
        image_path = f"{OUT_IMG_PATH}{image_data['track']}.png"
        if os.path.exists(image_path):
            images.append(Image.open(image_path))

    # Assuming all images are the same size
    img_width, img_height = images[0].size
    mosaic_width = img_width * 4
    mosaic_height = img_height * 4

    mosaic_image = Image.new('RGB', (mosaic_width, mosaic_height))

    for i, img in enumerate(images):
        x = (i % 4) * img_width
        y = (i // 4) * img_height
        mosaic_image.paste(img, (x, y))

    mosaic_image.save(f"{OUT_IMG_PATH}/../fez_spectrogram_images.png")



if __name__ == "__main__":
    load_config()
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    process_all_images(filter_name)
    create_mosaic_image()
    