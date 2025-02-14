import os
import sys
import random

from trackdata import TrackData
import trackloader
import plotter
import mosaicmaker

from typing import List
from PIL import Image
import numpy as np

CONFIG_PATH = os.path.dirname(__file__) + "/config.json"
OUT_IMG_PATH = os.path.dirname(__file__) + "/output/img/"
OUT_WAV_PATH = os.path.dirname(__file__) + "/output/wav/"
MOSAIC_OUT_PATH = os.path.dirname(__file__) + "/output/fez_spectrogram_images.png"

def process():
    plotter.setup()
    tracks = trackloader.load_tracks(CONFIG_PATH)
    plotter.tracks_count = len(tracks)

    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    if filter_name:
        process_track_starting_with(tracks, filter_name)
    else:
        process_all_tracks(tracks)
    
    mosaicmaker.create(tracks, OUT_IMG_PATH, MOSAIC_OUT_PATH)

def process_track(trackdata : TrackData, force_wav : bool = False, force_img : bool = True):
    print(f"Processing track {trackdata.name}")
    output_wav_path = f"{OUT_WAV_PATH}{trackdata.name}.wav"
    output_img_path = f"{OUT_IMG_PATH}{trackdata.name}.png"

    if force_wav or not os.path.exists(output_wav_path):
        trackdata.process_spectrogram_audio()
        trackdata.export_spectrogram_audio(output_wav_path)
    else:
        trackdata.load_spectrogram_audio(output_wav_path)

    if force_img or not os.path.exists(output_img_path):
        create_interpolated_image(trackdata, output_img_path)

def create_interpolated_image(trackdata : TrackData, output_img_path : str):
    gen_count = 200

    fft_size = 4096
    window_size = 4096
    hop_size = 512

    fft_size_randomness = 256
    window_size_randomness = 256
    hop_size_randomness = 256

    temp_path = f"temp/{trackdata.name}.png"
    np_images = []

    for i in range(gen_count):
        gen_fft_size = random.randint(fft_size - fft_size_randomness, fft_size + fft_size_randomness)
        gen_window_size = min(random.randint(window_size - window_size_randomness, window_size + window_size_randomness), gen_fft_size)
        gen_hop_size = random.randint(hop_size - hop_size_randomness, hop_size + hop_size_randomness)
    
        print(f"  Generating image part {i+1} / {gen_count} with fft_size={gen_fft_size}, window_size={gen_window_size}, hop_size={gen_hop_size}")
        plotter.generate_spectrogram_image(trackdata, temp_path, gen_fft_size, gen_window_size, gen_hop_size)

        img = Image.open(temp_path)
        np_images.append(np.array(img))
        os.remove(temp_path)

    avg_image = np.mean(np_images, axis=0).astype(np.uint8)
    avg_image = Image.fromarray(avg_image)
    avg_image.save(output_img_path)


def process_track_starting_with(tracks : List[TrackData], filter_name : str):
    tracks = [track for track in tracks if track.name.startswith(filter_name)]
    if tracks: process_track(tracks[0], force_wav=True)

def process_all_tracks(tracks : List[TrackData]):
    for track in tracks:
        process_track(track)


if __name__ == "__main__":
    process()
    