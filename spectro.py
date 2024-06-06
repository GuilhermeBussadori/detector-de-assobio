import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def create_spectrogram(audio_file, image_file, chunk_duration=30):
    plt.switch_backend('Agg')
    y, sr = librosa.load(audio_file, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(total_duration // chunk_duration)

    for i in range(num_chunks):
        start = int(i * chunk_duration * sr)
        end = int((i + 1) * chunk_duration * sr)
        y_chunk = y[start:end]
        S = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig = plt.figure(figsize=(10, 4), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
        fig.savefig(f"{image_file}_chunk_{i}.png", dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def create_pngs_from_audio_files(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in os.listdir(input_path):
        if file.endswith('.mp3') or file.endswith('.wav'):
            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, file.replace('.wav', '.png').replace('.mp3', '.png'))
            create_spectrogram(input_file, output_file)

create_pngs_from_audio_files('sla1', 'non_whistle_spectrograms')
