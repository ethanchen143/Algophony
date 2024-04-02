import librosa
import numpy as np
import os
from audioldm.audio import TacotronSTFT
from characteristics import get_hifi_mel, get_inst


def delete_files(directory, required, forbidden):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            delete_file = True
            filename = filename.lower()
            for substring in required:
                if substring in filename:
                    delete_file = False
                    break
            for substring in forbidden:
                if substring in filename:
                    delete_file = True
            if delete_file:
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file: {file_path}")
                    print(e)


def process_data(data_start, data_end, inst):
    processed_count = 0
    entry_count = 0
    directory = '/Users/ethanchen/Desktop/algophony_sample'
    id, name, spec, inst = [], [], [], []
    target_length = 800
    sr = 16000

    # Initialize TacotronSTFT
    fn_STFT = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=64,
        sampling_rate=sr,
        mel_fmin=0,
        mel_fmax=8000,
    )

    for filename in os.listdir(directory):
        if processed_count >= data_start:
            if processed_count % 100 == 0:
                print(
                    f"{filename}\nfinished processing {processed_count} out of {data_end} with {entry_count} snippets")

        if processed_count >= data_end:
            break

        processed_count += 1
        if processed_count < data_start:
            continue

        try:
            y, _ = librosa.load(os.path.join(directory, filename), sr=sr)
            y = y / np.max(np.abs(y))  # Normalize

            silence_threshold = 0.01

            for start_idx in range(0, len(y), target_length * sr // 100):
                end_idx = start_idx + target_length * sr // 100
                if end_idx > len(y):
                    # If the segment is too short, pad it
                    segment = np.pad(y[start_idx:], (0, end_idx - len(y)), 'constant')
                else:
                    segment = y[start_idx:end_idx]

                rms_value = np.sqrt(np.mean(np.square(segment)))

                # Check if the segment is mostly silent
                if rms_value > silence_threshold:
                    mel_spec = get_hifi_mel(segment, fn_STFT).cpu().numpy()
                    spec.append(mel_spec)
                    id.append(entry_count)
                    name.append(filename)
                    inst.append(get_inst(filename))
                    entry_count += 1
                else:
                    # If the segment is mostly silent, skip it
                    pass

        except Exception as e:
            if (processed_count % 100 == 0):
                print(f"Failed to process {filename}: {e}")
            continue

    np.savez_compressed('../algo_spec.npz', id=np.array(id), name=np.array(name), spec=np.array(spec),
                        inst=np.array(inst))


if __name__ == '__main__':
    process_data(0, 85000)

    # one = np.load('../algo_spec_1.npz')
    # two = np.load('../algo_spec_2.npz')
    # name = np.concatenate((one['name'],two['name']))
    # id = np.concatenate((one['id'],two['id']))
    # inst = np.concatenate((one['inst'],two['inst']))
    # spec = np.concatenate((one['spec'],two['spec']))
    # np.savez_compressed('../algo_spec_22g.npz', id=id, name=name, spec=spec,
    #                     inst=inst)

    # directory = '/Users/ethanchen/Desktop/data'
    # required = [
    #     'key', 'rhode', 'piano', 'organ', 'clav',
    #     'guitar', 'gtr', 'strum', 'riff',
    #     'bass', '808', 'sub',
    #     'synth', 'arp', 'pad', 'lead', 'pluck',
    #     'drum', 'groove',
    #     'string', 'cello', 'violin', 'viola', 'vln', 'vla'
    # ]
    # forbidden = [
    #     'vocal', 'vox', 'voice', 'choir', 'male', 'female', 'vocoder', 'acapella',
    #     'kick', 'snare', 'clap', 'hihat', 'hat', 'snap', 'hi hat', 'shaker',
    #     'crash', 'ride', 'cymbal', 'rim', 'tom', 'one shot', 'one_shot', 'shot','fx'
    # ]
    # delete_files(directory,required,forbidden)
    # pass
