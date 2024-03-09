import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from characteristics import getTheme,getMoodGroup,getSad,\
    getHappy,getRelaxed,getAggresive,getPopularity,getEngagement,getGenre,getTimbre

def process_data():
    first = 7500
    count = 0

    directory = "./algophony_sample/"
    name,id,spectrogram,instrument,theme,genre,\
        mood,happy,sad,aggresive,relaxed,popularity,engagement = [],[],[],[],[],[],[],[],[],[],[],[],[]

    # 30 seconds in sample frames (sr = 22050)
    target_length = 30 * 22050
    for filename in os.listdir(directory):
        if count % 10 == 0:
            print(filename)
            print("finished processing " + f"{count} out of {first}")
        count += 1
        startFrom = 0
        if count < startFrom:
            continue

        try:
            y, sr = librosa.load(directory + filename,sr = 22050)
            sample_length = len(y)
            if sample_length > target_length:
                # If longer, trim the excess at the end
                y = y[:target_length]
            elif sample_length < target_length:
                # If shorter, pad with zeros (silence) at the end
                padding_length = target_length - sample_length
                y = np.pad(y, (0, padding_length), mode='constant')
        except Exception:
            print("Not an audio file")
            continue

        # Compute the spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr)

        # audio_signal = librosa.feature.inverse.mel_to_audio(S)
        # import soundfile as sf
        # sf.write('test.wav', audio_signal, sr)

        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr)
        plt.tight_layout()
        output_dir = "./algophony_spectrogram/" + str(count) + ".png"
        plt.savefig(output_dir)

        if count == first:
            break

        name.append(filename)
        id.append(count)
        spectrogram.append(S)
        genre.append(getGenre(directory + filename,3))
        mood.append(getMoodGroup(directory + filename))
        theme.append(getTheme(directory + filename,5))
        happy.append(getHappy(directory + filename)[0])
        sad.append(getSad(directory + filename)[0])
        aggresive.append(getAggresive(directory + filename)[0])
        relaxed.append(getRelaxed(directory + filename)[0])
        popularity.append(getPopularity(directory + filename))
        engagement.append(getEngagement(directory + filename))

        vocal = r'(vocal|vox|voice|choir|male|female)'
        if re.findall(vocal, filename, re.IGNORECASE):
            instrument.append('vocal')
            continue
        key = r'(key|rhode|piano|ep)'
        if re.findall(key, filename, re.IGNORECASE):
            instrument.append('key')
            continue
        ag = r'(acoustic&(guitar|gtr|strum))'
        if re.findall(ag, filename, re.IGNORECASE):
            instrument.append('acoustic guitar')
            continue
        eg = r'(guitar|gtr|strum)'
        if re.findall(eg, filename, re.IGNORECASE):
            instrument.append('electric guitar')
            continue
        bass = r'(bass|808|sub)'
        if re.findall(bass, filename, re.IGNORECASE):
            instrument.append('bass')
            continue
        synth = r'(synth|arp|pad|lead|pluck)'
        if re.findall(synth, filename, re.IGNORECASE):
            instrument.append('synth')
            continue
        drum = r'(drum|perc|top)'
        if re.findall(drum, filename, re.IGNORECASE):
            instrument.append('drum')
            continue
        instrument.append('others')

    df = pd.DataFrame({
        'Name': name,
        'ID': id,
        'Spectrogram': spectrogram,
        'Instrument': instrument,
        'Genre': genre,
        'Mood': mood,
        'Theme': theme,
        'Happy': happy,
        'Sad': sad,
        'Aggressive': aggresive,
        'Relaxed': relaxed,
        'Popularity': popularity,
        'Engagement': engagement
    })
    # Export to CSV
    csv_filename = 'data1.csv'
    df.to_csv(csv_filename, index=False)




# # find patterns of two to three digits
# # pattern: find 2/3 consequent digits that is not preceded or followed by a digit
# matches = re.findall(r'(?:(?<=\D)|^)\d{2,3}(?=\D|$)', filename)
# candidates = [int(match) for match in matches if 60 <= int(match) <= 200]
# result = candidates[0] if candidates else None
# bpm.append(result)
#
# # list all possible keys
# capital = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A#', 'C#', 'D#', 'F#', 'G#', 'Ab', 'Bb', 'Db', 'Eb', 'Gb']
# low = [letter.lower() for letter in capital]
# keys = [note + suffix for note in capital for suffix in ['M','m','min','maj','Maj','MAJ','major','minor']]
# keys += [f"{note}_{suffix}" for note in capital for suffix in ['min','maj','Maj','MAJ','Major','Minor']]
# keys += capital
# keys += [note + 'min' for note in low]
# keys += [f"{note}_min" for note in low]
# keys += [f"{note}_minor" for note in low]
# # match longer keys first, escape special character
# pattern = '|'.join(sorted(map(re.escape, keys), key=len, reverse=True))
# found_keys = re.findall(pattern, filename)
# found_keys = sorted(found_keys, key=lambda x: len(x))
# result = found_keys[-1] if found_keys else None
# key.append(result)