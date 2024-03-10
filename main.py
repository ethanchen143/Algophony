from data import process_data
import numpy as np
import pandas as pd

import librosa
import soundfile as sf


if __name__ == '__main__':
    # vocoder = SpeechT5HifiGan.from_pretrained("cvssp/audioldm-s-full-v2", subfolder='vocoder')
    # remember to change file name!!!
    process_data(5000,6000)

    # df1 = pd.read_csv('new_data.csv')
    #
    # # Read the second CSV file into a DataFrame
    # df2 = pd.read_csv('data3.csv')
    #
    # # Concatenate the two DataFrames along the rows (axis=0)
    # concatenated_df = pd.concat([df1, df2])
    #
    # # Write the concatenated DataFrame to a new CSV file
    # concatenated_df.to_csv('data_first3k.csv', index=False)