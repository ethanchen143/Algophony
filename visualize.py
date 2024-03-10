import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = './data_first3k.csv'
df = pd.read_csv(dataset_path)

instrument_counts = df['Instrument'].value_counts()

# Convert string representations of lists to actual lists and concatenate
genres = df['Genre'].apply(eval).sum()
genre_counts = pd.Series(genres).value_counts()
genre_counts = genre_counts[genre_counts>200]

def plot_donut_chart(data, title, ax):
    # Pie Plot
    data.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=140, pctdistance=0.85, center=(0, 0))

    # Draw a circle at the center
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    ax.set_title(title)

fig, axs = plt.subplots(2, 1, figsize=(7, 14))

# Plot each category
plot_donut_chart(instrument_counts, 'Instrument Distribution', axs[0])
plot_donut_chart(genre_counts, 'Genre Distribution', axs[1])

plt.tight_layout()
plt.show()
