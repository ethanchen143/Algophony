# Overview
Algophony is an AI-powered platform designed for music generation. It enables users to generate individual parts of a song using natural language descriptions and deep learning models, including CVAE and HifiGAN, for a streamlined music production experience.

# Features
Natural Language Input: Describe what you need (e.g., "calm electric guitar") and get high-quality audio outputs.
Professional Sounds: Access ~30k audio samples for high-grade loops, sourced from top sample sites.
On-site Editing: Edit and mix loops directly on Algophony or export to use in your DAW.
AI-Driven Generation: Utilizes CVAE for generating new spectrograms and Hifi-GAN for converting spectrograms back to high-fidelity audio.
Data Processing

# Generation Model
Our CVAE model, enhanced with multi-class label encoding, is designed for nuanced sound generation across various categories. The model is trained to minimize a combined loss function, balancing reconstruction error and KL-Divergence, optimized via the ADAM algorithm.

# Audio Conversion
Post-generation, the Hifi-GAN model is employed for high-quality audio synthesis, ensuring the final output is clear and true to the original spectral representation.

# Data
[Google Drive Link](https://drive.google.com/drive/folders/1YGPTS1tKCzmGRIwc5OH8ZK4VTq-P13it?usp=drive_link)

# Data Features
Name: Title of the sample.
ID: A unique identifier.
Spectrogram: Visual representation of the audio spectrum.
Instrument: The instrument used in the sample.
Genre: Musical genres the sample fits into.
Mood: The mood or emotional qualities conveyed by the sample.
Theme: Key themes or concepts associated with the sample.
Happy, Sad, Aggressive, Relaxed: Emotional intensity scores.
Popularity: A measure of how popular the sample is.
Engagement: A measure of how engaging the sample is to listeners.
