# Introduction

## Overview

Algophony is an AI-powered platform designed for music generation. It enables users to generate individual parts of a song using natural language descriptions and deep learning models, including CVAE and HifiGAN, for a streamlined music production experience.
For more details, the whole proposal is [linked](https://drive.google.com/file/d/1NetlLv6Nm9rLHcOTTOqlBCKjbjd0z5CW/view?usp=drive_link).

# Technical Architecture
## Front End
The front end was built using HTML and Javascript. Basic HTML was used to create a UI that would take a text input and send it to a backend java script server. The server runs a local host that waits for a text input to run a python script. It then takes the audio file generated from the modle and outputs it back on the front end website allowing the user to listen to the final product.
![image](https://github.com/CS222-UIUC-SP24/group-project-51/assets/60461279/a719f1ca-cdcc-4fde-a441-135924a5e0bf)
![image](https://github.com/CS222-UIUC-SP24/group-project-51/assets/60461279/947f5bb1-d8df-4106-a740-edb7337d5c11)

## Back End
The back end hosts our CVAE generative model. It handles user requests by calling the inference function of the model with text input as its argument. The model generates a spectrogram and converts it into high-quality audio with a pre-trained HifiGAN model.
![CVAE](https://github.com/CS222-UIUC-SP24/group-project-51/assets/92761562/dd39dc84-b23d-4d24-8a56-d7dfd5521573)


# Developers
- **Ethan Chen**: Curated and preprocessed the audio/spectrogram dataset, designed and trained the CVAE model, designed backend components
- **Knud Andersen**: Worked and researched backend: designed, and trained initial VAE model, then added conditional latent space aspect by implementing CVAE model
- **Matthew Shang**: Created front end UI, set up JS server deployment and connected front and backend components
- **Ivan**: Assisted in the development of the front end. Developed potential features.


# Environment Setup

## Required Downloads and Packages
Download Algophony V1.0 File

[Node.JS](https://nodejs.org/en)

Python packages require Numpy, Pytorch, transformers, soundfile
```
pip install numpy
pip install torch
pip install transformers
pip install soundfile
```

# How to Run
Once all pre-requisite packages are downloaded run server.js file

After server is running, open index.html and enter text to begin music generation

Program will automatically begin to generate music and website will begin to play generated audio

Enjoy!



# Additional Information

## Features
Natural Language Input: Describe what you need (e.g., "calm electric guitar") and get high-quality audio outputs.
Professional Sounds: Access ~150k audio samples for high-grade loops, sourced from top sample sites.
On-site Editing: Edit and mix loops directly on Algophony or export to use in your DAW.
AI-Driven Generation: Utilizes CVAE for generating new spectrogram and Hifi-GAN for converting spectrogram back to high-fidelity audio.
Data Processing

## Generation Model
Our CVAE model, enhanced with multi-class label encoding, is designed for nuanced sound generation across various categories. The model is trained to minimize a combined loss function, balancing reconstruction error and KL-Divergence, optimized via the Adam algorithm.

## Audio Conversion
Post-generation, the Hifi-GAN model is employed for high-quality audio synthesis, ensuring the final output is clear and true to the original spectral representation.

## Data

### Data Features
Name: Title of the sample.

ID: A unique identifier.

Spectrogram: Representation of the audio spectrum.

Instrument: The instrument used in the sample.

Genre: Musical genres the sample fits into.

Mood: The mood or emotional qualities conveyed by the sample.

Theme: Key themes or concepts associated with the sample.

Happy, Sad, Aggressive, Relaxed: Emotional intensity scores.

Popularity: A measure of how popular the sample is.

Engagement: A measure of how engaging the sample is to listeners. 

