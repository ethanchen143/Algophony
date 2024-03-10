from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import numpy as np

def getGenre(filename,num):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    model = TensorflowPredict2D(graphFilename="essentia graphfiles/MTG Genre Classifier.pb")
    predictions = model(embeddings)

    data = {
        "classes": [
            "jazz",
            "70s",
            "80s",
            "hiphop",
            "jazz",
            "alt/indie",
            "rock",
            "ambient",
            "ambient",
            "blues",
            "blues",
            "jazz",
            "hiphop",
            "world",
            "world",
            "ambient",
            "symphony",
            "symphony",
            "rock",
            "electronic",
            "jazz",
            "country/folk",
            "country/folk",
            "ambient",
            "ambient",
            "house",
            "80s",
            "electronic",
            "electronic",
            "electronic",
            "electronic",
            "ambient",
            "electronic",
            "electronic",
            "electronic",
            "pop",
            "world",
            "house",
            "experimental",
            "country/folk",
            "funk",
            "jazz",
            "hiphop",
            "rock",
            "rock",
            "rock",
            "hiphop",
            "house",
            "experimental",
            "jazz",
            "alt/indie",
            "electronic",
            "pop",
            "rock",
            "jazz",
            "jazz",
            "pop",
            "jazz",
            "symphony",
            "rock",
            "experimental",
            "jazz",
            "rock",
            "symphony",
            "pop",
            "country/folk",
            "rock",
            "rock",
            "rock",
            "experimental",
            "rock",
            "hiphop",
            "world",
            "rnb",
            "rock",
            "rock",
            "pop",
            "soul",
            "symphony",
            "jazz",
            "symphony",
            "80s",
            "electronic",
            "electronic",
            "pop",
            "world",
            "world"
        ]
    }
    genre_labels = data["classes"]
    # calculate average probability of all frames for each genre
    average_predictions = np.mean(predictions, axis=0)
    top = np.argsort(average_predictions)[-num:]
    top = top[::-1]
    top_genres = [genre_labels[idx] for idx in top]
    return list(set(top_genres))

def getTheme(filename,num):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Moodtheme model.pb")
    predictions = model(embeddings)
    data = {
        "classes": [
            "action",
            "adventure",
            "advertising",
            "background",
            "ballad",
            "calm",
            "children",
            "christmas",
            "commercial",
            "cool",
            "corporate",
            "dark",
            "deep",
            "documentary",
            "drama",
            "dramatic",
            "dream",
            "emotional",
            "energetic",
            "epic",
            "fast",
            "film",
            "fun",
            "funny",
            "game",
            "groovy",
            "happy",
            "heavy",
            "holiday",
            "hopeful",
            "inspiring",
            "love",
            "meditative",
            "melancholic",
            "melodic",
            "motivational",
            "movie",
            "nature",
            "party",
            "positive",
            "powerful",
            "relaxing",
            "retro",
            "romantic",
            "sad",
            "sexy",
            "slow",
            "soft",
            "soundscape",
            "space",
            "sport",
            "summer",
            "trailer",
            "travel",
            "upbeat",
            "uplifting"
        ]
    }
    theme_labels = data["classes"]
    average_predictions = np.mean(predictions, axis=0)
    top = np.argsort(average_predictions)[-num:]
    top = top[::-1]
    top_themes = [theme_labels[idx] for idx in top]
    if 'bass' in top_themes and ('bass' not in filename or 'Bass' not in filename):
        top_themes.remove('bass')
    return top_themes

from essentia.standard import TensorflowPredictMusiCNN
def getMoodGroup(filename):
    # try catch
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictMusiCNN(graphFilename="essentia graphfiles/MusicNN model.pb", output="model/dense/BiasAdd")
    embeddings = embedding_model(audio)

    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Moods MusicNN Model.pb", input="serving_default_model_Placeholder",
                                output="PartitionedCall")
    try:
        predictions = model(embeddings)
    except TypeError:
        print("Oops! TypeError getting mood group")
        return []

    data = {
        "classes": [
            "passionate, rousing, confident, boisterous, rowdy",
            "rollicking, cheerful, fun, sweet, amiable/good natured",
            "literate, poignant, wistful, bittersweet, autumnal, brooding",
            "humorous, silly, campy, quirky, whimsical, witty, wry",
            "aggressive, fiery, tense/anxious, intense, volatile, visceral"
        ]
    }
    theme_labels = data["classes"]
    array_labels = []
    for string in theme_labels:
        words = []
        for word_group in string.split(','):
            words += [word.strip() for word in word_group.split('/')]
        array_labels.append(words)
    theme_labels = array_labels
    average_predictions = np.mean(predictions, axis=0)
    top = np.argsort(average_predictions)[-1]
    return theme_labels[top]
def getPopularity(filename):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Approachability Regression Discogs Effnet.pb", output="model/Identity")
    predictions = model(embeddings)
    # return the mean of each frame's popularity and penalize a large STD
    return np.mean(predictions) - np.std(predictions)

def getEngagement(filename):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Engagement Regression Discogs Effnet Model.pb", output="model/Identity")
    predictions = model(embeddings)
    # return the mean of each frame's popularity and penalize a large STD
    return np.mean(predictions) - np.std(predictions)

def getAggresive(filename):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)
    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Aggressive Mood Model.pb", output="model/Softmax")
    predictions = model(embeddings)

    # return the average confidence that it is aggressive or non
    agg = np.mean([i[0] for i in predictions])
    non = np.mean([i[1] for i in predictions])
    return (agg,non)

def getHappy(filename):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)
    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Happy Mood Model.pb", output="model/Softmax")
    predictions = model(embeddings)

    # return the average confidence that it is aggressive or non
    hap = np.mean([i[0] for i in predictions])
    non = np.mean([i[1] for i in predictions])
    return (hap,non)

def getRelaxed(filename):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)
    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Relaxed Mood Model.pb", output="model/Softmax")
    predictions = model(embeddings)
    # return the average confidence that it is aggressive or non
    rel = np.mean([i[0] for i in predictions])
    non = np.mean([i[1] for i in predictions])
    return (rel,non)

def getSad(filename):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)
    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Sad Mood Model.pb", output="model/Softmax")
    predictions = model(embeddings)

    # return the average confidence that it is aggressive or non
    sad = np.mean([i[0] for i in predictions])
    non = np.mean([i[1] for i in predictions])
    return (sad,non)

#bright/dark
def getTimbre(filename):
    audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
                                                     output="PartitionedCall:1")
    embeddings = embedding_model(audio)
    model = TensorflowPredict2D(graphFilename="essentia graphfiles/Timbre Discogs Model.pb", output="model/Softmax")
    predictions = model(embeddings)
    bright = np.mean([i[0] for i in predictions])
    dark = np.mean([i[1] for i in predictions])
    return (bright,dark)

# def getInst(filename,num):
#     audio = MonoLoader(filename=filename, sampleRate=16000, resampleQuality=4)()
#     embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="essentia graphfiles/Discogs Effnet BS64 Model.pb",
#                                                      output="PartitionedCall:1")
#     embeddings = embedding_model(audio)
#
#     model = TensorflowPredict2D(graphFilename="essentia graphfiles/Instrument model.pb")
#     predictions = model(embeddings)
#     data = {
#         "classes": [
#             "reed",
#             "guitar",
#             "guitar",
#             "bass",
#             "beat",
#             "bell",
#             "bongo",
#             "brass",
#             "cello",
#             "clarinet",
#             "classicalguitar",
#             "computer",
#             "doublebass",
#             "drummachine",
#             "drums",
#             "electricguitar",
#             "electricpiano",
#             "flute",
#             "guitar",
#             "harmonica",
#             "harp",
#             "horn",
#             "keyboard",
#             "oboe",
#             "orchestra",
#             "organ",
#             "pad",
#             "percussion",
#             "piano",
#             "pipeorgan",
#             "rhodes",
#             "sampler",
#             "saxophone",
#             "strings",
#             "synthesizer",
#             "trombone",
#             "trumpet",
#             "viola",
#             "violin",
#             "voice"
#         ]
#     }
#     genre_labels = data["classes"]
#     average_predictions = np.mean(predictions, axis=0)
#     top = np.argsort(average_predictions)[-num:]
#     top = top[::-1]
#     top_insts = [genre_labels[idx] for idx in top]
#     return top_insts