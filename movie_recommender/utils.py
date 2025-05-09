import pandas as pd
import numpy as np

def preprocess_genres(genres_series):
    all_genres = sorted(set(genre for sublist in genres_series for genre in sublist))
    genre_to_index = {genre: idx for idx, genre in enumerate(all_genres)}

    matrix = np.zeros((len(genres_series), len(all_genres)))
    for i, genres in enumerate(genres_series):
        for genre in genres:
            matrix[i][genre_to_index[genre]] = 1
    return matrix, all_genres

# Mood to genre preference (soft mapping)
mood_genre_pref = {
    "happy": ["Comedy", "Family", "Adventure"],
    "sad": ["Drama", "Romance"],
    "excited": ["Action", "Thriller"],
    "scared": ["Horror", "Mystery"],
    "thoughtful": ["Documentary", "Drama"],
    "fantasy": ["Fantasy", "Science Fiction"]
}

def get_cluster_for_mood(mood, kmeans_model, genre_names):
    genres = mood_genre_pref.get(mood, [])
    clusters = set()

    for genre in genres:
        vec = np.zeros((1, len(genre_names)))
        if genre in genre_names:
            vec[0][genre_names.index(genre)] = 1
            cluster = kmeans_model.predict(vec)[0]
            clusters.add(cluster)
    return list(clusters)
