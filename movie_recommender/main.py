import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import KMeans
from sentiment.sentiment_analysis import get_sentiment
from models.ga_optimizer import genetic_algorithm
from utils import preprocess_genres, get_cluster_for_mood

# Load datasets
movies_df = pd.read_csv('data/tmdb_5000_movies.csv')
credits_df = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge datasets
movies_df = movies_df.merge(credits_df, on='title')

# Preprocessing
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')
movies_df['description'] = movies_df['overview'] + ' ' + movies_df['keywords']
movies_df['genres'] = movies_df['genres'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)])

# One-hot encode genres and cluster
genre_matrix, genre_names = preprocess_genres(movies_df['genres'])
kmeans = KMeans(n_clusters=5, random_state=42)
movies_df['genre_cluster'] = kmeans.fit_predict(genre_matrix)

# TF-IDF similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

def recommend(title, mood=None, top_n=5):
    idx = indices.get(title)
    if idx is None:
        return "Movie not found!"

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:]]

    recommended = movies_df.iloc[movie_indices]

    if mood:
        mood_clusters = get_cluster_for_mood(mood, kmeans, genre_names)
        recommended = recommended[recommended['genre_cluster'].isin(mood_clusters)]

    recommended = recommended[['title', 'overview']].head(top_n)

    results = []
    for _, row in recommended.iterrows():
        sentiment = get_sentiment(row['overview'])
        results.append({
            'title': row['title'],
            'overview': row['overview'][:200] + '...',
            'sentiment': sentiment
        })

    return recommended

def run_cli():
    movie_name = input("Enter a movie name: ")
    user_mood = input("Enter your mood (e.g., happy, sad, excited, scared, thoughtful, fantasy): ").lower()
    results = recommend(movie_name, user_mood)

    if isinstance(results, str):
        print(results)
    else:
        print(f"\nTop recommendations for '{movie_name}' with mood '{user_mood}':\n")
        for row in results:
            print(f"{row['title']} — Sentiment: {row['sentiment']}")
            print(f"  Overview: {row['overview']}\n")

    print("\nOptimized Movie Recommendations based on GA:\n")
    result = genetic_algorithm()
    for idx, row in result.iterrows():
        sentiment = get_sentiment(row['title'])
        runtime_mins = round(row['runtime'] * 178)
        print(f"{row['title']} — Sentiment: {sentiment}")
        print(f"  Popularity: {row['popularity']:.2f}, Rating: {row['vote_average']:.2f}, Runtime: {runtime_mins} mins\n")

if __name__ == "__main__":
    run_cli()
