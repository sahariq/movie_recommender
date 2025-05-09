import streamlit as st
from main import recommend
from models.ga_optimizer import genetic_algorithm
from sentiment.sentiment_analysis import get_sentiment

# Title
st.title("🎬 Movie Recommendation System")

# Movie Input
movie_title = st.text_input("Enter a movie title:")

# Recommend Button
if st.button("Get Recommendations"):
    if movie_title:
        st.subheader(f"Top Recommendations for '{movie_title}'")
        result = recommend(movie_title, top_n=5)
        
        if isinstance(result, str):
            st.warning(result)
        else:
            for _, row in result.iterrows():
                sentiment = get_sentiment(row['overview'])
                st.markdown(f"**{row['title']}** — *{sentiment}*")
                st.write(row['overview'][:300] + '...')

        st.subheader("🎯 Optimized GA Recommendations")
        ga_results = genetic_algorithm()

        for _, row in ga_results.iterrows():
            sentiment = get_sentiment(row['title'])
            runtime = round(row['runtime'] * 178)
            st.markdown(f"**{row['title']}** — *{sentiment}*")
            st.write(f"Rating: {row['vote_average']:.2f}, Popularity: {row['popularity']:.2f}, Runtime: {runtime} mins")
    else:
        st.error("Please enter a movie title.")
