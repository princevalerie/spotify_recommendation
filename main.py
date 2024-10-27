import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components  # To embed HTML
from datetime import datetime

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('kpop_tracks.csv')
    df = df.drop_duplicates(subset=['Track Name', 'Artists'])
    return df

# Calculate the difference in months from the current date
def calculate_month_difference(df):
    # Convert 'Release Date' to datetime if it's not already
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')  # Convert to datetime
    df['Month Difference'] = (datetime.now() - df['Release Date']) / pd.Timedelta(days=30.44)  # Average days in a month
    df['Month Difference'] = df['Month Difference'].fillna(df['Month Difference'].mean())  # Fill NaN with mean
    return df

# Preprocess the data for the KNN model
def preprocess_data(df):
    features = [
        "Danceability", "Energy", "Key", "Loudness", "Mode", "Speechiness",
        "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo"
    ]

    df = df.copy()
    df[features] = df[features].fillna(df[features].mean())  # Fill missing values with column mean
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])  # Scale features for normalization
    return df, features

# Function to recommend tracks
def recommend_tracks(df, track_name, features, num_recommendations):
    try:
        # Set up the KNN model
        knn_model = NearestNeighbors(n_neighbors=num_recommendations + 1, algorithm='brute')
        knn_model.fit(df[features])

        # Find the index of the selected track
        # track_index = df[df['Track Name'].str.lower() == track_name.lower()].index[0]
        track_index = df[df['Track Name'] == track_name].index[0]
        
        track_features = df.iloc[track_index][features].values.reshape(1, -1)

        # Get nearest neighbors
        distances, indices = knn_model.kneighbors(track_features)
        recommended_indices = indices.flatten()[1:]  # Exclude the first index (input track)

        # Get recommended tracks
        recommendations = df.iloc[recommended_indices]
        return recommendations
    except IndexError:
        st.error(f"Track '{track_name}' not found. Please select an existing track.")
        return pd.DataFrame(columns=['Track Name', 'Track ID', 'Artists'])

# Main app
def main():
    st.title('Spotify K-Pop Song Recommender')

    # Load the data
    kpop_df = load_data()

    # Calculate month difference
    kpop_df = calculate_month_difference(kpop_df)

    # Preprocess the data
    kpop_df, features = preprocess_data(kpop_df)

    # Input for the track name (selectbox with typing capability)
    track_name = st.selectbox(
        "Enter a song name you like to get recommendations:",
        options=kpop_df['Track Name'].unique(),
        format_func=lambda x: x if x in kpop_df['Track Name'].unique() else f"Add '{x}'"
    )

    # Number of recommendations
    num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

    # Button to get recommendations
    if st.button("Get Recommendations"):
        recommendations = recommend_tracks(kpop_df, track_name, features, num_recommendations)

        st.write("### Recommendation Song")
        
        for i, (index, row) in enumerate(recommendations.iterrows()):
            # Embed Spotify preview using Track ID
            track_id = row['Track ID']
            spotify_embed_html = f"""
            <iframe src="https://open.spotify.com/embed/track/{track_id}" 
                    width="100%" height="500" frameborder="0" 
                    allowtransparency="true" allow="encrypted-media"></iframe>
            """
            components.html(spotify_embed_html, height=500)  # Set height to ensure it doesn't get cut off

if __name__ == "__main__":
    main()
