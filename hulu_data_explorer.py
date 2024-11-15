import pandas as pd
import streamlit as st
from transformers import pipeline
from typing import Optional
import matplotlib.pyplot as plt

# Load dataset with basic cleaning and data type enforcement
@st.cache_data
def load_data(file_path: str = "data.csv") -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        data['title'] = data['title'].fillna("Unknown Title").astype(str)
        data['genres'] = data['genres'].fillna("Unknown Genre").astype(str)
        data['releaseYear'] = data['releaseYear'].fillna(data['releaseYear'].median()).astype(int)
        data['imdbAverageRating'] = data['imdbAverageRating'].fillna(data['imdbAverageRating'].mean())
        data['availableCountries'] = data['availableCountries'].fillna("Unknown Country").astype(str)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Convert country codes to full country names
def get_country_name(code: str) -> str:
    country_dict = {
        'JP': 'Japan',
        'US': 'United States',
        'CA': 'Canada',
        # Add more mappings as needed
    }
    return ", ".join(country_dict.get(c.strip(), c) for c in code.split(','))

# Initialize sentiment analysis pipeline for genres
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Interpret the sentiment to user-friendly labels
def interpret_sentiment(sentiment: str) -> str:
    if sentiment == "POSITIVE":
        return "The genre tone suggests a generally enjoyable or light-hearted experience."
    elif sentiment == "NEGATIVE":
        return "The genre tone suggests a more intense, dramatic, or serious theme."
    else:
        return "The genre tone is neutral or mixed."

# Get sentiment score based on genres
def get_sentiment(genres: str, sentiment_analyzer) -> Optional[str]:
    try:
        result = sentiment_analyzer(genres[:512])
        sentiment = result[0]['label']
        return interpret_sentiment(sentiment)
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return None

# Function to display IMDb rating with stars
def get_rating_stars(rating):
    return "⭐" * int(round(rating))

# Function to create and save a compact IMDb Rating Distribution plot
def save_imdb_rating_distribution(data, filename="imdb_distribution.png"):
    # Create the figure and axes with a compact size
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot the histogram of IMDb ratings
    ax.hist(
        data['imdbAverageRating'].dropna(),
        bins=20,  # Increased bins for finer granularity
        color="#4CAF50",  # Green color for visual appeal
        edgecolor="white",
        alpha=0.85
    )
    
    # Set a descriptive title and axis labels
    ax.set_title("Distribution of IMDb Ratings in Hulu Dataset", fontsize=10, color="#333", weight="bold", pad=10)
    ax.set_xlabel("IMDb Rating (Out of 10)", fontsize=8, color="#555", labelpad=5)
    ax.set_ylabel("Number of Titles", fontsize=8, color="#555", labelpad=5)
    
    # Customize tick parameters for a cleaner look
    ax.tick_params(axis='both', which='major', labelsize=6, colors="#555")
    
    # Remove unnecessary spines for a modern design
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Hide y-axis ticks for a simplified appearance
    ax.yaxis.set_ticks([])

    # Adjust layout to ensure everything fits within the figure area
    fig.tight_layout()
    
    # Save the figure to the specified filename with a set DPI for clarity
    fig.savefig(filename, dpi=100)
    plt.close(fig)  # Close the figure to free memory


# Streamlit application with single-screen layout
def main():
    st.set_page_config(page_title="Hulu Data Explorer", layout="wide")
    st.title("📊 Hulu Data Explorer")
    st.markdown("Explore Hulu's content library with insights on **genres**, **ratings**, **availability**, and **genre-based tone analysis**.")

    # Load Data
    data = load_data("data.csv")
    if data.empty:
        st.write("Please ensure the file data.csv is in the current directory.")
        return

    # Sidebar filters
    with st.sidebar:
        st.header("Filter Options")
        genre_filter = st.multiselect("Select Genre(s):", data['genres'].str.split(',').explode().unique())
        min_year, max_year = int(data['releaseYear'].min()), int(data['releaseYear'].max())
        release_year_range = st.slider("Release Year Range:", min_year, max_year, (min_year, max_year))
        min_rating = st.slider("Minimum IMDb Rating:", 0.0, 10.0, 5.0)

        # Apply filters
        filtered_data = data[
            (data['genres'].apply(lambda x: any(genre in x for genre in genre_filter) if genre_filter else True)) &
            (data['releaseYear'].between(*release_year_range)) &
            (data['imdbAverageRating'] >= min_rating)
        ]

        title_selection = st.selectbox("Select a Title to Explore:", filtered_data['title'].unique())
    
    # Retrieve content details for the selected title
    selected_content = filtered_data[filtered_data['title'] == title_selection].iloc[0]
    
    # Translate country code(s) to full name(s)
    country_names = get_country_name(selected_content['availableCountries'])
    
    # IMDb link for the selected title
    imdb_url = f"https://www.imdb.com/title/{selected_content['imdbId']}" if not pd.isna(selected_content['imdbId']) else None
    imdb_link_html = f"<a href='{imdb_url}' target='_blank' style='text-decoration: none; color: #3498db;'>View on IMDb</a>" if imdb_url else "IMDb link not available"
    
    # Display content details in a card layout
    st.markdown(f"""
        <div style='border:1px solid #ddd; padding:15px; border-radius:10px; background-color:#f9f9f9;'>
            <h2 style='color:#2c3e50;'>{selected_content['title']}</h2>
            <p><strong>Type:</strong> {selected_content['type']} | <strong>Genres:</strong> {selected_content['genres']}</p>
            <p><strong>Release Year:</strong> {selected_content['releaseYear']} | 
            <strong>IMDb Rating:</strong> {selected_content['imdbAverageRating']:.1f} {get_rating_stars(selected_content['imdbAverageRating'])}
            ({int(selected_content['imdbNumVotes']) if not pd.isna(selected_content['imdbNumVotes']) else 'N/A'} votes)</p>
            <p><strong>Available in:</strong> {country_names}</p>
            <p>{imdb_link_html}</p>
        </div>
    """, unsafe_allow_html=True)

    # Compare rating to average
    avg_rating = data['imdbAverageRating'].mean()
    rating_diff = selected_content['imdbAverageRating'] - avg_rating
    comparison_text = "above average" if rating_diff > 0 else "below average"
    st.write(f"Rating is {abs(rating_diff):.1f} points {comparison_text} compared to other titles.")

    # Genre-Based Sentiment Analysis
    st.subheader("Genre-Based Tone Analysis")
    sentiment_analyzer = load_sentiment_analyzer()
    if st.button("Analyze Genre Tone"):
        genres = selected_content['genres']
        sentiment_interpretation = get_sentiment(genres, sentiment_analyzer)
        if sentiment_interpretation:
            st.markdown(f"<p style='color:#2c3e50; font-size:1.1em;'>{sentiment_interpretation}</p>", unsafe_allow_html=True)

    # Similar Titles Suggestions
    similar_titles = data[(data['genres'] == selected_content['genres']) & 
                          (data['imdbAverageRating'] >= selected_content['imdbAverageRating'] - 0.5) &
                          (data['title'] != selected_content['title'])]
    
    if not similar_titles.empty:
        st.subheader("You Might Also Like:")
        for _, row in similar_titles.head(3).iterrows():  # Limit to 3 suggestions
            st.markdown(f"- {row['title']} ({int(row['releaseYear'])}) - IMDb Rating: {row['imdbAverageRating']}")

    # IMDb Rating Distribution with compact size
    st.subheader("IMDb Rating Distribution")
    save_imdb_rating_distribution(data, filename="imdb_distribution.png")
    st.image("imdb_distribution.png", width=500)  # Display image with controlled width

# Run the application
if __name__ == "__main__":
    main()