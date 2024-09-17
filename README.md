# Movie Recommender System

## Project Overview

This project builds a movie recommender system using machine learning. It recommends movies based on their similarity to a given movie title and includes deployment via Streamlit.

## Repository Structure

- **model**: Jupyter notebook for data preprocessing, feature extraction, and model building.
- **deployment**: Streamlit app scripts (`app.py` and `app(with_images).py`).
- **processed_data**: Contains processed data files.

## Dataset

The dataset is sourced from TMDB and contains 5000 movies across two CSV files:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## Deployment

To run the Streamlit app:

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run deployment/app.py
   ```
   or
   ```bash
   streamlit run deployment/app(with_images).py
   ```

3. Open `http://localhost:8501` in your browser to use the app.

## Processed Data

- `merged_movies_credits.csv`: Processed movie data.
- `movie_dict.pkl`: Serialized movie data dictionary.
- `similarity.pkl`: Serialized similarity matrix.

## Demo Video

[Movie Recommender System Demo](Demo/Movie%20Reommender%20system%20demo.mp4)
