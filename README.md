# Movie-Recommender-System

## Project Overview

This project aims to create a movie recommender system using machine learning techniques. The system recommends movies based on their similarity to a given movie title. The project includes data preprocessing, feature extraction, and deployment using Streamlit.

## Repository Structure

The repository is organized into the following directories:

1. **model**: Contains the Jupyter notebook used for data preprocessing, feature extraction, and model building.
2. **deployment**: Contains the deployment scripts for the recommender system.
   - `app.py`: A Streamlit app that displays movie recommendations by title.
   - `app(with_images).py`: A Streamlit app that displays movie recommendations with titles and images.
3. **processed_data**: Contains processed data files generated from the model.

## Data Preprocessing and Model Building

### Dataset

The dataset used in this project is sourced from TMDB and contains information on 5000 movies. It consists of two CSV files:
- `tmdb_5000_movies.csv`: Contains details about movies.
- `tmdb_5000_credits.csv`: Contains cast and crew information.

### Jupyter Notebook

The Jupyter notebook file in the `model` directory performs the following steps:

1. **Data Loading**: Reads the movies and credits datasets from CSV files.
   ```python
   movies=pd.read_csv('tmdb_5000_movies.csv')
   credits=pd.read_csv('tmdb_5000_credits.csv')
   movies=movies.merge(credits, on='title')
   ```

2. **Data Cleaning**: Drops missing values and selects relevant columns.
   ```python
   movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
   movies.dropna(inplace=True)
   ```

3. **Feature Extraction**: Extracts relevant features from JSON columns.
   ```python
   def convert(obj):
       L = []
       for i in ast.literal_eval(obj):
           L.append(i['name'])
       return L
   movies['genres'] = movies['genres'].apply(convert)
   movies['keywords'] = movies['keywords'].apply(convert)
   movies['cast'] = movies['cast'].apply(convert3)
   movies['crew'] = movies['crew'].apply(fetch_director)
   ```

4. **Text Processing**: Combines and processes text features for vectorization.
   ```python
   movies['overview'] = movies['overview'].apply(lambda x: x.split())
   movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
   new_df = movies[['movie_id','title','tags']]
   new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
   new_df['tags'] = new_df['tags'].apply(stem)
   ```

5. **Vectorization and Similarity Calculation**: Converts text to vectors and calculates cosine similarity.
   ```python
   cv = CountVectorizer(max_features=5000, stop_words='english')
   vectors = cv.fit_transform(new_df['tags']).toarray()
   similarity = cosine_similarity(vectors)
   ```

6. **Model Serialization**: Saves the processed data and similarity matrix as pickle files for deployment.
   ```python
   import pickle
   pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
   pickle.dump(similarity, open('similarity.pkl', 'wb'))
   ```

### Running the Jupyter Notebook

To run the Jupyter notebook, execute the following command in your terminal:

```bash
jupyter notebook model/movie_recommender_system.ipynb
```

## Deployment

### Streamlit Applications

Two Streamlit applications are provided for deployment:

1. **app.py**: Recommends movies based on a given title.
2. **app(with_images).py**: Recommends movies based on a given title and displays movie posters.

#### app.py

```python
import streamlit as st
import pickle
import pandas as pd

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))
st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select a movie to get recommendations',
    movies['title'].values
)

if st.button('Recommend'):
    names = recommend(selected_movie_name)
    for name in names:
        st.write(name)
```

#### app(with_images).py

```python
import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=ed08ba3dd40a4fe896a47b63daf0be95&language=en-US')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))
st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Select a movie to get recommendations',
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.header(names[0])
        st.image(posters[0])
    with col2:
        st.header(names[1])
        st.image(posters[1])
    with col3:
        st.header(names[2])
        st.image(posters[2])
    with col4:
        st.header(names[3])
        st.image(posters[3])
    with col5:
        st.header(names[4])
        st.image(posters[4])
```

### Running the Streamlit App

To run the Streamlit app, follow these steps:

1. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run deployment/app.py
   ```
   or
   ```bash
   streamlit run deployment/app(with_images).py
   ```

3. **Open your browser and go to `http://localhost:8501` to use the recommender system.**

### Working in DataSpell

DataSpell is an IDE designed for data science. To run the code in DataSpell:

1. **Open DataSpell and load your project.**
2. **Open the respective `.py` file (app.py or app(with_images).py).**
3. **Run the file using the built-in run configuration.**

## Processed Data

The `processed_data` directory contains the following files:

- `merged_movies_credits.csv`: The merged and processed movie data.
- `movie_dict.pkl`: Serialized dictionary of movie data.
- `similarity.pkl`: Serialized similarity matrix.

## Conclusion

This project demonstrates the development of a movie recommender system using machine learning techniques. It includes data preprocessing, feature extraction, model building, and deployment using Streamlit. The provided scripts and instructions allow for easy setup and usage of the recommender system.
