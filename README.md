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

### Running code in Anaconda Prompt

1. **Open Anaconda prompt and load your project.**
2. **Open the respective `.py` file (app.py or app(with_images).py).**
3. **Run the file using 'streamlit run app.py'.**

## Processed Data

The `processed_data` directory contains the following files:

- `merged_movies_credits.csv`: The merged and processed movie data.
- `movie_dict.pkl`: Serialized dictionary of movie data.
- `similarity.pkl`: Serialized similarity matrix.


## Demo Video
https://github.com/Akmech001/Movie-Recommender-System/blob/Data-Science-Capstone-Porject/Demo/Movie%20Reommender%20system%20demo.mp4

## Conclusion

This project demonstrates the development of a movie recommender system using machine learning techniques. It includes data preprocessing, feature extraction, model building, and deployment using Streamlit. The provided scripts and instructions allow for easy setup and usage of the recommender system.
