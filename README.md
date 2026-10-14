# 🎬 Movie Recommendation System

A content-based movie recommender built in a Jupyter Notebook on the TMDB 5000 Movies dataset. Give it a movie title and it returns the 5 most similar movies, ranked by cosine similarity over each film's overview, genres, keywords, top cast and director.

## Features

- Merges the TMDB movies and credits datasets and keeps only the useful columns
- Parses JSON-style columns (genres, keywords, cast, crew) into clean lists
- Extracts the top 3 cast members and the director for each movie
- Builds a single `tags` text per movie, then lowercases and stems it with the Porter Stemmer
- Vectorizes text with `CountVectorizer` (top 5,000 features, English stop words removed)
- Computes a 4806 x 4806 cosine similarity matrix
- `recommend(movie)` prints the top 5 most similar movies

## Project Structure

```
movie-recommendations-system-/
├── movie_recomadation_project.ipynb   # Preprocessing, vectorization, similarity, recommend()
├── tmdb_5000_movies.csv               # TMDB 5000 movies dataset
└── README.md
```

> The notebook also reads `tmdb_5000_credits.csv` (cast and crew). Download it with the movies file from the [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place it in the project root.

## Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Environment:** Jupyter Notebook
- **Dataset:** TMDB 5000 Movies and Credits

## How It Works

1. **Load and merge**: read the movies and credits CSVs and merge them on `title`.
2. **Select features**: keep `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`; drop null rows.
3. **Parse**: convert JSON strings to lists with `ast.literal_eval`; keep the top 3 cast members and the director only.
4. **Clean**: remove spaces inside names (e.g. `Sam Worthington` becomes `SamWorthington`) so each person or genre stays one token.
5. **Create tags**: concatenate overview, genres, keywords, cast and director, then lowercase and apply Porter stemming.
6. **Vectorize**: `CountVectorizer(max_features=5000, stop_words='english')`.
7. **Similarity**: `cosine_similarity` across all movie vectors.
8. **Recommend**: look up the movie's index, sort by similarity, and return the top 5 (excluding itself).

## Getting Started

```bash
git clone https://github.com/pulkitpareek1915/movie-recommendations-system-.git
cd movie-recommendations-system-
pip install pandas numpy scikit-learn nltk jupyter
jupyter notebook movie_recomadation_project.ipynb
```

Run all cells, then:

```python
recommend('John Carter')
```

## Example Output

```
Input: John Carter
Star Trek: Insurrection
Mission to Mars
Captain America: The First Avenger
Escape from Planet Earth
Ghosts of Mars
```

## Future Improvements

- Hybrid model combining content-based and collaborative filtering
- Streamlit / Flask web app with posters from the TMDB API
- Switch to TF-IDF or embeddings for better semantic matching
- Handle duplicate titles created by the title merge (4,809 rows from 4,803 movies)
- Save the similarity matrix with pickle for faster loading

## Author

**Pulkit Pareek**
GitHub: [@pulkitpareek1915](https://github.com/pulkitpareek1915)
