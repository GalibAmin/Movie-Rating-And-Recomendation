from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests

app = Flask(__name__)


#LOAD ALL MODELS & DATA

# Original text dataset (for KNN + display)
with open("dataset.pkl", "rb") as f:
    Rdataset_text = pickle.load(f)

# Encoded numeric dataset (for rating prediction)
with open("encoded_dataset.pkl", "rb") as f:
    Rdataset_numeric = pickle.load(f)

# KNN similarity model
with open("model.pkl", "rb") as f:
    knn = pickle.load(f)

# TF-IDF Vectorizer
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Random Forest rating model
with open("rfrating_model.pkl", "rb") as f:
    rating_model = pickle.load(f)

# Columns for RandomForest input
with open("features_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Encoders (loaded but not used directly)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# TF-IDF matrix for all movies
tfidf_matrix = tfidf.transform(Rdataset_text["tags"])

# lowercase for searching
Rdataset_text["title_lower"] = Rdataset_text["original_title"].str.lower()


# TMDB POSTER FETCH FUNCTION
TMDB_API_KEY = "9fee797989b5d275957d74ca506e184f"

def get_poster(movie_title):
    """Fetch poster URL from TMDB API"""
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url).json()

        if response["results"]:
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
    except:
        pass

    # fallback local placeholder
    return "/static/posters/default.jpeg"

# ROUTES
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.form["movie"].lower().strip()

    # if movie not found
    if user_input not in Rdataset_text["title_lower"].values:
        return render_template("index.html", recommendations=["Movie not found!"])

    # get index
    movie_index = Rdataset_text[Rdataset_text["title_lower"] == user_input].index[0]

    # get 5 nearest neighbours
    distances, indices = knn.kneighbors(tfidf_matrix[movie_index], n_neighbors=6)
    rec_indices = indices[0][1:]  # skip input movie

    recommendations = []

    for idx in rec_indices:
        row_text = Rdataset_text.iloc[idx]
        row_num = Rdataset_numeric.iloc[idx]

        # build feature vector
        feature_values = []
        for col in feature_cols:
            value = row_num[col] if col in row_num else 0
            feature_values.append(value)

        # predicted rating
        predicted_rating = rating_model.predict([feature_values])[0]

        # final data
        recommendations.append({
            "title": row_text["original_title"],
            "rating": round(predicted_rating, 2),
            "poster": get_poster(row_text["original_title"])
        })

    return render_template("index.html", recommendations=recommendations)


# RUN APP
if __name__ == "__main__":
    app.run(debug=True)
