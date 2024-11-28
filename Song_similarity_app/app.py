from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load embeddings from .npy file
embeddings = np.load("lyric_embeddings2.npy")

# Load other data associated with the embeddings
df4 = pd.read_csv(
    "./vectorized_for_app.csv"
)  # Assuming you have a corresponding CSV file with the lyrics and other metadata


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_lyrics = request.form["lyrics"]
        similar_songs = find_similar_songs(input_lyrics, embeddings, df4)
        return render_template("results.html", similar_songs=similar_songs)
    return render_template("index.html")


def find_similar_songs(input_lyrics, embeddings, data, top_n=5):
    model = SentenceTransformer("all-mpnet-base-v2")
    input_embedding = model.encode(input_lyrics)
    similarities = cosine_similarity([input_embedding], embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    top_songs = data.iloc[top_indices]
    top_scores = similarities[0][top_indices]
    results = [
        (row.Artist, row.Title, score)
        for row, score in zip(top_songs.itertuples(), top_scores)
    ]
    return results


if __name__ == "__main__":
    app.run(debug=True)
