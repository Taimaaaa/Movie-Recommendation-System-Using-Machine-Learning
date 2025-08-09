from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from rapidfuzz import process, fuzz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

data_set_size = 100000
test_split = 0.2
k_neighbors = 7


app = Flask(__name__)
app.secret_key = "super_secret_key"

MOVIES = pd.read_csv("movies.csv").head(data_set_size)
processed_data = {}


@app.route("/", methods=["GET", "POST"])
def home():
    # Initialize session if not already set
    if "liked_movies" not in session:
        session["liked_movies"] = []

    if request.method == "POST":
        if "add_movie" in request.form:  # Handle adding a movie
            movie_name = request.form["movie_name"].strip().lower()

            # Handle NaN values in 'title' column
            MOVIES["title"] = MOVIES["title"].fillna(
                ""
            )  # Replace NaN with empty string

            # Check for exact matches first
            exact_match = MOVIES[MOVIES["title"].str.lower() == movie_name]
            if not exact_match.empty:
                matched_movie = exact_match.iloc[0]
            else:
                # Perform fuzzy matching to find the best match
                movie_choices = MOVIES["title"].tolist()
                best_match = process.extractOne(
                    movie_name, movie_choices, scorer=fuzz.ratio
                )

                if (
                    best_match and best_match[1] > 70
                ):  # Only consider matches with score > 70
                    matched_movie = MOVIES[MOVIES["title"] == best_match[0]].iloc[0]
                else:
                    matched_movie = None

            if matched_movie is not None:
                movie_id = int(matched_movie["id"])  # Convert to native int
                if movie_id not in session["liked_movies"]:
                    if len(session["liked_movies"]) < 10:  # Limit to 10 movies
                        session["liked_movies"].append(movie_id)
                        session.modified = True

        elif "delete_movie" in request.form:  # Handle deleting a movie
            movie_id = int(request.form["delete_movie"])
            session["liked_movies"].remove(movie_id)
            session.modified = True

        elif "process_data" in request.form:  # Handle processing data
            if len(session["liked_movies"]) >= 4:  # Require at least 3 movies
                return redirect(url_for("select_model"))

    # Get movie details for liked movies
    liked_movie_details = MOVIES[MOVIES["id"].isin(session["liked_movies"])]

    return render_template("home.html", liked_movies=liked_movie_details)


@app.route("/process-data", methods=["POST"])
def process_data():
    global processed_data

    # Get the selected movie IDs
    selected_movie_ids = session.get("liked_movies", [])
    if not selected_movie_ids or len(selected_movie_ids) < 4:
        return redirect(url_for("home"))  # Ensure at least 3 movies are selected

    # Process the data
    X, y, movies, genre_df = preprocess_movie_data("movies.csv", selected_movie_ids)

    # Store processed data
    processed_data["X"] = X
    processed_data["y"] = y
    processed_data["movies"] = movies
    processed_data["genre_df"] = genre_df

    session["processed_data"] = True
    session.modified = True

    return redirect(url_for("home"))


@app.route("/naive-bayes")
def showNaiveBayesResults():
    global processed_data

    # Check if data has been processed
    if not session.get("processed_data"):
        return redirect(url_for("home"))

    # Fetch processed data
    X = processed_data["X"]
    y = processed_data["y"]
    movies = processed_data["movies"]
    genre_df = processed_data["genre_df"]

    # Perform the Naive Bayes analysis
    results = perform_naive_bayes_analysis(X, y, movies, genre_df)

    recommendations = results[0]

    # Base URL for poster images
    base_url = "https://image.tmdb.org/t/p/w500"

    # Convert recommendations dictionary to a list of dictionaries
    recommendations_list = [
        {
            "title": recommendations["title"][key],
            "poster_path": f"{base_url}{recommendations['poster_path'][key]}",
        }
        for key in recommendations["title"].keys()
    ]

    return render_template(
        "naiveBayes.html",
        recommendations=recommendations_list,
        accuracy=results[1],
        precision=results[2],
        recall=results[3],
        f1=results[4],
        conf_matrix=results[5],
    )


def perform_naive_bayes_analysis(X, y, movies, genre_df):

    # Address class imbalance using SMOTE
    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy=0.3)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_split, random_state=42
    )

    nb = GaussianNB() 
    nb.fit(X_train, y_train)

    # Predict probabilities and classify with adjusted threshold
    prob_threshold = 0.005
    movies["predicted_prob"] = nb.predict_proba(X)[:, 1]
    movies["predicted_liked"] = (movies["predicted_prob"] >= prob_threshold).astype(int)

    # Compute similarity for recommendations
    selected_movie_ids = session.get("liked_movies", [])
    selected_movie_indices = movies[movies["id"].isin(selected_movie_ids)].index
    selected_movies_features = X.loc[selected_movie_indices]

    # Filter movies predicted as "liked"
    predicted_liked_movies = movies[movies["predicted_liked"] == 1]
    predicted_liked_features = X.loc[predicted_liked_movies.index]

    # Compute similarity using cosine similarity
    similarity_matrix = cosine_similarity(
        selected_movies_features, predicted_liked_features
    )

    # Average similarity scores across all selected movies
    average_similarity = similarity_matrix.mean(axis=0)

    # Add similarity scores to predicted_liked_movies
    predicted_liked_movies["similarity"] = average_similarity

    # Exclude the originally selected movies from recommendations
    predicted_liked_movies = predicted_liked_movies[
        ~predicted_liked_movies["id"].isin(selected_movie_ids)
    ]

    # Sort by similarity and get the top 6 recommendations
    recommended_movies = predicted_liked_movies.sort_values(
        by="similarity", ascending=False
    ).head(6)

    # Predict on the test set
    y_pred = nb.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return (
        recommended_movies,
        accuracy,
        precision,
        recall,
        f1,
        conf_matrix,
    )


@app.route("/decision-tree")
def showDecisionTreeResults():
    global processed_data

    # Check if data has been processed
    if not session.get("processed_data"):
        return redirect(url_for("home"))

    # Fetch processed data
    X = processed_data["X"]
    y = processed_data["y"]
    movies = processed_data["movies"]
    genre_df = processed_data["genre_df"]

    # Perform the decision tree analysis
    results = perform_decision_tree_analysis(X, y, movies, genre_df)

    # Convert to dictionary
    feature_importances_dict = results[1].to_dict()

    recommendations = results[0]

    # Base URL for poster images
    base_url = "https://image.tmdb.org/t/p/w500"

    # Convert recommendations dictionary to a list of dictionaries
    recommendations_list = [
        {
            "title": recommendations["title"][key],
            "poster_path": f"{base_url}{recommendations['poster_path'][key]}",
        }
        for key in recommendations["title"].keys()
    ]

    return render_template(
        "decisionTree.html",
        recommendations=recommendations_list,
        feature_importances=feature_importances_dict,
        accuracy=results[2],
        precision=results[3],
        recall=results[4],
        f1=results[5],
        conf_matrix=results[6],
    )


def perform_decision_tree_analysis(X, y, movies, genre_df):

    # Address class imbalance using SMOTE
    smote = SMOTE(k_neighbors=3, sampling_strategy=0.1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_split, random_state=42
    )

    # Train Decision Tree Classifier
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities and classify with adjusted threshold
    prob_threshold = 0.005
    movies["predicted_prob"] = model.predict_proba(X)[:, 1]
    movies["predicted_liked"] = (movies["predicted_prob"] >= prob_threshold).astype(int)

    # Filter only the movies predicted as "liked"
    predicted_liked_movies = movies[movies["predicted_liked"] == 1]

    # Compute similarity with the selected movies
    selected_movie_indices = movies[movies["liked"] == 1].index
    selected_movies_features = X.loc[selected_movie_indices]
    predicted_liked_features = X.loc[predicted_liked_movies.index]

    # Use cosine similarity to find the closest movies
    similarity_matrix = cosine_similarity(
        selected_movies_features, predicted_liked_features
    )

    # Average similarity scores across all selected movies
    average_similarity = similarity_matrix.mean(axis=0)

    # Add similarity scores to the predicted_liked_movies DataFrame
    predicted_liked_movies["similarity"] = average_similarity

    selected_movie_ids = session.get("liked_movies", [])
    # Exclude liked movies from recommendations
    predicted_liked_movies = predicted_liked_movies[
        ~predicted_liked_movies["id"].isin(selected_movie_ids)
    ]

    # Sort by similarity and select the top 6 recommendations
    recommended_movies = predicted_liked_movies.sort_values(
        by="similarity", ascending=False
    ).head(6)

    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Feature importances
    feature_importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return (
        recommended_movies,
        feature_importances[:10],
        accuracy,
        precision,
        recall,
        f1,
        conf_matrix,
    )


@app.route("/knn-clustering")
def showKNNClusteringResults():
    global processed_data

    # Check if data has been processed
    if not session.get("processed_data"):
        return redirect(url_for("home"))

    # Fetch processed data
    X = processed_data["X"]
    y = processed_data["y"]
    movies = processed_data["movies"]
    genre_df = processed_data["genre_df"]

    # Perform the KNN analysis
    results = perform_knn_analysis(X, y, movies, genre_df)

    recommendations = results[0]

    # Base URL for poster images
    base_url = "https://image.tmdb.org/t/p/w500"

    # Convert recommendations dictionary to a list of dictionaries
    recommendations_list = [
        {
            "title": recommendations["title"][key],
            "poster_path": f"{base_url}{recommendations['poster_path'][key]}",
        }
        for key in recommendations["title"].keys()
    ]

    return render_template(
        "knnClustering.html",
        recommendations=recommendations_list,
        accuracy=results[1],
        precision=results[2],
        recall=results[3],
        f1=results[4],
        conf_matrix=results[5],
    )


def perform_knn_analysis(X, y, movies, genre_df):

    # Address class imbalance using SMOTE
    smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy=0.3)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_split, random_state=42
    )

    # Train KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
    knn.fit(X_train, y_train)

    # Predict probabilities and classify with adjusted threshold
    prob_threshold = 0.005
    movies["predicted_prob"] = knn.predict_proba(X)[:, 1]
    movies["predicted_liked"] = (movies["predicted_prob"] >= prob_threshold).astype(int)

    # Compute similarity for recommendations
    selected_movie_ids = session.get("liked_movies", [])
    selected_movie_indices = movies[movies["id"].isin(selected_movie_ids)].index
    selected_movies_features = X.loc[selected_movie_indices]

    # Filter movies predicted as "liked"
    predicted_liked_movies = movies[movies["predicted_liked"] == 1]
    predicted_liked_features = X.loc[predicted_liked_movies.index]

    # Compute similarity using cosine similarity
    similarity_matrix = cosine_similarity(
        selected_movies_features, predicted_liked_features
    )

    # Average similarity scores across all selected movies
    average_similarity = similarity_matrix.mean(axis=0)

    # Add similarity scores to predicted_liked_movies
    predicted_liked_movies["similarity"] = average_similarity

    # Exclude the originally selected movies from recommendations
    predicted_liked_movies = predicted_liked_movies[
        ~predicted_liked_movies["id"].isin(selected_movie_ids)
    ]

    # Sort by similarity and get the top 6 recommendations
    recommended_movies = predicted_liked_movies.sort_values(
        by="similarity", ascending=False
    ).head(6)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return (
        recommended_movies,
        accuracy,
        precision,
        recall,
        f1,
        conf_matrix,
    )


def preprocess_movie_data(file_path, selected_movie_ids=None):

    # Load dataset
    try:
        movies = pd.read_csv(file_path).head(data_set_size)
    except FileNotFoundError:
        return "Error: Dataset file not found."

    # 1. Process genres (multi-label binarization)
    mlb = MultiLabelBinarizer()
    movies["genres"] = movies["genres"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )
    genres_encoded = mlb.fit_transform(movies["genres"])
    genre_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=movies.index)

    # 2. Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ["popularity", "runtime"]
    movies[numerical_cols] = scaler.fit_transform(movies[numerical_cols])

    # 3. Encode categorical features
    le_language = LabelEncoder()
    le_companies = LabelEncoder()
    le_countries = LabelEncoder()
    le_adult = LabelEncoder()

    movies["original_language"] = le_language.fit_transform(movies["original_language"])
    movies["production_companies"] = le_companies.fit_transform(
        movies["production_companies"]
    )
    movies["production_countries"] = le_countries.fit_transform(
        movies["production_countries"]
    )
    movies["adult"] = le_adult.fit_transform(movies["adult"])

    # 4. Combine all features
    features = pd.concat(
        [
            movies[
                [
                    "original_language",
                    "production_companies",
                    "production_countries",
                    "runtime",
                    "adult",
                ]
            ],
            genre_df,
        ],
        axis=1,
    )

    # 5. Add 'liked' target variable based on selected_movie_ids
    movies["liked"] = 0
    if selected_movie_ids:
        movies.loc[movies["id"].isin(selected_movie_ids), "liked"] = 1

    return features, movies["liked"], movies, genre_df


if __name__ == "__main__":
    app.run(debug=True)
