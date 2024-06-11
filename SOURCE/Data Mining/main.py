import streamlit as st
import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree
import networkx as nx

movies_metadata_df = pd.read_csv("data/CleanData.csv")
# Load Model
lr = pickle.load(open('model/LinearRegression.sav', 'rb'))
tree = pickle.load(open('model/PostDecisionTree.sav', 'rb'))

# Load dictionaries
def load_score_dict(file_path, index_col, score_col):
    df = pd.read_csv(file_path)
    return df.set_index(index_col)[score_col].to_dict()

director_score_dict = load_score_dict("data/director.csv", "director", "director_score")
company_score_dict = load_score_dict("data/company.csv", "company", "company_score")
genre_score_dict = load_score_dict("data/genre.csv", "genre", "genre_score")
star_score_dict = load_score_dict("data/star.csv", "star", "star_score")
writer_score_dict = load_score_dict("data/writer.csv", "writer", "writer_score")

# Calculate the top 10 for each factor
# def get_top_10(df, column, score_column='profit'):
#     top_10 = df.groupby(column)[score_column].mean().sort_values(ascending=False).head(10)
#     return top_10

# top_10_directors = get_top_10(movies_metadata_df, 'director')
# top_10_companies = get_top_10(movies_metadata_df, 'company')
# top_10_genres = get_top_10(movies_metadata_df, 'genre')
# top_10_stars = get_top_10(movies_metadata_df, 'star')
# top_10_writers = get_top_10(movies_metadata_df, 'writer')


# Function to convert rating to numerical
def combine_rating(rating):
    if rating in ["R", "NC-17", "TV-MA"]:
        return 'R'
    elif rating in ["G", "Approved"]:
        return 'G'
    elif rating in ["PG-13", "TV-14"]:
        return 'PG-13'
    elif rating in ["PG", "TV-PG"]:
        return 'PG'
    else:
        return 'Unknown'

# Function to preprocess features
def preprocess_features(features_df):
    # Copy features to avoid modifying the original DataFrame
    features = features_df.copy()

    # Preprocess rating column
    features['rating_converted'] = features['rating'].apply(combine_rating)
    rating_map = {'R': 3, 'G': 0, 'PG-13': 2, 'PG': 1, 'Unknown': 4}
    features['rating_converted'] = features['rating_converted'].replace(rating_map)

    # Convert other categorical columns to numerical scores
    features['director_score'] = features['director'].map(director_score_dict).fillna(0)
    features['company_score'] = features['company'].map(company_score_dict).fillna(0)
    features['genre_score'] = features['genre'].map(genre_score_dict).fillna(0)
    features['star_score'] = features['star'].map(star_score_dict).fillna(0)
    features['writer_score'] = features['writer'].map(writer_score_dict).fillna(0)

    # Extract month from release date
    features['release_date'] = pd.to_datetime(features['release_date'])
    features['month_converted'] = features['release_date'].dt.month

    # Drop original categorical columns
    features.drop(['rating', 'director', 'company', 'genre', 'star', 'writer', 'release_date'], axis=1, inplace=True)

    return features

# Function to predict profit class and profit
def predict_profit(features, model):
    # Make prediction
    predicted_profit = model.predict(features)[0]

    # Calculate estimated revenue
    budget = features['budget'].iloc[0]
    estimated_revenue = budget + predicted_profit

    # Calculate profit ratio
    profit_ratio = (predicted_profit / estimated_revenue) * 100

    # Classify profit based on profit ratio
    if profit_ratio >= 125:
        profit_class = "ALL TIME BLOCKBUSTER"
    elif profit_ratio >= 75:
        profit_class = "BLOCKBUSTER"
    elif profit_ratio >= 40:
        profit_class = "SUPERHIT"
    elif profit_ratio >= 25:
        profit_class = "HIT"
    elif profit_ratio >= 10:
        profit_class = "ABOVE AVERAGE"
    elif profit_ratio >= 0:
        profit_class = "AVERAGE"
    elif profit_ratio >= -15:
        profit_class = "BELOW AVERAGE"
    elif profit_ratio >= -40:
        profit_class = "FLOP"
    else:
        profit_class = "DISASTER"
    
    return predicted_profit, profit_class, profit_ratio

# Function to classify movie success
def classify_movie(features, model):
    # Make prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        return 'The movie is successful'
    else:
        return 'The movie is not successful'
    
def get_score_category(score):
    if score == 5:
        return 'Top Performer'
    elif score == 3:
        return 'Steady Performer'
    elif score == 1:
        return 'Under Performer'
    elif score == 0:
        return 'Loss Maker'
    else:
        return 'Unknown'
    
# # Function to get the rank of elements
# def get_feature_rank(value, top_10_series):
#     if value in top_10_series.index:
#         return top_10_series.index.get_loc(value) + 1
#     else:
#         return 'Out of top 10'

# Streamlit UI
title_html = f"""<h1 style="font-size: 40px; font-weight: bold; color: #B85042; font-family: 'Times New Roman', serif; text-shadow: 2px 2px 4px #000000;">Predicting Movie Success and Profit</h1>"""

st.markdown(title_html, unsafe_allow_html=True)
# Custom CSS to style the button
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("./Images/cinema-background.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://d1csarkz8obe9u.cloudfront.net/posterpreviews/powerpoint-cinema-background-design-template-f8795378be50233a9b188fd4ccfbeead_screen.jpg?ts=1698344870");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
# Use columns to arrange input fields
col1, col2 = st.columns(2)

with col1:
    budget = st.number_input('Budget ($)', format="%f", help="Enter the movie budget in dollars", placeholder="Enter budget in dollars")
    runtime = st.number_input('Runtime (minutes)', format="%f", help="Enter the runtime of the movie in minutes", placeholder="Enter runtime in minutes")
    rating = st.selectbox('Rating', ['R', 'NC-17', 'TV-MA', 'G', 'Approved', 'PG-13', 'TV-14', 'PG', 'TV-PG', 'Unknown'], help="Select the movie rating")
    release_date = st.date_input('Release Date', help="Select the release date of the movie")

with col2:
    
    director = st.text_input('Director', help="Enter the name of the director", placeholder="Enter director name")
    company = st.text_input('Company', help="Enter the name of the production company", placeholder="Enter company name")
    genre_options = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Documentary", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Short Film", "Sport", "Thriller", "War", "Western"]  # Assuming these are the genre options
    genre = st.selectbox('Genre', genre_options)
    star = st.text_input('Star', help="Enter the name of the main star", placeholder="Enter main star name")
    writer = st.text_input('Writer', help="Enter the name of the writer", placeholder="Enter writer name")

features = {
    'budget': budget,
    'runtime': runtime,
    'rating': rating,
    'director': director,
    'company': company,
    'genre': genre,
    'star': star,
    'writer': writer,
    'release_date': release_date
}

low_budget_threshold = movies_metadata_df['budget'].quantile(0.33)
medium_budget_threshold = movies_metadata_df['budget'].quantile(0.66)

def evaluate_movie_budget(budget):
    if budget <= low_budget_threshold:
        return 'low budget'
    elif budget <= medium_budget_threshold:
        return 'medium budget'
    else:
        return 'high budget'
    
def evaluate_movie_runtime(runtime):
    if runtime <= 75:
        return 'short film'
    elif runtime <= 120:
        return 'long film'
    else:
        return 'super long film'

# Định nghĩa hàm để đánh giá mùa phát hành của phim
def evaluate_movie_release_date(release_date):
    month = pd.to_datetime(release_date).month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'
       
features_df = pd.DataFrame([features])


if st.button('Get Result'):
    st.subheader('User Input Movie')
    st.write(features_df)

    # Preprocess features
    features_preprocessed = preprocess_features(features_df)

    # Perform decision tree classification
    class_prediction = classify_movie(features_preprocessed, tree)
    st.write('Result:', class_prediction)

    # Evaluate budget
    st.write('Budget:', evaluate_movie_budget(features_df['budget'].iloc[0]))
    
    # Evaluate runtime
    st.write('Runtime:', evaluate_movie_runtime(features_df['runtime'].iloc[0]))
    
    # Evaluate release date
    st.write('Release Date:', evaluate_movie_release_date(features_df['release_date'].iloc[0]))
    
    # Evaluate Director, Company, Genre, Star, Writer
    # director_rank = get_feature_rank(features_df["director"].iloc[0], top_10_directors)
    # company_rank = get_feature_rank(features_df["company"].iloc[0], top_10_companies)
    # genre_rank = get_feature_rank(features_df["genre"].iloc[0], top_10_genres)
    # star_rank = get_feature_rank(features_df["star"].iloc[0], top_10_stars)
    # writer_rank = get_feature_rank(features_df["writer"].iloc[0], top_10_writers)

    # st.write(f'Director {features_df["director"].iloc[0]} is ranked in the top {director_rank} directors with the highest profits' if isinstance(director_rank, int) else f'Director {features_df["director"].iloc[0]} is not in the top 10 directors with the highest profits')
    # st.write(f'Company {features_df["company"].iloc[0]} is ranked in the top {company_rank} companies with the highest profits' if isinstance(company_rank, int) else f'Company {features_df["company"].iloc[0]} is not in the top 10 companies with the highest profits')
    # st.write(f'Genre {features_df["genre"].iloc[0]} is ranked in the top {genre_rank} genres with the highest profits' if isinstance(genre_rank, int) else f'Genre {features_df["genre"].iloc[0]} is not in the top 10 genres with the highest profits')
    # st.write(f'Star {features_df["star"].iloc[0]} is ranked in the top {star_rank} stars with the highest profits' if isinstance(star_rank, int) else f'Star {features_df["star"].iloc[0]} is not in the top 10 stars with the highest profits')
    # st.write(f'Writer {features_df["writer"].iloc[0]} is ranked in the top {writer_rank} writers with the highest profits' if isinstance(writer_rank, int) else f'Writer {features_df["writer"].iloc[0]} is not in the top 10 writers with the highest profits')

    # Evaluate director, company, genre, etc.
    st.write(f'Director: {get_score_category(features_preprocessed["director_score"].iloc[0])}')
    st.write(f'Company: {get_score_category(features_preprocessed["company_score"].iloc[0])}')
    st.write(f'Genre: {get_score_category(features_preprocessed["genre_score"].iloc[0])}')
    st.write(f'Star: {get_score_category(features_preprocessed["star_score"].iloc[0])}')
    st.write(f'Writer: {get_score_category(features_preprocessed["writer_score"].iloc[0])}')
    
    # Perform decision tree classification
    class_prediction = classify_movie(features_preprocessed, tree)
    st.write('Successful or not:', class_prediction)

    # Predict using the loaded decision tree model
    decision_tree_predictions = tree.predict(features_preprocessed)

    # Get the decision path for the sample
    decision_path = tree.decision_path(features_preprocessed)
    leaf_id = tree.apply(features_preprocessed)

    # Extract the features used for decision making
    decision_features = features_preprocessed.columns
    decision_values = features_preprocessed.values[0]

    # Initialize the decision path string
    decision_path_str = ""

    # Set sample_id
    sample_id = 0

    # Traverse the decision path to construct the string representation
    node_index = 0
    for node_id in range(len(decision_path.indices) - 1):
        if node_index in decision_path.indices[decision_path.indptr[sample_id]:decision_path.indptr[sample_id + 1]]:
            # Get the feature and threshold for the current node
            feature_index = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]

            # Determine the comparison direction
            if decision_values[feature_index] <= threshold:
                comparison = "<="
            else:
                comparison = ">"

            # Append the decision node to the decision path string
            decision_path_str += f"({decision_features[feature_index]} {comparison} {threshold}) -> "
        
            # Move to the next node
            if decision_values[feature_index] <= threshold:
                node_index = tree.tree_.children_left[node_index]
            else:
                node_index = tree.tree_.children_right[node_index]

    # Get the leaf node ID and corresponding class label
    leaf_node_id = leaf_id[sample_id]
    class_label = tree.classes_[np.argmax(tree.tree_.value[leaf_node_id])]

    # Append the leaf node and class label to the decision path string
    decision_path_str += f"Leaf Node: {leaf_node_id} -> Class: {class_label}"

    # Display the decision path string
    st.write(f'Decision Path: {decision_path_str}')

    # Visualize Decision Path
    st.subheader('Decision Tree Path')

    node_indicator = tree.decision_path(features_preprocessed)
    leaf_id = tree.apply(features_preprocessed)

    # Boolean array indicating the nodes the sample goes through
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

    # Reverse the node_index to have root node on top
    node_index = node_index[::-1]

    plt.figure(figsize=(16, 12))  # Increase width by changing the first value
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    # Plot the nodes
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            label = f'Leaf node {node_id}'
            output = tree.tree_.value[node_id]
        else:
            comparison = "<=" if (features_preprocessed.iloc[sample_id, feature[node_id]] <= threshold[node_id]) else ">"
            label = f'Node {node_id}: (X[{feature[node_id]}] = {features_preprocessed.iloc[sample_id, feature[node_id]]}) {comparison} {threshold[node_id]}'
            output = tree.tree_.value[node_id]

        plt.plot([node_id], [node_id], 'ro')
        plt.text(node_id, node_id, label + f"\nOutput: {output}", fontsize=12, ha='right', va='center')

        if node_id != node_index[0]:  # Do not draw a line from the first node
            next_node_id = node_index[np.where(node_index == node_id)[0][0] - 1]
            plt.plot([node_id, next_node_id], [node_id, next_node_id], 'k-')
        
        # Plot the opposite branch at each decision node
        if node_id != node_index[-1]:  # Skip the last node (leaf)
            opposite_node_id = children_left[node_id] if comparison == ">" else children_right[node_id]
            opposite_comparison = "<=" if comparison == ">" else ">"
            opposite_label = f'Opposite node {opposite_node_id}: (X[{feature[node_id]}] = {features_preprocessed.iloc[sample_id, feature[node_id]]}) {opposite_comparison} {threshold[node_id]}'
            opposite_output = tree.tree_.value[opposite_node_id]

            plt.plot([node_id, opposite_node_id], [node_id, opposite_node_id], 'bo')
            plt.text(opposite_node_id, opposite_node_id, opposite_label + f"\nOutput: {opposite_output}", fontsize=12, ha='left', va='center')

    # Adjust the plot to rotate 90 degrees
    plt.gca().invert_yaxis()
    plt.gca().yaxis.tick_right()

    plt.title('Decision Tree Path')
    plt.ylabel('Node ID')
    plt.xlabel('Node ID')
    st.pyplot(plt)

    # Visualize Decision Tree
    st.subheader('Decision Tree Visualization')
    plt.figure(figsize=(45, 45))
    plot_tree(tree, filled=True, feature_names=features_preprocessed.columns, class_names=['Not Successful', 'Successful'], fontsize=8)
    st.pyplot(plt)



# File upload

title1_html = f"""<h1 style="font-size: 30px; font-weight: bold; color: #FFFFFF; font-family: 'Times New Roman', serif; text-shadow: 2px 2px 4px #000000;">Upload CSV for Batch Prediction</h1>"""

st.markdown(title1_html, unsafe_allow_html=True)
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Ensure the input data has all the necessary columns
    required_columns = ['budget', 'runtime', 'rating', 'director', 'company', 'genre', 'star', 'writer', 'release_date']
    if not all(col in data.columns for col in required_columns):
        st.error(f"Uploaded CSV must contain the following columns: {', '.join(required_columns)}")
        raise ValueError("Uploaded CSV is missing required columns.")
    else:
        # Preprocess the data
        preprocessed_data = preprocess_features(data)
        
        # Predict using the loaded models
        profit_predictions = lr.predict(preprocessed_data)
        class_predictions = tree.predict(preprocessed_data)
        
        # Create a DataFrame for the output
        output_data = preprocessed_data.copy()
        # output_data['predicted_profit'] = profit_predictions
        output_data['success'] = ['Successful' if pred == 1 else 'Not Successful' for pred in class_predictions]
        
        # Convert DataFrame to CSV
        output_csv = output_data.to_csv(index=False)
        
        # Create a download button for the output CSV
        st.download_button(
            label="Download Predictions",
            data=output_csv,
            file_name='predictions.csv',
            mime='text/csv'
        )