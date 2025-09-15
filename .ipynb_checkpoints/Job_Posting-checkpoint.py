import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------------
# Load and preprocess job data
# -------------------------------
df = pd.read_csv("data job posts.csv")
df = df[['Title', 'Company', 'Location', 'JobDescription', 'JobRequirment', 'RequiredQual', 'IT']]

# Fill missing values
df = df.fillna({
    'JobDescription': "",
    'JobRequirment': "",
    'RequiredQual': "",
    'Company': "Unknown",
    'Location': "Unknown"
})

# Combine all skill-related columns
df['AllSkills'] = df['JobDescription'] + " " + df['JobRequirment'] + " " + df['RequiredQual']

# Clean text: lowercase & remove special characters
df['AllSkills'] = df['AllSkills'].str.lower().apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', ' ', x))

# -------------------------------
# TF-IDF vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['AllSkills'])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔍 Job Posting Ranking System")

user_input = st.text_input("Enter your skills (comma separated)", "python, sql, machine learning")
filter_it = st.checkbox("Show only IT jobs")

if st.button("Find Jobs"):
    user_skills = [s.strip().lower() for s in user_input.split(",")]
    user_query = " ".join(user_skills)
    user_vec = vectorizer.transform([user_query])

    if filter_it:
        jobs = df[df['IT'] == True].copy()
        similarity_scores = cosine_similarity(user_vec, vectorizer.transform(jobs['AllSkills'])).flatten()
        jobs['similarity'] = similarity_scores
    else:
        similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
        df['similarity'] = similarity_scores
        jobs = df

    results = jobs.sort_values(by="similarity", ascending=False)[['Title', 'Company', 'Location', 'similarity']].head(10)
    
    st.write("### Top Matching Jobs:")
    st.dataframe(results)
