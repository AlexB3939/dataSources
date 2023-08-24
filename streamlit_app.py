import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

import github_api
import semantic_search


model_names = [
    "msmarco-distilbert-base-v2",
    "all-MiniLM-L6-v2",
]
min_num_results = 1
max_num_results = 10

model_name = st.sidebar.radio("Select a model to use for semantic similarity:", model_names)
sentence_embedder = semantic_search.SentenceEmbedder(model_name)

repo = st.text_input('Repository full name:', value="sourcegraph/cody", placeholder="sourcegraph/cody")
attribute = st.selectbox("Pick issue attribute to evaluate similarity on", ["title", "body"])
query = st.text_area("Query")
if query == "":
    st.stop()
num_results = st.slider("Number of results:", min_num_results, max_num_results, value=5)

gh = github_api.load_github_api_obj()

issues = github_api.issues_for_repo(gh, repo)
try:
    issues_attributes = [getattr(issue, attribute) for issue in issues]
except AttributeError:
    st.error(f"Error, no issue attribute with name {attribute}", icon="🚨")
    st.stop()

results_indices = sentence_embedder.get_top_k_similar_sentence_indices(query, issues_attributes, num_results)
results = [(issues[idx], score) for (idx, score) in results_indices]

results_table_header = [["Title", "Body", "Score", "URL"]]
decimal_places = 2
results_table = [[result.title, result.body, round(score, decimal_places), result.html_url] for result, score in results]

df = pd.DataFrame.from_records(results_table, columns=results_table_header)

st.header("Top Result:")
st.write(f"Title: {results_table[0][0]}")
st.write(f"Body: ")
st.markdown(results_table[0][1])
st.write(f"Score: {results_table[0][2]}")
st.write(f"URL: {results_table[0][3]}")

st.write("All Results:")
st.dataframe(df, column_config={"URL": st.column_config.LinkColumn("url")})
