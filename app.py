import streamlit as st
import pandas as pd

import main

# repo = st.sidebar.radio("Select a repository:", ["sourcegraph/cody", "sourcegraph/sourcegraph"])
repo = st.text_input('Repository full name:', value="sourcegraph/cody", placeholder="sourcegraph/cody")
attribute = st.selectbox("Pick attribute to evaluate similarity on", ["title", "body"])
query = st.text_area("Query")
num_results = st.slider("Number of results:", 1, 10)
gh = main.load_github_api_obj()
issues = main.issues_for_repo(gh, repo)
issue_attribute = [getattr(issue, attribute) for issue in issues]
results_indices = main.get_most_related_issues_indices(query, issue_attribute, num_results)
results = [(issues[idx_tensor.item()], score_tensor.item()) for (idx_tensor, score_tensor) in results_indices]
results_table = [["Title", "Body", "Score", "URL"]]
for result, score in results:
    results_table.append([result.title, result.body, round(score, 2), result.html_url])
df = pd.DataFrame.from_records(results_table[1:], columns=results_table[0])
st.header("Top Result:")
st.write(f"Title: {results[0][0].title}")
st.write(f"Body: ")
st.markdown(results[0][0].body)
st.write(f"Score: {round(results[0][1], 2)}")
st.write(f"URL: {results[0][0].html_url}")

st.write("All Results:")
st.dataframe(df, column_config={"URL": st.column_config.LinkColumn("url")})
