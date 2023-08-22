import json
import os
from functools import cache

import fire
import streamlit
from github import Github
from github import Auth
import subprocess

from dotenv import load_dotenv

import semantic_search
from pydantic import BaseModel


load_dotenv()  # take environment variables from .env.


class Issue(BaseModel):
    title: str
    body: str
    id: int


# def get_most_related_issues(repo_name: str, query: str, attribute: str):
#     # repo_name = "sourcegraph/cody"
#     # query = "bad summary"
#     gh = load_github_api_obj()
#     issues = issues_for_repo(gh, repo_name)
#     most_related_issues = get_most_related_issues(query, [issue.attribute for issue in issues])
#     return most_related_issues
#     # for issue, score in most_related_issues:
#     #     print(f"score: {score}")
#     #     print(f"issue title: {issue}")


def get_code_owners_for_repo(repo_name: str) -> list[str]:
    result = subprocess.run(["src", "codeowners", "get", f"-repo='{repo_name}'"], capture_output=True)
    try:
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        print(e)
        exit(1)

    owners = json.loads(result.stdout)
    assert isinstance(owners, list)
    return owners


def message_to_cody(repo_name: str) -> str:
    owners = get_code_owners_for_repo(repo_name)
    message = f"Code owners for {repo_name} are:"
    message += ", ".join(owners)
    return message


def load_github_api_obj():
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    g = Github(auth=auth)
    return g


@cache
def issues_for_repo(github_instance: Github, repo_name: str):
    repo = github_instance.get_repo(repo_name)
    issues = repo.get_issues()
    return issues


@streamlit.cache_data
def get_most_related_issues_indices(query, issues: list, k=5):
    indices = semantic_search.get_top_k_similar_sentences(query, issues, k=k)
    return indices




if __name__ == '__main__':
    pass
    # fire.Fire(get_most_related_issues)

