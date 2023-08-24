import os
from functools import cache

import streamlit
from dotenv import load_dotenv
from github import Auth
from github import Github
from github.Issue import Issue
from github.PaginatedList import PaginatedList

load_dotenv()  # take environment variables from .env.


@streamlit.cache_resource
def load_github_api_obj() -> Github:
    auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
    return Github(auth=auth)


@cache
def issues_for_repo(github_instance: Github, repo_name: str) -> PaginatedList[Issue]:
    repo = github_instance.get_repo(repo_name)
    issues = repo.get_issues()
    return issues
