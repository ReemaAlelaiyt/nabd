"""Tests must be repeatable: never call the real AI or LangSmith, even if a .env file exists.
(load_dotenv does not override variables that are already set.)"""
import os

os.environ["LLM_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"
