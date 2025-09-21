import os
from dotenv import load_dotenv

# Explicitly point to the .env file's absolute location inside the container
dotenv_path = "/app/.env"
load_dotenv(dotenv_path=dotenv_path)

# --- Reliable Path Definitions (Absolute paths inside the container) ---
# These paths correspond directly to the volume mounts in your docker-compose.yml
ARTIFACTS_DIR = "/app/artifacts"
VECTOR_STORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'vector_store'))


# --- API keys and LLM Configuration ---
# We just need to ensure the main variables are loaded. The LangChain/OpenAI
# libraries will handle the rest automatically.
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

# Validation check to ensure the app fails fast if secrets aren't set
if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError(
        "Azure OpenAI environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY) "
        "are not configured in your .env file."
    )