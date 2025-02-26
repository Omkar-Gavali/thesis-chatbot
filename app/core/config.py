import yaml
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    app_name: str
    app_version: str
    host: str
    port: int
    thesis_pdf_path: str
    vector_store_path: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    vector_store_type: str
    chunk_size: int
    chunk_overlap: int
    groq_api_key: str
    caching_enabled: bool
    cache_dir: str
    groq_api_key: str

    class Config:
        env_file = ".env"

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return Settings(
        app_name=config["app"]["name"],
        app_version=config["app"]["version"],
        host=config["app"]["host"],
        port=config["app"]["port"],
        thesis_pdf_path=config["paths"]["thesis_pdf"],
        vector_store_path=config["paths"]["vector_store"],
        embedding_model=config["embedding"]["model"],
        llm_provider=config["llm"]["provider"],
        llm_model=config["llm"]["model"],
        vector_store_type=config["vector_store"]["type"],
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )

settings = load_config()
