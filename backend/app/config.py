from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Supply Chain Model"
    environment: str = "development"
    
    # GCP Configuration
    gcp_project_id: str = "local-dev-project"
    gcp_location: str = "us-central1"
    
    # Model Configurations
    gemini_model_name: str = "gemini-1.5-flash"
    vertex_ai_endpoint: str = ""
    
    # Vector Search
    vector_search_index_id: str = ""
    vector_search_endpoint_id: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
