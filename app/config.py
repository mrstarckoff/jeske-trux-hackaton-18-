from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'weighted-rrf-orchestrator'
    chroma_path: str = './.chroma'
    dataset_csv: str = './data/billboards.csv'
    max_candidates_per_channel: int = 10
    final_top_k: int = 5
    rrf_k: int = 60


settings = Settings()
