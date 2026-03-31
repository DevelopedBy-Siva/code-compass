import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "development")

    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "openai")
    DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o-mini")


settings = Settings()