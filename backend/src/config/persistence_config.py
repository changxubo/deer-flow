from pydantic import BaseModel, Field

class PersistenceConfig(BaseModel):
    """Configuration for the persistence middleware."""

    enabled: bool = Field(default=False, description="Whether to enable persistence.")
    connection_string: str = Field(description="The connection string for the database.")

_persistence_config = PersistenceConfig(enabled=False, connection_string="")

def load_persistence_config_from_dict(config: dict) -> None:
    """Load the persistence config from a dictionary."""
    global _persistence_config
    _persistence_config = PersistenceConfig.model_validate(config)

def get_persistence_config() -> PersistenceConfig:
    """Get the persistence config."""
    return _persistence_config
