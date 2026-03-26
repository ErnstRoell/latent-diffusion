from pydantic import BaseModel, ConfigDict


class Configuration(BaseModel):
    model_config = ConfigDict(extra="forbid")
