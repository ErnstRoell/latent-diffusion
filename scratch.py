from pydantic import BaseModel, ConfigDict


class Model_actual(BaseModel):
    model_config = ConfigDict(extra="forbid")
    val1: str


dict_to_be_validated = {"val1": "Hello", "val2": "World"}

x = Model_actual(**dict_to_be_validated)
print(x)
