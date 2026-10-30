from pydantic import BaseModel, Field

class HousingInput(BaseModel):
    MedInc: float = Field(..., description="Median Income", json_schema_extra={"examples": [3.83]})
    HouseAge: float = Field(..., description="House Age", json_schema_extra={"examples": [28.0]})
    AveRooms: float = Field(..., description="Average Rooms", json_schema_extra={"examples": [5.2]})
    Population: float = Field(..., description="Population", json_schema_extra={"examples": [1040.0]})

class HousingOutput(BaseModel):
    predicted_raw_val: float
    price_usd: float
    formatted_price: str

class ClassificationInput(BaseModel):
    sepal_length: float = Field(..., json_schema_extra={"examples": [5.1]})
    sepal_width: float = Field(..., json_schema_extra={"examples": [3.5]})
    petal_length: float = Field(..., json_schema_extra={"examples": [1.4]})
    petal_width: float = Field(..., json_schema_extra={"examples": [0.2]})
    threshold: float = Field(0.50, description="Decision threshold", json_schema_extra={"examples": [0.30]})

class ClassificationOutput(BaseModel):
    is_rush_event: bool
    probability: float
    threshold_used: float
    status: str

class TextInput(BaseModel):
    text: str = Field(..., json_schema_extra={"examples": ["Cold coffee and slow service"]})

class TextOutput(BaseModel):
    sentiment: str
    confidence: float