from pydantic import BaseModel, Field, validator

class HeartRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: int
    cp: int
    trestbps: int
    chol: int = Field(..., gt=0)
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    @validator("sex", "fbs", "exang")
    def binary_fields(cls, v):
        if v not in [0, 1]:
            raise ValueError("Must be 0 or 1")
        return v


class HeartResponse(BaseModel):
    prediction: int
    probability: float
