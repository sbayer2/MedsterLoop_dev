from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Task(BaseModel):
    """Represents a single task in a task list."""
    id: int = Field(..., description="Unique identifier for the task.")
    description: str = Field(..., description="The description of the task.")
    done: bool = Field(False, description="Whether the task is completed.")


class TaskList(BaseModel):
    """Represents a list of tasks."""
    tasks: List[Task] = Field(..., description="The list of tasks.")


class IsDone(BaseModel):
    """Represents the boolean status of a task."""
    done: bool = Field(..., description="Whether the task is done or not.")


class Answer(BaseModel):
    """Represents an answer to the user's clinical query."""
    answer: str = Field(..., description="A comprehensive clinical analysis including relevant values, findings, temporal context, and clinical implications.")


class OptimizedToolArgs(BaseModel):
    """Represents optimized arguments for a tool call."""
    arguments: Dict[str, Any] = Field(..., description="The optimized arguments dictionary for the tool call.")


# Medical-specific schemas for future use

class CriticalValue(BaseModel):
    """Represents a critical lab or vital value requiring immediate attention."""
    parameter: str = Field(..., description="The parameter name (e.g., 'Potassium', 'Troponin')")
    value: float = Field(..., description="The measured value")
    unit: str = Field(..., description="Unit of measurement")
    reference_range: str = Field(..., description="Normal reference range")
    severity: str = Field(..., description="Severity level: critical, high, low")


class Medication(BaseModel):
    """Represents a medication entry."""
    name: str = Field(..., description="Medication name")
    dose: str = Field(..., description="Dosage")
    frequency: str = Field(..., description="Administration frequency")
    route: str = Field(..., description="Route of administration")
    start_date: Optional[str] = Field(None, description="Start date")
    indication: Optional[str] = Field(None, description="Clinical indication")


class LabResult(BaseModel):
    """Represents a laboratory result."""
    test_name: str = Field(..., description="Name of the lab test")
    value: str = Field(..., description="Result value")
    unit: str = Field(..., description="Unit of measurement")
    reference_range: str = Field(..., description="Reference range")
    status: str = Field(..., description="normal, high, low, critical")
    timestamp: str = Field(..., description="Collection timestamp")


class VitalSign(BaseModel):
    """Represents a vital sign measurement."""
    type: str = Field(..., description="Vital type: BP, HR, RR, Temp, SpO2")
    value: str = Field(..., description="Measured value")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: str = Field(..., description="Measurement timestamp")
