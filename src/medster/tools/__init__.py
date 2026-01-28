# Medical tools for Medster clinical analysis agent
from typing import Callable, Any, Dict, Optional

# Import medical data retrieval tools
from medster.tools.medical.patient_data import (
    list_patients,
    get_patient_labs,
    get_vital_signs,
    get_demographics,
    get_patient_conditions,
    analyze_batch_conditions,
)
from medster.tools.medical.clinical_notes import (
    get_clinical_notes,
    get_soap_notes,
    get_discharge_summary,
)
from medster.tools.medical.medications import (
    get_medication_list,
    check_drug_interactions,
)
from medster.tools.medical.imaging import get_radiology_reports

# Import clinical scoring tools
from medster.tools.clinical.scores import (
    calculate_clinical_score,
    calculate_patient_score,
)

# Import MCP analysis tools
from medster.tools.analysis.mcp_client import (
    analyze_medical_document,
)

# Import code generation tool (LLM-as-orchestrator)
from medster.tools.analysis.code_generator import (
    generate_and_run_analysis,
)

# Import vision analysis tools
from medster.tools.analysis.vision_analyzer import (
    analyze_patient_dicom,
    analyze_dicom_file,
    analyze_patient_ecg,
    analyze_medical_images,
)


# Register all available tools
TOOLS: list[Callable[..., any]] = [
    # Patient data from Coherent Data Set
    list_patients,
    get_patient_labs,
    get_vital_signs,
    get_demographics,
    get_patient_conditions,
    analyze_batch_conditions,

    # Clinical notes
    get_clinical_notes,
    get_soap_notes,
    get_discharge_summary,

    # Medications
    get_medication_list,
    check_drug_interactions,

    # Imaging
    get_radiology_reports,

    # Clinical scores
    calculate_clinical_score,
    calculate_patient_score,  # Patient-aware scoring (auto-extracts from FHIR)

    # Complex analysis via MCP server
    analyze_medical_document,

    # Dynamic code generation for custom analysis
    generate_and_run_analysis,

    # Vision analysis for medical images
    analyze_patient_dicom,  # RECOMMENDED: takes patient_id, analyzes DICOM image
    analyze_dicom_file,  # Direct: analyze specific DICOM file by filename
    analyze_patient_ecg,  # Simple: takes patient_id, loads ECG internally
    analyze_medical_images,  # Advanced: takes raw base64 image data
]


# ============================================================================
# Tool Execution Utilities - For Single-Turn Architecture
# ============================================================================

def get_tool_by_name(name: str) -> Optional[Callable]:
    """
    Get a tool function by its name.

    Args:
        name: The tool name (e.g., 'get_patient_labs', 'get_vital_signs')

    Returns:
        The tool function if found, None otherwise
    """
    for tool in TOOLS:
        if hasattr(tool, 'name') and tool.name == name:
            return tool
        elif hasattr(tool, '__name__') and tool.__name__ == name:
            return tool
    return None


def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> Any:
    """
    Execute a tool by name with the given arguments.

    This function handles both LangChain-style tools (with .invoke()) and
    regular Python functions.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Dictionary of arguments to pass to the tool

    Returns:
        The result of the tool execution

    Raises:
        ValueError: If tool is not found
        Exception: Any exception raised by the tool itself
    """
    tool = get_tool_by_name(tool_name)

    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found in available tools")

    # LangChain tools have an invoke() method
    if hasattr(tool, 'invoke'):
        return tool.invoke(tool_args)

    # Regular Python functions - call directly with kwargs
    return tool(**tool_args)
