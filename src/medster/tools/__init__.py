# Medical tools for Medster clinical analysis agent
from typing import Callable

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
