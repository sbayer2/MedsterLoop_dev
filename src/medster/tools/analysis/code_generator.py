# Dynamic code generation tool for custom analysis
# Allows the agent to generate and execute Python code when existing tools are insufficient

from langchain.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
import traceback
import logging
from datetime import datetime
import signal
import threading

# Timeout for code execution (seconds)
CODE_EXECUTION_TIMEOUT = 120  # 2 minutes max

from medster.tools.analysis.primitives import (
    get_patients,
    load_patient,
    search_resources,
    get_conditions,
    get_observations,
    get_medications,
    filter_by_text,
    filter_by_value,
    count_by_field,
    group_by_field,
    aggregate_numeric,
    scan_dicom_directory,
    get_dicom_metadata_from_path,
    find_patient_images,
    load_dicom_image,
    load_ecg_image,
    get_dicom_metadata,
    analyze_image_with_llm,
    analyze_ecg_for_rhythm,
    analyze_multiple_images_with_llm,
    PRIMITIVES_SPEC
)


####################################
# Input Schema
####################################

class CodeExecutionTimeout(Exception):
    """Raised when code execution exceeds timeout."""
    pass


class CodeGenerationInput(BaseModel):
    analysis_description: str = Field(
        description="Natural language description of the analysis to perform. Be specific about what data to collect and how to aggregate it."
    )
    code: str = Field(
        description=f"Python code to execute using these primitives:\n{PRIMITIVES_SPEC}\nThe code must define a function called 'analyze()' that returns a dict with results. If analyzing uploaded file content, use the 'uploaded_content' variable which contains the file text."
    )
    patient_limit: int = Field(
        default=10,  # Reduced from 50 - vision analysis is expensive!
        description="Maximum number of patients to analyze. Keep LOW (5-10) for vision/DICOM analysis to avoid timeouts."
    )
    uploaded_content: Optional[str] = Field(
        default=None,
        description="Content of an uploaded file to analyze. Available in sandbox as 'uploaded_content' variable. Use this when user uploads a file for analysis instead of querying the database."
    )


####################################
# Sandbox Environment
####################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - MEDSTER CODE EXEC - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_sandbox_globals(patient_limit: int, uploaded_content: Optional[str] = None) -> dict:
    """Create a restricted global namespace for code execution.

    Args:
        patient_limit: Max patients for get_patients()
        uploaded_content: Optional file content uploaded by user for analysis
    """

    def log_progress(message: str):
        """Progress logging function available to generated code."""
        logger.info(message)

    def search_uploaded_content(pattern: str, case_insensitive: bool = True) -> list:
        """Search uploaded content for lines matching a pattern.

        Args:
            pattern: Text pattern to search for
            case_insensitive: Whether to ignore case (default True)

        Returns:
            List of matching lines with line numbers
        """
        if not uploaded_content:
            return []

        import re
        flags = re.IGNORECASE if case_insensitive else 0
        results = []
        for i, line in enumerate(uploaded_content.split('\n'), 1):
            if re.search(pattern, line, flags):
                results.append({"line_number": i, "content": line.strip()})
        return results

    def extract_sections(start_pattern: str, end_pattern: str = None) -> list:
        """Extract sections from uploaded content between patterns.

        Args:
            start_pattern: Pattern marking section start
            end_pattern: Pattern marking section end (optional, uses next start_pattern if None)

        Returns:
            List of extracted sections
        """
        if not uploaded_content:
            return []

        import re
        lines = uploaded_content.split('\n')
        sections = []
        current_section = None
        current_lines = []

        for line in lines:
            if re.search(start_pattern, line, re.IGNORECASE):
                if current_section and current_lines:
                    sections.append({"header": current_section, "content": '\n'.join(current_lines)})
                current_section = line.strip()
                current_lines = []
            elif end_pattern and re.search(end_pattern, line, re.IGNORECASE):
                if current_section:
                    sections.append({"header": current_section, "content": '\n'.join(current_lines)})
                current_section = None
                current_lines = []
            elif current_section:
                current_lines.append(line)

        if current_section and current_lines:
            sections.append({"header": current_section, "content": '\n'.join(current_lines)})

        return sections

    return {
        # Uploaded Content (if available)
        "uploaded_content": uploaded_content,
        "search_uploaded_content": search_uploaded_content,
        "extract_sections": extract_sections,

        # FHIR Data Primitives
        "get_patients": lambda limit=patient_limit: get_patients(limit),
        "load_patient": load_patient,
        "search_resources": search_resources,
        "get_conditions": get_conditions,
        "get_observations": get_observations,
        "get_medications": get_medications,
        "filter_by_text": filter_by_text,
        "filter_by_value": filter_by_value,
        "count_by_field": count_by_field,
        "group_by_field": group_by_field,
        "aggregate_numeric": aggregate_numeric,

        # Vision/Imaging Primitives
        "scan_dicom_directory": scan_dicom_directory,
        "get_dicom_metadata_from_path": get_dicom_metadata_from_path,
        "find_patient_images": find_patient_images,
        "load_dicom_image": load_dicom_image,
        "load_ecg_image": load_ecg_image,
        "get_dicom_metadata": get_dicom_metadata,
        "analyze_image_with_llm": analyze_image_with_llm,
        "analyze_ecg_for_rhythm": analyze_ecg_for_rhythm,
        "analyze_multiple_images_with_llm": analyze_multiple_images_with_llm,

        # Safe built-ins
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "any": any,
        "all": all,
        "print": print,
        "hasattr": hasattr,
        "getattr": getattr,
        "isinstance": isinstance,
        "type": type,
        "ord": ord,  # For hash-based pseudo-random selection
        "chr": chr,  # Inverse of ord, useful for string operations

        # Exception handling
        "Exception": Exception,

        # Progress logging
        "log_progress": log_progress,

        # No dangerous functions
        "__builtins__": {},
    }


####################################
# Tool
####################################

@tool(args_schema=CodeGenerationInput)
def generate_and_run_analysis(
    analysis_description: str,
    code: str,
    patient_limit: int = 50,
    uploaded_content: Optional[str] = None
) -> dict:
    """
    Generates and executes custom Python analysis code using FHIR/vision primitives OR uploaded file content.
    Use this when existing tools don't support the specific analysis pattern needed.

    The code MUST define a function called 'analyze()' that returns a dict.

    **IMPORTANT**: If the user uploaded a file, the content is available as 'uploaded_content' variable.
    Use search_uploaded_content(pattern) and extract_sections(start, end) to analyze it.

    Example analyzing UPLOADED FILE content:
    ```
    def analyze():
        # Search for specific diagnoses in uploaded medical records
        hypertension = search_uploaded_content("hypertension")
        hyperlipidemia = search_uploaded_content("hyperlipidemia")

        # Extract visit sections
        visits = extract_sections("Visit Date:", "---")

        return {{
            "hypertension_mentions": len(hypertension),
            "hyperlipidemia_mentions": len(hyperlipidemia),
            "hypertension_lines": hypertension[:10],
            "visits_found": len(visits),
            "file_analyzed": uploaded_content is not None
        }}
    ```

    Example analyzing DATABASE (Coherent Data Set):
    ```
    def analyze():
        patients = get_patients(50)
        results = []
        for pid in patients:
            bundle = load_patient(pid)
            conditions = get_conditions(bundle)
            # ... custom analysis logic ...
        return {{"summary": results, "count": len(results)}}
    ```

    Available primitives:
    - UPLOADED FILE: uploaded_content (str), search_uploaded_content(pattern), extract_sections(start, end)
    - FHIR: get_patients, load_patient, get_conditions, get_observations, get_medications
    - Filtering: filter_by_text, filter_by_value
    - Aggregation: count_by_field, group_by_field, aggregate_numeric
    - Vision: find_patient_images, load_dicom_image, load_ecg_image, analyze_image_with_llm

    NOTE: When user uploads a file, ALWAYS use uploaded_content/search_uploaded_content instead of database queries!
    """
    try:
        logger.info(f"Starting code execution: {analysis_description}")
        logger.info(f"Patient limit: {patient_limit}")
        if uploaded_content:
            logger.info(f"Uploaded content available: {len(uploaded_content)} characters")

        # Create restricted sandbox with uploaded content if available
        sandbox_globals = create_sandbox_globals(patient_limit, uploaded_content)
        sandbox_locals = {}

        # Execute the generated code
        exec(code, sandbox_globals, sandbox_locals)

        # Check if analyze function was defined
        if "analyze" not in sandbox_locals:
            return {
                "status": "error",
                "error": "Code must define a function called 'analyze()'",
                "description": analysis_description
            }

        # Run the analysis
        logger.info("Executing analyze() function...")
        start_time = datetime.now()
        result = sandbox_locals["analyze"]()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {elapsed:.2f} seconds")

        return {
            "status": "success",
            "description": analysis_description,
            "patient_limit": patient_limit,
            "result": result
        }

    except SyntaxError as e:
        return {
            "status": "error",
            "error": f"Syntax error in generated code: {str(e)}",
            "line": e.lineno,
            "description": analysis_description
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Execution error: {str(e)}",
            "traceback": traceback.format_exc(),
            "description": analysis_description
        }


# Export the primitives spec for the agent to reference
def get_primitives_spec() -> str:
    """Return the API specification for code generation."""
    return PRIMITIVES_SPEC
