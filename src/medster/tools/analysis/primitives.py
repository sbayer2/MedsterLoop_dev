# Primitive functions for LLM-generated code
# These are the building blocks the agent can compose into custom analysis

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from medster.tools.medical.api import (
    load_patient_bundle,
    list_available_patients,
    extract_conditions,
    extract_observations,
    extract_medications
)
from medster.config import (
    COHERENT_DICOM_PATH_ABS,
    COHERENT_CSV_PATH_ABS
)
# Import GCS storage functions for cloud-native data access
from medster.utils.gcs_storage import (
    USE_GCS,
    list_dicom_files as gcs_list_dicom_files,
    load_dicom_file as gcs_load_dicom_file,
    get_dicom_metadata_from_gcs,
    convert_dicom_to_png_from_gcs,
    load_csv_file as gcs_load_csv_file,
    list_dna_files as gcs_list_dna_files,
    load_dna_file as gcs_load_dna_file,
)
# Local file utilities (used when USE_GCS=false)
from medster.utils.image_utils import (
    dicom_to_base64_png,
    load_ecg_image_from_csv,
    find_patient_dicom_files,
    scan_all_dicom_files,
    get_image_metadata,
    ImageConversionError
)


def load_patient(patient_id: str) -> Dict[str, Any]:
    """Load a patient's complete FHIR bundle."""
    bundle = load_patient_bundle(patient_id)
    return bundle if bundle else {}


def get_patients(limit: Optional[int] = None) -> List[str]:
    """Get list of available patient IDs."""
    return list_available_patients(limit=limit)


def search_resources(bundle: Dict, resource_type: str) -> List[Dict]:
    """Extract all resources of a given type from a FHIR bundle."""
    if not bundle:
        return []

    resources = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            resources.append(resource)
    return resources


def get_conditions(bundle: Dict) -> List[Dict]:
    """Extract condition/diagnosis data from a FHIR bundle."""
    return extract_conditions({"entry": [{"resource": r} for r in search_resources(bundle, "Condition")]})


def get_observations(bundle: Dict, category: Optional[str] = None) -> List[Dict]:
    """Extract observations (labs, vitals) from a FHIR bundle."""
    obs_bundle = {"entry": [{"resource": r} for r in search_resources(bundle, "Observation")]}
    observations = extract_observations(obs_bundle)

    if category:
        # Filter by FHIR category field (e.g., 'laboratory', 'vital-signs')
        filtered = []
        for obs in observations:
            obs_categories = obs.get("category", [])
            # Check if any of the observation's categories match the requested category
            if any(category.lower() == cat.lower() for cat in obs_categories):
                filtered.append(obs)
        return filtered
    return observations


def get_medications(bundle: Dict) -> List[Dict]:
    """Extract medication data from a FHIR bundle."""
    return extract_medications({"entry": [{"resource": r} for r in search_resources(bundle, "MedicationRequest")]})


def filter_by_text(items: List[Dict], field: str, search_text: str, case_sensitive: bool = False) -> List[Dict]:
    """Filter items where field contains search text."""
    results = []
    search = search_text if case_sensitive else search_text.lower()

    for item in items:
        value = str(item.get(field, ""))
        if not case_sensitive:
            value = value.lower()
        if search in value:
            results.append(item)
    return results


def filter_by_value(items: List[Dict], field: str, operator: str, threshold: float) -> List[Dict]:
    """Filter items by numeric comparison (gt, lt, gte, lte, eq)."""
    results = []
    for item in items:
        value = item.get(field)
        if value is None:
            continue
        try:
            num_value = float(value)
            if operator == "gt" and num_value > threshold:
                results.append(item)
            elif operator == "lt" and num_value < threshold:
                results.append(item)
            elif operator == "gte" and num_value >= threshold:
                results.append(item)
            elif operator == "lte" and num_value <= threshold:
                results.append(item)
            elif operator == "eq" and num_value == threshold:
                results.append(item)
        except (ValueError, TypeError):
            continue
    return results


def count_by_field(items: List[Dict], field: str) -> Dict[str, int]:
    """Count occurrences of each unique value in a field."""
    counts = {}
    for item in items:
        value = str(item.get(field, "Unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def group_by_field(items: List[Dict], field: str) -> Dict[str, List[Dict]]:
    """Group items by a field value."""
    groups = {}
    for item in items:
        key = str(item.get(field, "Unknown"))
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups


def aggregate_numeric(items: List[Dict], field: str) -> Dict[str, float]:
    """Calculate statistics for a numeric field."""
    values = []
    for item in items:
        val = item.get(field)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "sum": 0}

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "sum": sum(values)
    }


# Vision and Imaging Primitives

def scan_dicom_directory() -> List[str]:
    """
    Scan the DICOM directory and return all DICOM filenames.
    Works with both local and GCS storage.

    Returns:
        List of DICOM filenames (not full paths in GCS mode)

    Example:
        dicom_files = scan_dicom_directory()
        for filename in dicom_files[:10]:  # Sample first 10
            metadata = get_dicom_metadata_from_path(filename)
            print(f"Modality: {metadata.get('modality')}")
    """
    if USE_GCS:
        # Use GCS storage to list DICOM files
        return gcs_list_dicom_files()

    # Local mode: return full paths
    try:
        dicom_files = scan_all_dicom_files(COHERENT_DICOM_PATH_ABS)
        return [str(f) for f in dicom_files]
    except Exception as e:
        return []


def find_patient_images(patient_id: str) -> Dict[str, Any]:
    """
    Find all available images for a patient (DICOM and ECG).
    Works with both local and GCS storage.

    Args:
        patient_id: Patient UUID (FHIR)

    Returns:
        Dictionary with 'dicom_files' (list of filenames) and 'has_ecg' (bool)
    """
    try:
        # Load patient FHIR bundle to get demographics
        bundle = load_patient_bundle(patient_id)
        dicom_files = []
        given_name = ""
        family_name = ""

        if bundle:
            # Extract patient name from FHIR bundle
            patient_resources = [entry.get('resource') for entry in bundle.get('entry', [])
                                 if entry.get('resource', {}).get('resourceType') == 'Patient']

            if patient_resources:
                patient = patient_resources[0]
                names = patient.get('name', [])
                if names:
                    name_parts = names[0]
                    given_name = name_parts.get('given', [''])[0] if name_parts.get('given') else ''
                    family_name = name_parts.get('family', '')

        if USE_GCS:
            # GCS mode: search through GCS-hosted DICOM files
            all_dicom_files = gcs_list_dicom_files()

            # Match by patient name pattern in filename
            if given_name and family_name:
                for filename in all_dicom_files:
                    # DICOM filename pattern: Given###_Family###_UUID.dcm
                    if given_name.lower() in filename.lower() and family_name.lower() in filename.lower():
                        dicom_files.append(filename)

            # Also try matching by patient_id in filename
            if not dicom_files:
                for filename in all_dicom_files:
                    if patient_id in filename:
                        dicom_files.append(filename)
        else:
            # Local mode: use filesystem glob
            if given_name and family_name:
                patterns = [
                    f"{given_name}*_{family_name}*",
                    f"*{given_name}*{family_name}*",
                ]
                for pattern in patterns:
                    matched_files = list(COHERENT_DICOM_PATH_ABS.glob(f"{pattern}.dcm"))
                    if matched_files:
                        dicom_files = [str(f) for f in matched_files]
                        break

            # Fallback: try UUID direct match
            if not dicom_files:
                local_files = find_patient_dicom_files(COHERENT_DICOM_PATH_ABS, patient_id)
                dicom_files = [str(f) for f in local_files]

        # Check for ECG (works in both modes now)
        has_ecg = False
        try:
            ecg_image = load_ecg_image(patient_id)
            has_ecg = ecg_image is not None
        except Exception:
            pass

        return {
            "dicom_files": dicom_files,
            "dicom_count": len(dicom_files),
            "has_ecg": has_ecg,
            "storage_mode": "gcs" if USE_GCS else "local"
        }
    except Exception as e:
        return {"error": str(e), "dicom_files": [], "dicom_count": 0, "has_ecg": False}


def load_dicom_image(patient_id: str, image_index: int = 0) -> Optional[str]:
    """
    Load a DICOM image for a patient as optimized base64 PNG.
    Works with both local and GCS storage.

    Args:
        patient_id: Patient UUID
        image_index: Which image to load (0 for first, 1 for second, etc.)

    Returns:
        Base64-encoded PNG string, or None if not found
    """
    try:
        # First find the patient's DICOM files
        image_info = find_patient_images(patient_id)
        dicom_files = image_info.get("dicom_files", [])

        if not dicom_files or image_index >= len(dicom_files):
            return None

        filename = dicom_files[image_index]

        if USE_GCS:
            # Use GCS conversion function
            return convert_dicom_to_png_from_gcs(filename)
        else:
            # Local mode: use local file path
            return dicom_to_base64_png(filename, target_size=(800, 800), quality=85)

    except Exception as e:
        print(f"Error loading DICOM image: {e}")
        return None


def load_dicom_image_by_filename(filename: str) -> Optional[str]:
    """
    Load a DICOM image by filename (for direct file access).
    Works with both local and GCS storage.

    Args:
        filename: DICOM filename

    Returns:
        Base64-encoded PNG string, or None if not found
    """
    try:
        if USE_GCS:
            return convert_dicom_to_png_from_gcs(filename)
        else:
            # Local mode: construct full path
            filepath = COHERENT_DICOM_PATH_ABS / filename
            if filepath.exists():
                return dicom_to_base64_png(str(filepath), target_size=(800, 800), quality=85)
            return None
    except Exception as e:
        print(f"Error loading DICOM image by filename: {e}")
        return None


def _extract_ecg_from_csv_content(csv_content: str, patient_id: str) -> Optional[str]:
    """
    Extract ECG base64 image from observations.csv content.

    Args:
        csv_content: Raw CSV content as string
        patient_id: Patient UUID to find

    Returns:
        Base64-encoded PNG string, or None if not found
    """
    import csv
    from io import StringIO

    try:
        reader = csv.DictReader(StringIO(csv_content))

        for row in reader:
            # ECG observations have LOINC code 29303009 (Electrocardiographic procedure)
            # or contain base64-encoded image data in the VALUE field
            row_patient = row.get('PATIENT', '')

            if patient_id in row_patient:
                # Check for ECG observation
                description = row.get('DESCRIPTION', '').lower()
                code = row.get('CODE', '')
                value = row.get('VALUE', '')

                # ECG entries typically have base64 PNG data
                if 'electrocardiogram' in description or code == '29303009':
                    # Value should contain base64 PNG data
                    if value and len(value) > 100:  # Base64 images are typically long
                        # Check if it's base64 encoded
                        if value.startswith('data:image'):
                            # Extract base64 from data URL
                            return value.split(',')[1] if ',' in value else value
                        else:
                            # Assume raw base64
                            return value

        return None
    except Exception as e:
        print(f"Error extracting ECG from CSV: {e}")
        return None


def load_ecg_image(patient_id: str) -> Optional[str]:
    """
    Load ECG image for a patient from observations.csv.
    Works with both local and GCS storage.

    Args:
        patient_id: Patient UUID

    Returns:
        Base64-encoded PNG string, or None if not found
    """
    try:
        if USE_GCS:
            # Load observations.csv from GCS and parse ECG
            csv_content = gcs_load_csv_file("observations.csv")
            if csv_content:
                return _extract_ecg_from_csv_content(csv_content, patient_id)
            return None
        else:
            # Local mode
            ecg_path = COHERENT_CSV_PATH_ABS / "observations.csv"
            return load_ecg_image_from_csv(ecg_path, patient_id)
    except Exception as e:
        print(f"Error loading ECG image: {e}")
        return None


def get_dicom_metadata(patient_id: str, image_index: int = 0) -> Dict[str, Any]:
    """
    Get metadata for a patient's DICOM image.
    Works with both local and GCS storage.

    Args:
        patient_id: Patient UUID
        image_index: Which image to get metadata for

    Returns:
        Dictionary with modality, study description, dimensions, etc.
    """
    try:
        # First find the patient's DICOM files
        image_info = find_patient_images(patient_id)
        dicom_files = image_info.get("dicom_files", [])

        if not dicom_files or image_index >= len(dicom_files):
            return {"error": "Image not found", "patient_id": patient_id}

        filename = dicom_files[image_index]

        if USE_GCS:
            # Use GCS metadata function
            return get_dicom_metadata_from_gcs(filename)
        else:
            # Local mode: use local file
            return get_image_metadata(Path(filename) if not isinstance(filename, Path) else filename)

    except Exception as e:
        return {"error": str(e), "patient_id": patient_id}


def get_dicom_metadata_from_path(dicom_path: str) -> Dict[str, Any]:
    """
    Get metadata for a DICOM file from its path or filename.
    Works with both local and GCS storage.

    Args:
        dicom_path: Full path to DICOM file (local) or filename (GCS)

    Returns:
        Dictionary with modality, study description, dimensions, etc.

    Example:
        dicom_files = scan_dicom_directory()
        for path in dicom_files[:5]:
            metadata = get_dicom_metadata_from_path(path)
            print(f"Modality: {metadata.get('modality')}")
    """
    try:
        if USE_GCS:
            # In GCS mode, dicom_path is a filename
            # Extract just the filename if full path provided
            filename = dicom_path.split('/')[-1] if '/' in dicom_path else dicom_path
            return get_dicom_metadata_from_gcs(filename)
        else:
            # Local mode: use full path
            return get_image_metadata(Path(dicom_path))
    except Exception as e:
        return {"error": str(e), "path": dicom_path}


def analyze_image_with_llm(image_base64: str, prompt: str) -> str:
    """
    Analyze a medical image using Claude's vision API.

    This primitive enables autonomous vision analysis within generated code.
    Use this after loading images with load_dicom_image() or load_ecg_image().

    Args:
        image_base64: Base64-encoded PNG image string
        prompt: Clinical question or analysis request (e.g., "Does this ECG show atrial fibrillation?")

    Returns:
        Vision analysis as text

    Example:
        ecg = load_ecg_image(patient_id)
        if ecg:
            analysis = analyze_image_with_llm(
                ecg,
                "Analyze this ECG for atrial fibrillation pattern. Report yes/no and key findings."
            )
    """
    try:
        from medster.model import call_llm

        response = call_llm(
            prompt=prompt,
            images=[image_base64],
            model="claude-sonnet-4.5"  # Use Claude Sonnet 4.5 for vision
        )

        # Extract text content from response
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        return f"Vision analysis error: {str(e)}"


def analyze_ecg_for_rhythm(patient_id: str, clinical_context: str = "") -> Dict[str, Any]:
    """
    Analyze ECG image for cardiac rhythm with structured parsing.

    This primitive loads the ECG, performs vision analysis, and parses the result
    into structured data to avoid false positives from keyword matching.

    Args:
        patient_id: Patient UUID
        clinical_context: Optional clinical context (e.g., "Patient with HTN and Hyperlipidemia")

    Returns:
        Dictionary with structured rhythm analysis:
        {
            "patient_id": str,
            "ecg_available": bool,
            "rhythm": str (e.g., "Normal Sinus Rhythm", "Atrial Fibrillation", "Other"),
            "afib_detected": bool,
            "rr_intervals": str (e.g., "Regular", "Irregular", "Irregularly Irregular"),
            "p_waves": str (e.g., "Present and normal", "Absent", "Abnormal"),
            "baseline": str (e.g., "Normal", "Fibrillatory", "Other"),
            "confidence": str (e.g., "High", "Medium", "Low"),
            "clinical_significance": str,
            "raw_analysis": str
        }

    Example:
        result = analyze_ecg_for_rhythm("patient-uuid-123", "HTN + Hyperlipidemia")
        if result["afib_detected"]:
            print(f"AFib detected with {result['confidence']} confidence")
    """
    try:
        # Load ECG image
        ecg_image = load_ecg_image(patient_id)

        if not ecg_image:
            return {
                "patient_id": patient_id,
                "ecg_available": False,
                "rhythm": "Unknown",
                "afib_detected": False,
                "rr_intervals": "Unknown",
                "p_waves": "Unknown",
                "baseline": "Unknown",
                "confidence": "N/A",
                "clinical_significance": "No ECG image available for analysis",
                "raw_analysis": ""
            }

        # Structured prompt for ECG rhythm analysis
        context_str = f" (Clinical context: {clinical_context})" if clinical_context else ""
        prompt = f"""Analyze this ECG tracing for patient {patient_id}{context_str}.

Specifically assess for atrial fibrillation patterns and provide your analysis in this EXACT format:

RHYTHM: [State the rhythm - Normal Sinus Rhythm, Atrial Fibrillation, or Other]
R-R INTERVALS: [Regular, Irregular, or Irregularly Irregular]
P WAVES: [Present and normal, Absent, or Abnormal]
BASELINE: [Normal, Fibrillatory, or Other]
CLINICAL SIGNIFICANCE: [Brief clinical assessment]
CONFIDENCE: [High, Medium, or Low]

Be precise in your RHYTHM classification. Only state "Atrial Fibrillation" if you see irregularly irregular R-R intervals, absent P waves, AND fibrillatory baseline."""

        # Get vision analysis
        from medster.model import call_llm
        response = call_llm(
            prompt=prompt,
            images=[ecg_image],
            model="claude-sonnet-4.5"  # Use Claude Sonnet 4.5 for ECG
        )

        raw_text = response.content if hasattr(response, 'content') else str(response)

        # Parse structured response with better logic
        def extract_field(text: str, field_name: str) -> str:
            """Extract value after 'FIELD_NAME:' line"""
            import re
            pattern = rf'{field_name}:\s*(.+?)(?:\n|$)'
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else "Unknown"

        rhythm = extract_field(raw_text, "RHYTHM")
        rr_intervals = extract_field(raw_text, "R-R INTERVALS")
        p_waves = extract_field(raw_text, "P WAVES")
        baseline = extract_field(raw_text, "BASELINE")
        significance = extract_field(raw_text, "CLINICAL SIGNIFICANCE")
        confidence = extract_field(raw_text, "CONFIDENCE")

        # Determine AFib based on RHYTHM field, not keyword matching
        afib_detected = False
        rhythm_lower = rhythm.lower()

        if "atrial fibrillation" in rhythm_lower or rhythm_lower == "afib":
            afib_detected = True
        elif "normal sinus rhythm" in rhythm_lower or rhythm_lower == "nsr":
            afib_detected = False
        # Secondary check: if rhythm unclear, check for classic AFib triad
        elif rhythm_lower == "unknown" or rhythm_lower == "other":
            afib_triad = (
                "irregularly irregular" in rr_intervals.lower() and
                "absent" in p_waves.lower() and
                "fibrillatory" in baseline.lower()
            )
            afib_detected = afib_triad

        return {
            "patient_id": patient_id,
            "ecg_available": True,
            "rhythm": rhythm,
            "afib_detected": afib_detected,
            "rr_intervals": rr_intervals,
            "p_waves": p_waves,
            "baseline": baseline,
            "confidence": confidence,
            "clinical_significance": significance,
            "raw_analysis": raw_text
        }

    except Exception as e:
        return {
            "patient_id": patient_id,
            "ecg_available": False,
            "rhythm": "Error",
            "afib_detected": False,
            "rr_intervals": "Error",
            "p_waves": "Error",
            "baseline": "Error",
            "confidence": "N/A",
            "clinical_significance": f"Analysis error: {str(e)}",
            "raw_analysis": ""
        }


def analyze_multiple_images_with_llm(images: List[str], prompt: str) -> str:
    """
    Analyze multiple medical images together using the LLM's vision API.

    Use this to compare images or analyze them in context of each other.

    Args:
        images: List of base64-encoded PNG image strings
        prompt: Clinical question or analysis request

    Returns:
        Vision analysis as text

    Example:
        images = [load_dicom_image(pid, 0) for pid in patient_ids]
        images = [img for img in images if img]  # Remove None values
        if images:
            analysis = analyze_multiple_images_with_llm(
                images,
                "Compare these brain MRIs and identify any masses or hemorrhage."
            )
    """
    try:
        from medster.model import call_llm

        # Filter out None values
        valid_images = [img for img in images if img]

        if not valid_images:
            return "No valid images to analyze"

        response = call_llm(
            prompt=prompt,
            images=valid_images,
            model="claude-sonnet-4.5"  # Use Claude Sonnet 4.5 for vision
        )

        # Extract text content from response
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        return f"Vision analysis error: {str(e)}"


# API specification for LLM code generation
PRIMITIVES_SPEC = """
Available functions for custom analysis:

# Patient Data
get_patients(limit: int = None) -> List[str]
    # Returns list of patient IDs

load_patient(patient_id: str) -> Dict
    # Returns complete FHIR bundle for a patient

# Resource Extraction
search_resources(bundle: Dict, resource_type: str) -> List[Dict]
    # Extract resources by type: "Patient", "Condition", "Observation", "MedicationRequest"

get_conditions(bundle: Dict) -> List[Dict]
    # Returns: [{"name": str, "code": str, "clinical_status": str, "category": list}]

get_observations(bundle: Dict, category: str = None) -> List[Dict]
    # Returns: [{"code": str, "value": any, "unit": str, "effectiveDateTime": str}]
    # category: "laboratory", "vital-signs"

get_medications(bundle: Dict) -> List[Dict]
    # Returns: [{"medication": str, "status": str, "dosageInstruction": str}]

# Filtering
filter_by_text(items: List, field: str, search_text: str) -> List[Dict]
    # Filter where field contains text (case-insensitive)

filter_by_value(items: List, field: str, operator: str, threshold: float) -> List[Dict]
    # operator: "gt", "lt", "gte", "lte", "eq"

# Aggregation
count_by_field(items: List, field: str) -> Dict[str, int]
    # Count occurrences of each unique value

group_by_field(items: List, field: str) -> Dict[str, List]
    # Group items by field value

aggregate_numeric(items: List, field: str) -> Dict
    # Returns: {"count", "min", "max", "mean", "sum"}

# Vision and Imaging (Multimodal Analysis)
scan_dicom_directory() -> List[str]
    # Scan DICOM directory and return ALL DICOM file paths
    # Returns: List of file path strings
    # Use this for database-wide DICOM analysis (fast - no patient iteration)
    # Example: dicom_files = scan_dicom_directory()  # Returns all 298 files

get_dicom_metadata_from_path(dicom_path: str) -> Dict
    # Get metadata for DICOM file from file path
    # Returns: {"modality": str, "study_description": str, "body_part": str, "dimensions": str, ...}
    # Use with scan_dicom_directory() for fast metadata extraction
    # Example: metadata = get_dicom_metadata_from_path(dicom_files[0])

find_patient_images(patient_id: str) -> Dict
    # Returns: {"dicom_files": List[str], "dicom_count": int, "has_ecg": bool}
    # Find all available images for a patient

load_dicom_image(patient_id: str, image_index: int = 0) -> Optional[str]
    # Load DICOM image as optimized base64 PNG string
    # image_index: 0 for first image, 1 for second, etc.
    # Returns base64 string ready for vision analysis

load_ecg_image(patient_id: str) -> Optional[str]
    # Load ECG image as base64 PNG string from observations.csv
    # Returns base64 string ready for vision analysis

get_dicom_metadata(patient_id: str, image_index: int = 0) -> Dict
    # Returns: {"modality": str, "study_description": str, "body_part": str, "dimensions": str, ...}
    # Get DICOM metadata without loading pixel data (requires patient ID)

analyze_image_with_llm(image_base64: str, prompt: str) -> str
    # Analyze a single medical image using LLM vision API
    # image_base64: Base64 PNG string from load_dicom_image() or load_ecg_image()
    # prompt: Clinical question (e.g., "Does this ECG show atrial fibrillation?")
    # Returns: Vision analysis as text
    # Example: analysis = analyze_image_with_llm(ecg, "Detect AFib pattern")

analyze_ecg_for_rhythm(patient_id: str, clinical_context: str = "") -> Dict
    # RECOMMENDED FOR ECG RHYTHM ANALYSIS - Structured parsing prevents false positives
    # Loads ECG, performs vision analysis, and parses result into structured data
    # Returns: {"patient_id", "ecg_available", "rhythm", "afib_detected", "rr_intervals",
    #           "p_waves", "baseline", "confidence", "clinical_significance", "raw_analysis"}
    # rhythm: "Normal Sinus Rhythm", "Atrial Fibrillation", or "Other"
    # afib_detected: bool (based on RHYTHM field, not keyword matching)
    # Example: result = analyze_ecg_for_rhythm(pid, "HTN + Hyperlipidemia")
    #          if result["afib_detected"]: print(f"AFib: {result['confidence']} confidence")

analyze_multiple_images_with_llm(images: List[str], prompt: str) -> str
    # Analyze multiple images together using LLM vision API
    # images: List of base64 PNG strings
    # prompt: Clinical question for comparative analysis
    # Returns: Vision analysis as text
    # Example: analysis = analyze_multiple_images_with_llm([img1, img2], "Compare these MRIs")

# Progress Logging
log_progress(message: str) -> None
    # Log progress during long-running analysis
    # Use this to report status when iterating through many patients
    # Example: log_progress(f"Processing patient {i+1}/{total}")

# ============================================================================
# UPLOADED FILE ANALYSIS (Use when user uploads a file instead of database)
# ============================================================================
# When user uploads a file, the content is available as 'uploaded_content' variable
# ALWAYS use these for uploaded files - DO NOT use database primitives for uploaded files!

uploaded_content: str
    # The full text content of the uploaded file (if available)
    # Check with: if uploaded_content: ...
    # Example: lines = uploaded_content.split('\\n')

search_uploaded_content(pattern: str, case_insensitive: bool = True) -> List[Dict]
    # Search uploaded content for lines matching a regex pattern
    # Returns: [{"line_number": int, "content": str}]
    # Example: hypertension_lines = search_uploaded_content("hypertension")
    # Example: dates = search_uploaded_content(r"\\d{1,2}/\\d{1,2}/\\d{4}")

extract_sections(start_pattern: str, end_pattern: str = None) -> List[Dict]
    # Extract sections from uploaded content between patterns
    # Returns: [{"header": str, "content": str}]
    # Example: visits = extract_sections("Visit Date:", "---")
    # Example: diagnoses = extract_sections("DIAGNOSIS:", "PLAN:")
"""
