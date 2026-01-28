import os
import json
import glob
from typing import Optional, List
from pathlib import Path

# Import GCS storage module for cloud-aware data access
from medster.utils.gcs_storage import (
    load_patient_bundle as gcs_load_patient_bundle,
    list_patients as gcs_list_patients,
    load_csv_file,
    get_storage_info,
)

####################################
# Coherent Data Set Configuration
####################################

# Path to the Coherent Data Set FHIR bundles (for local fallback)
# Download from: https://synthea.mitre.org/downloads (Coherent Dataset: 9 GB)
COHERENT_DATA_PATH = os.getenv("COHERENT_DATA_PATH", "./coherent_data/fhir")


def load_patient_bundle(patient_id: str) -> Optional[dict]:
    """
    Load a patient's FHIR bundle from the Coherent Data Set.

    Uses GCS when deployed to Cloud Run, local filesystem otherwise.

    Args:
        patient_id: The patient's unique identifier

    Returns:
        dict: FHIR Bundle containing all patient resources, or None if not found
    """
    return gcs_load_patient_bundle(patient_id)


def list_available_patients(limit: Optional[int] = None) -> List[str]:
    """
    List available patient IDs in the Coherent Data Set.

    Uses GCS when deployed to Cloud Run, local filesystem otherwise.

    Args:
        limit: Maximum number of patients to return. None returns all patients.

    Returns:
        List of patient IDs
    """
    return gcs_list_patients(limit)


def search_fhir(resource_type: str, **search_params) -> dict:
    """
    Search for FHIR resources in a patient's bundle.

    Args:
        resource_type: FHIR resource type (Patient, Observation, etc.)
        **search_params: Search parameters including 'patient' ID

    Returns:
        dict: FHIR Bundle with search results
    """
    patient_id = search_params.get("patient", search_params.get("subject", ""))

    if not patient_id:
        return {"resourceType": "Bundle", "entry": [], "total": 0}

    bundle = load_patient_bundle(patient_id)
    if not bundle:
        return {
            "resourceType": "Bundle",
            "entry": [],
            "total": 0,
            "error": f"Patient {patient_id} not found in Coherent Data Set"
        }

    # Filter resources by type
    matching_entries = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            # Apply additional filters
            if _matches_search_params(resource, search_params):
                matching_entries.append(entry)

    # Apply limit
    limit = search_params.get("_count", 100)
    matching_entries = matching_entries[:limit]

    # Sort by date if requested
    sort_param = search_params.get("_sort", "")
    if sort_param:
        reverse = sort_param.startswith("-")
        sort_field = sort_param.lstrip("-")
        matching_entries = _sort_entries(matching_entries, sort_field, reverse)

    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(matching_entries),
        "entry": matching_entries
    }


def get_fhir_resource(resource_type: str, resource_id: str) -> dict:
    """
    Get a specific FHIR resource by ID.

    For Patient resources, the resource_id is the patient_id.
    For other resources, we need to search through patient bundles.

    Args:
        resource_type: FHIR resource type
        resource_id: Resource ID

    Returns:
        dict: FHIR Resource
    """
    if resource_type == "Patient":
        bundle = load_patient_bundle(resource_id)
        if bundle:
            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Patient":
                    return resource

    # For other resources, we'd need to know which patient bundle to search
    # This is a limitation of file-based storage
    return {"error": f"Resource {resource_type}/{resource_id} not found"}


def _matches_search_params(resource: dict, params: dict) -> bool:
    """Check if a resource matches the search parameters."""
    # Category filter (e.g., 'laboratory', 'vital-signs')
    category = params.get("category", "")
    if category:
        resource_categories = resource.get("category", [])
        if isinstance(resource_categories, list):
            category_codes = []
            for cat in resource_categories:
                for coding in cat.get("coding", []):
                    category_codes.append(coding.get("code", "").lower())
            if category.lower() not in category_codes:
                return False

    # Code text filter
    code_text = params.get("code:text", "")
    if code_text:
        resource_code = resource.get("code", {}).get("text", "").lower()
        if code_text.lower() not in resource_code:
            return False

    # Status filter
    status = params.get("status", "")
    if status:
        if resource.get("status", "").lower() != status.lower():
            return False

    # Date filters (simplified)
    # In production, would parse and compare dates properly

    return True


def _sort_entries(entries: list, sort_field: str, reverse: bool) -> list:
    """Sort entries by a field."""
    def get_sort_key(entry):
        resource = entry.get("resource", {})
        # Common date fields
        for field in ["effectiveDateTime", "date", "issued", "authoredOn"]:
            if field in resource:
                return resource[field]
        return ""

    return sorted(entries, key=get_sort_key, reverse=reverse)


# Helper functions for common FHIR operations

def extract_observations(bundle: dict) -> list:
    """Extract observation data from a FHIR Bundle."""
    observations = []
    entries = bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            # Extract category codes from FHIR structure
            category_codes = []
            for cat in resource.get("category", []):
                for coding in cat.get("coding", []):
                    code = coding.get("code", "")
                    if code:
                        category_codes.append(code)

            obs = {
                "code": resource.get("code", {}).get("text", "Unknown"),
                "value": None,
                "unit": None,
                "effectiveDateTime": resource.get("effectiveDateTime", ""),
                "status": resource.get("status", ""),
                "category": category_codes,  # e.g., ["vital-signs"] or ["laboratory"]
            }

            # Extract value (can be valueQuantity, valueString, etc.)
            if "valueQuantity" in resource:
                obs["value"] = resource["valueQuantity"].get("value")
                obs["unit"] = resource["valueQuantity"].get("unit", "")
            elif "valueString" in resource:
                obs["value"] = resource["valueString"]
            elif "valueCodeableConcept" in resource:
                obs["value"] = resource["valueCodeableConcept"].get("text", "")

            # Extract reference ranges if available
            if "referenceRange" in resource:
                ref_range = resource["referenceRange"][0]
                low = ref_range.get("low", {}).get("value", "")
                high = ref_range.get("high", {}).get("value", "")
                obs["reference_range"] = f"{low}-{high}" if low and high else ""

            observations.append(obs)

    return observations


def extract_conditions(bundle: dict) -> list:
    """Extract condition/diagnosis data from a FHIR Bundle."""
    conditions = []
    entries = bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            condition = {
                "name": "",
                "code": "",
                "system": "",
                "clinical_status": "",
                "verification_status": "",
                "category": [],
                "onset_date": "",
                "abatement_date": "",
                "recorded_date": resource.get("recordedDate", ""),
            }

            # Extract condition code and name
            code_obj = resource.get("code", {})
            condition["name"] = code_obj.get("text", "")
            if "coding" in code_obj and code_obj["coding"]:
                coding = code_obj["coding"][0]
                condition["code"] = coding.get("code", "")
                condition["system"] = coding.get("system", "")
                if not condition["name"]:
                    condition["name"] = coding.get("display", "")

            # Extract clinical status
            clinical_status = resource.get("clinicalStatus", {})
            if "coding" in clinical_status and clinical_status["coding"]:
                condition["clinical_status"] = clinical_status["coding"][0].get("code", "")

            # Extract verification status
            verification = resource.get("verificationStatus", {})
            if "coding" in verification and verification["coding"]:
                condition["verification_status"] = verification["coding"][0].get("code", "")

            # Extract categories (primary, secondary, problem-list, etc.)
            categories = resource.get("category", [])
            for cat in categories:
                if "coding" in cat:
                    for coding in cat["coding"]:
                        condition["category"].append(coding.get("code", ""))

            # Extract onset date
            if "onsetDateTime" in resource:
                condition["onset_date"] = resource["onsetDateTime"]
            elif "onsetPeriod" in resource:
                condition["onset_date"] = resource["onsetPeriod"].get("start", "")

            # Extract abatement (resolution) date
            if "abatementDateTime" in resource:
                condition["abatement_date"] = resource["abatementDateTime"]

            conditions.append(condition)

    return conditions


def extract_medications(bundle: dict) -> list:
    """Extract medication data from a FHIR Bundle."""
    medications = []
    entries = bundle.get("entry", [])

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "MedicationRequest":
            med = {
                "medication": resource.get("medicationCodeableConcept", {}).get("text", "Unknown"),
                "status": resource.get("status", ""),
                "authoredOn": resource.get("authoredOn", ""),
                "dosageInstruction": "",
            }

            # Extract dosage
            if "dosageInstruction" in resource and resource["dosageInstruction"]:
                dosage = resource["dosageInstruction"][0]
                med["dosageInstruction"] = dosage.get("text", "")

            medications.append(med)

    return medications
