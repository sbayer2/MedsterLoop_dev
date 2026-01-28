from langchain.tools import tool
from typing import Literal, Optional
from pydantic import BaseModel, Field
from medster.tools.medical.api import search_fhir, get_fhir_resource, extract_observations, extract_conditions, list_available_patients

####################################
# Input Schemas
####################################

class PatientLabsInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    lab_type: Optional[str] = Field(default=None, description="Specific lab type to filter by (e.g., 'CBC', 'CMP', 'BMP', 'lipid panel', 'troponin', 'BNP'). Leave empty for all labs.")
    limit: int = Field(default=20, description="Maximum number of lab results to retrieve.")
    date_start: Optional[str] = Field(default=None, description="Filter for labs collected after this date (YYYY-MM-DD).")
    date_end: Optional[str] = Field(default=None, description="Filter for labs collected before this date (YYYY-MM-DD).")


class VitalSignsInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    vital_type: Optional[str] = Field(default=None, description="Specific vital sign type (e.g., 'blood-pressure', 'heart-rate', 'respiratory-rate', 'body-temperature', 'oxygen-saturation'). Leave empty for all vitals.")
    limit: int = Field(default=50, description="Maximum number of vital sign measurements to retrieve.")
    date_start: Optional[str] = Field(default=None, description="Filter for vitals measured after this date (YYYY-MM-DD).")
    date_end: Optional[str] = Field(default=None, description="Filter for vitals measured before this date (YYYY-MM-DD).")


class DemographicsInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")


class ListPatientsInput(BaseModel):
    limit: Optional[int] = Field(default=None, description="Maximum number of patients to return. None returns all available patients.")


class PatientConditionsInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    condition_filter: Optional[str] = Field(default=None, description="Filter conditions by keyword (e.g., 'cancer', 'diabetes', 'hypertension'). Leave empty for all conditions.")
    include_resolved: bool = Field(default=True, description="Include resolved/historical conditions in results.")


class BatchConditionsInput(BaseModel):
    patient_limit: int = Field(default=50, description="Number of patients to analyze.")
    condition_filter: Optional[str] = Field(default=None, description="Filter conditions by keyword (e.g., 'cancer', 'diabetes', 'hypertension').")


####################################
# Tools
####################################

@tool(args_schema=PatientLabsInput)
def get_patient_labs(
    patient_id: str,
    lab_type: Optional[str] = None,
    limit: int = 20,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None
) -> dict:
    """
    Fetches a patient's laboratory results including CBC, metabolic panels, cardiac markers, etc.
    Returns lab values with reference ranges and collection timestamps.
    Useful for assessing patient's current clinical status and trends over time.
    """
    # Build FHIR search parameters
    params = {
        "patient": patient_id,
        "category": "laboratory",
        "_count": limit,
        "_sort": "-date",  # Most recent first
    }

    # Add date filters
    if date_start:
        params["date"] = f"ge{date_start}"
    if date_end:
        if "date" in params:
            params["date"] = [params["date"], f"le{date_end}"]
        else:
            params["date"] = f"le{date_end}"

    # Add lab type filter using LOINC codes or text search
    if lab_type:
        params["code:text"] = lab_type

    try:
        bundle = search_fhir("Observation", **params)
        observations = extract_observations(bundle)

        return {
            "patient_id": patient_id,
            "lab_count": len(observations),
            "labs": observations
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "labs": []
        }


@tool(args_schema=VitalSignsInput)
def get_vital_signs(
    patient_id: str,
    vital_type: Optional[str] = None,
    limit: int = 50,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None
) -> dict:
    """
    Retrieves a patient's vital sign measurements including blood pressure, heart rate,
    respiratory rate, temperature, and oxygen saturation.
    Useful for assessing hemodynamic stability and identifying trends.
    """
    # Build FHIR search parameters
    params = {
        "patient": patient_id,
        "category": "vital-signs",
        "_count": limit,
        "_sort": "-date",
    }

    # Add date filters
    if date_start:
        params["date"] = f"ge{date_start}"
    if date_end:
        if "date" in params:
            params["date"] = [params["date"], f"le{date_end}"]
        else:
            params["date"] = f"le{date_end}"

    # Add vital type filter
    if vital_type:
        # Map common names to FHIR vital sign categories
        vital_code_map = {
            "blood-pressure": "85354-9",
            "heart-rate": "8867-4",
            "respiratory-rate": "9279-1",
            "body-temperature": "8310-5",
            "oxygen-saturation": "2708-6",
        }
        if vital_type in vital_code_map:
            params["code"] = vital_code_map[vital_type]
        else:
            params["code:text"] = vital_type

    try:
        bundle = search_fhir("Observation", **params)
        vitals = extract_observations(bundle)

        return {
            "patient_id": patient_id,
            "vital_count": len(vitals),
            "vitals": vitals
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "vitals": []
        }


@tool(args_schema=DemographicsInput)
def get_demographics(patient_id: str) -> dict:
    """
    Retrieves a patient's demographic information including name, date of birth,
    gender, address, and contact information.
    Useful for patient identification and communication context.
    """
    try:
        patient = get_fhir_resource("Patient", patient_id)

        # Extract demographic data
        demographics = {
            "patient_id": patient_id,
            "name": "",
            "birth_date": patient.get("birthDate", ""),
            "gender": patient.get("gender", ""),
            "address": "",
            "phone": "",
            "marital_status": "",
        }

        # Extract name
        if "name" in patient and patient["name"]:
            name_entry = patient["name"][0]
            given = " ".join(name_entry.get("given", []))
            family = name_entry.get("family", "")
            demographics["name"] = f"{given} {family}".strip()

        # Extract address
        if "address" in patient and patient["address"]:
            addr = patient["address"][0]
            line = " ".join(addr.get("line", []))
            city = addr.get("city", "")
            state = addr.get("state", "")
            postal = addr.get("postalCode", "")
            demographics["address"] = f"{line}, {city}, {state} {postal}".strip(", ")

        # Extract phone
        if "telecom" in patient:
            for telecom in patient["telecom"]:
                if telecom.get("system") == "phone":
                    demographics["phone"] = telecom.get("value", "")
                    break

        # Extract marital status
        if "maritalStatus" in patient:
            demographics["marital_status"] = patient["maritalStatus"].get("text", "")

        return demographics

    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e)
        }


@tool(args_schema=PatientConditionsInput)
def get_patient_conditions(
    patient_id: str,
    condition_filter: Optional[str] = None,
    include_resolved: bool = True
) -> dict:
    """
    Retrieves a patient's diagnoses and conditions from FHIR Condition resources.
    Returns structured diagnosis data including ICD codes, clinical status, and categories.
    Useful for identifying primary/secondary diagnoses, cancer history, chronic conditions.
    Can filter by condition keyword (e.g., 'cancer', 'diabetes', 'heart').
    """
    try:
        # Search for Condition resources
        bundle = search_fhir("Condition", patient=patient_id, _count=500)
        conditions = extract_conditions(bundle)

        # Filter by keyword if specified
        if condition_filter:
            filter_lower = condition_filter.lower()
            conditions = [c for c in conditions if filter_lower in c.get("name", "").lower()
                         or filter_lower in c.get("code", "").lower()
                         or filter_lower in str(c.get("category", "")).lower()]

        # Filter out resolved if requested
        if not include_resolved:
            conditions = [c for c in conditions if c.get("clinical_status", "").lower() not in ["resolved", "inactive", "remission"]]

        # Separate by category
        primary = [c for c in conditions if "encounter-diagnosis" in str(c.get("category", "")).lower() or "primary" in str(c.get("category", "")).lower()]
        secondary = [c for c in conditions if c not in primary]

        return {
            "patient_id": patient_id,
            "total_conditions": len(conditions),
            "primary_diagnoses": primary,
            "other_conditions": secondary,
            "all_conditions": conditions
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "conditions": []
        }


@tool(args_schema=BatchConditionsInput)
def analyze_batch_conditions(
    patient_limit: int = 50,
    condition_filter: Optional[str] = None
) -> dict:
    """
    Analyzes conditions/diagnoses across multiple patients from the Coherent Data Set.
    Returns aggregated diagnosis statistics including most common conditions and patient lists.
    Use this for population-level analysis like finding cancer prevalence or common diagnoses.
    Much more efficient than calling get_patient_conditions for each patient individually.

    Filter supports comma-separated values for OR logic (e.g., 'diabetes,hypertension'
    finds patients with diabetes OR hypertension).
    """
    try:
        # Get patient list
        patient_ids = list_available_patients(limit=patient_limit)

        # Aggregate conditions across all patients
        all_conditions = []
        condition_counts = {}
        patients_with_condition = {}

        # Parse comma-separated filters for OR logic
        filter_terms = []
        if condition_filter:
            filter_terms = [term.strip().lower() for term in condition_filter.split(',')]

        for patient_id in patient_ids:
            bundle = search_fhir("Condition", patient=patient_id, _count=500)
            conditions = extract_conditions(bundle)

            for c in conditions:
                name = c.get("name", "Unknown")

                # Apply filter if specified (OR logic for comma-separated terms)
                if filter_terms:
                    name_lower = name.lower()
                    code_lower = c.get("code", "").lower()
                    # Check if ANY filter term matches
                    if not any(term in name_lower or term in code_lower for term in filter_terms):
                        continue

                # Count occurrences
                if name not in condition_counts:
                    condition_counts[name] = 0
                    patients_with_condition[name] = []
                condition_counts[name] += 1
                if patient_id not in patients_with_condition[name]:
                    patients_with_condition[name].append(patient_id)

                all_conditions.append({
                    "patient_id": patient_id,
                    "condition": name,
                    "code": c.get("code", ""),
                    "status": c.get("clinical_status", "")
                })

        # Sort by frequency
        sorted_conditions = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)

        # Build summary
        top_conditions = []
        for name, count in sorted_conditions[:20]:
            top_conditions.append({
                "condition": name,
                "occurrence_count": count,
                "patient_count": len(patients_with_condition[name]),
                "patient_ids": patients_with_condition[name][:10]  # Limit to first 10 for readability
            })

        return {
            "patients_analyzed": len(patient_ids),
            "total_condition_occurrences": len(all_conditions),
            "unique_conditions": len(condition_counts),
            "filter_applied": condition_filter,
            "most_common_conditions": top_conditions,
            "all_matching_records": all_conditions if condition_filter else []  # Only include all records if filtered
        }
    except Exception as e:
        return {
            "error": str(e),
            "patients_analyzed": 0
        }


@tool(args_schema=ListPatientsInput)
def list_patients(limit: Optional[int] = None) -> dict:
    """
    Lists available patient IDs in the Coherent Data Set.
    Use this to discover what patients are available for analysis.
    Returns patient IDs that can be used with other medical data tools.
    Specify a limit (e.g., 100, 305, 1000) or leave empty for all patients.
    """
    try:
        patient_ids = list_available_patients(limit=limit)
        return {
            "patient_count": len(patient_ids),
            "patients": patient_ids,
            "note": "Use these patient IDs with get_patient_labs, get_vital_signs, etc."
        }
    except Exception as e:
        return {
            "error": str(e),
            "patients": []
        }
