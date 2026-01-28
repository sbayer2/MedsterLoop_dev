from langchain.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
from medster.tools.medical.api import search_fhir, extract_medications

####################################
# Input Schemas
####################################

class MedicationListInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    active_only: bool = Field(default=True, description="If true, only return active medications. If false, include all medications including discontinued.")
    limit: int = Field(default=50, description="Maximum number of medications to retrieve.")


class DrugInteractionsInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    new_medication: Optional[str] = Field(default=None, description="New medication to check for interactions against current medications. Leave empty to check all current medication interactions.")


####################################
# Tools
####################################

@tool(args_schema=MedicationListInput)
def get_medication_list(
    patient_id: str,
    active_only: bool = True,
    limit: int = 50
) -> dict:
    """
    Retrieves a patient's medication list including drug names, dosages,
    frequencies, and routes of administration.
    Useful for medication reconciliation and safety checks.
    """
    # Build FHIR search parameters
    params = {
        "patient": patient_id,
        "_count": limit,
        "_sort": "-authoredon",
    }

    if active_only:
        params["status"] = "active"

    try:
        bundle = search_fhir("MedicationRequest", **params)
        medications = extract_medications(bundle)

        # Also search for MedicationStatement for current medications
        statement_params = {
            "patient": patient_id,
            "_count": limit,
        }
        if active_only:
            statement_params["status"] = "active"

        statement_bundle = search_fhir("MedicationStatement", **statement_params)
        for entry in statement_bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "MedicationStatement":
                med = {
                    "medication": resource.get("medicationCodeableConcept", {}).get("text", "Unknown"),
                    "status": resource.get("status", ""),
                    "effectiveDateTime": resource.get("effectiveDateTime", ""),
                    "dosageInstruction": "",
                }
                if "dosage" in resource and resource["dosage"]:
                    med["dosageInstruction"] = resource["dosage"][0].get("text", "")
                medications.append(med)

        return {
            "patient_id": patient_id,
            "medication_count": len(medications),
            "active_only": active_only,
            "medications": medications
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "medications": []
        }


@tool(args_schema=DrugInteractionsInput)
def check_drug_interactions(
    patient_id: str,
    new_medication: Optional[str] = None
) -> dict:
    """
    Checks for potential drug-drug interactions in a patient's medication list.
    Can optionally check a new medication against current medications.
    Returns potential interactions with severity levels.
    IMPORTANT: This is a simplified check - always verify with clinical pharmacist.
    """
    # First, get current medications
    med_params = {
        "patient": patient_id,
        "status": "active",
        "_count": 100,
    }

    try:
        bundle = search_fhir("MedicationRequest", **med_params)
        medications = extract_medications(bundle)

        current_meds = [med["medication"] for med in medications]

        # Common high-risk drug interaction patterns (simplified)
        # In production, this would call a drug interaction API like DrugBank or Medscape
        high_risk_combinations = [
            (["warfarin"], ["aspirin", "ibuprofen", "naproxen"], "Increased bleeding risk"),
            (["metformin"], ["contrast dye"], "Risk of lactic acidosis"),
            (["ssri", "sertraline", "fluoxetine"], ["tramadol", "maoi"], "Risk of serotonin syndrome"),
            (["ace inhibitor", "lisinopril", "enalapril"], ["potassium", "spironolactone"], "Risk of hyperkalemia"),
            (["digoxin"], ["amiodarone"], "Digoxin toxicity risk"),
            (["statin", "atorvastatin", "simvastatin"], ["gemfibrozil"], "Risk of rhabdomyolysis"),
        ]

        interactions_found = []

        # Check current medications against each other
        for med1_patterns, med2_patterns, warning in high_risk_combinations:
            found_med1 = None
            found_med2 = None

            for med in current_meds:
                med_lower = med.lower()
                for pattern in med1_patterns:
                    if pattern in med_lower:
                        found_med1 = med
                        break
                for pattern in med2_patterns:
                    if pattern in med_lower:
                        found_med2 = med
                        break

            # Check new medication if provided
            if new_medication:
                new_med_lower = new_medication.lower()
                for pattern in med1_patterns:
                    if pattern in new_med_lower and found_med2:
                        interactions_found.append({
                            "medication_1": new_medication,
                            "medication_2": found_med2,
                            "warning": warning,
                            "severity": "HIGH"
                        })
                for pattern in med2_patterns:
                    if pattern in new_med_lower and found_med1:
                        interactions_found.append({
                            "medication_1": found_med1,
                            "medication_2": new_medication,
                            "warning": warning,
                            "severity": "HIGH"
                        })
            elif found_med1 and found_med2:
                interactions_found.append({
                    "medication_1": found_med1,
                    "medication_2": found_med2,
                    "warning": warning,
                    "severity": "HIGH"
                })

        return {
            "patient_id": patient_id,
            "current_medications": current_meds,
            "new_medication_checked": new_medication,
            "interaction_count": len(interactions_found),
            "interactions": interactions_found,
            "disclaimer": "This is a simplified interaction check. Always verify with a clinical pharmacist for comprehensive drug interaction screening."
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "interactions": []
        }
