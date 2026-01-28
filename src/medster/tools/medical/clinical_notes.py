from langchain.tools import tool
from typing import Literal, Optional
from pydantic import BaseModel, Field
from medster.tools.medical.api import search_fhir

####################################
# Input Schemas
####################################

class ClinicalNotesInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    note_type: Optional[str] = Field(default=None, description="Type of clinical note (e.g., 'progress-note', 'discharge-summary', 'consultation', 'history-physical', 'operative-note'). Leave empty for all notes.")
    specialty: Optional[str] = Field(default=None, description="Filter by specialty (e.g., 'cardiology', 'pulmonology', 'surgery'). Leave empty for all specialties.")
    limit: int = Field(default=10, description="Maximum number of notes to retrieve.")
    date_start: Optional[str] = Field(default=None, description="Filter for notes created after this date (YYYY-MM-DD).")
    date_end: Optional[str] = Field(default=None, description="Filter for notes created before this date (YYYY-MM-DD).")


class SOAPNotesInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    encounter_id: Optional[str] = Field(default=None, description="Specific encounter/visit ID. Leave empty for most recent.")
    limit: int = Field(default=5, description="Maximum number of SOAP notes to retrieve.")


class DischargeSummaryInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    admission_id: Optional[str] = Field(default=None, description="Specific admission/encounter ID. Leave empty for most recent discharge.")


####################################
# Tools
####################################

@tool(args_schema=ClinicalNotesInput)
def get_clinical_notes(
    patient_id: str,
    note_type: Optional[str] = None,
    specialty: Optional[str] = None,
    limit: int = 10,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None
) -> dict:
    """
    Retrieves clinical documentation for a patient including progress notes,
    consultation notes, and procedure notes.
    Useful for understanding clinical history and provider assessments.
    """
    # Build FHIR search parameters for DocumentReference
    params = {
        "patient": patient_id,
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

    # Add note type filter
    if note_type:
        # Map common note types to LOINC document types
        note_type_map = {
            "progress-note": "11506-3",
            "discharge-summary": "18842-5",
            "consultation": "11488-4",
            "history-physical": "34117-2",
            "operative-note": "11504-8",
        }
        if note_type in note_type_map:
            params["type"] = note_type_map[note_type]
        else:
            params["type:text"] = note_type

    try:
        bundle = search_fhir("DocumentReference", **params)
        notes = []

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "DocumentReference":
                note = {
                    "id": resource.get("id", ""),
                    "date": resource.get("date", ""),
                    "type": resource.get("type", {}).get("text", "Clinical Note"),
                    "status": resource.get("status", ""),
                    "content": "",
                    "author": "",
                }

                # Extract content
                if "content" in resource and resource["content"]:
                    content_item = resource["content"][0]
                    if "attachment" in content_item:
                        attachment = content_item["attachment"]
                        if "data" in attachment:
                            import base64
                            note["content"] = base64.b64decode(attachment["data"]).decode("utf-8", errors="ignore")
                        elif "url" in attachment:
                            note["content"] = f"[Content available at: {attachment['url']}]"

                # Extract author
                if "author" in resource and resource["author"]:
                    author_ref = resource["author"][0]
                    note["author"] = author_ref.get("display", author_ref.get("reference", ""))

                notes.append(note)

        return {
            "patient_id": patient_id,
            "note_count": len(notes),
            "notes": notes
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "notes": []
        }


@tool(args_schema=SOAPNotesInput)
def get_soap_notes(
    patient_id: str,
    encounter_id: Optional[str] = None,
    limit: int = 5
) -> dict:
    """
    Retrieves SOAP (Subjective, Objective, Assessment, Plan) formatted notes.
    These are standard clinical progress notes documenting patient encounters.
    Useful for understanding the clinical reasoning and treatment plans.
    """
    params = {
        "patient": patient_id,
        "type": "11506-3",  # LOINC for progress note
        "_count": limit,
        "_sort": "-date",
    }

    if encounter_id:
        params["encounter"] = encounter_id

    try:
        bundle = search_fhir("DocumentReference", **params)
        soap_notes = []

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "DocumentReference":
                note = {
                    "id": resource.get("id", ""),
                    "date": resource.get("date", ""),
                    "author": "",
                    "subjective": "",
                    "objective": "",
                    "assessment": "",
                    "plan": "",
                    "full_text": "",
                }

                # Extract author
                if "author" in resource and resource["author"]:
                    note["author"] = resource["author"][0].get("display", "")

                # Extract content and try to parse SOAP structure
                if "content" in resource and resource["content"]:
                    content_item = resource["content"][0]
                    if "attachment" in content_item and "data" in content_item["attachment"]:
                        import base64
                        text = base64.b64decode(content_item["attachment"]["data"]).decode("utf-8", errors="ignore")
                        note["full_text"] = text

                        # Simple parsing for SOAP sections (can be enhanced)
                        text_lower = text.lower()
                        if "subjective:" in text_lower:
                            note["subjective"] = "[Subjective section found]"
                        if "objective:" in text_lower:
                            note["objective"] = "[Objective section found]"
                        if "assessment:" in text_lower:
                            note["assessment"] = "[Assessment section found]"
                        if "plan:" in text_lower:
                            note["plan"] = "[Plan section found]"

                soap_notes.append(note)

        return {
            "patient_id": patient_id,
            "note_count": len(soap_notes),
            "soap_notes": soap_notes
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "soap_notes": []
        }


@tool(args_schema=DischargeSummaryInput)
def get_discharge_summary(
    patient_id: str,
    admission_id: Optional[str] = None
) -> dict:
    """
    Retrieves the discharge summary for a hospitalization.
    Contains admission diagnosis, hospital course, discharge diagnosis,
    discharge medications, and follow-up instructions.
    Useful for understanding the full context of a hospital stay.
    """
    params = {
        "patient": patient_id,
        "type": "18842-5",  # LOINC for discharge summary
        "_count": 1,
        "_sort": "-date",
    }

    if admission_id:
        params["encounter"] = admission_id

    try:
        bundle = search_fhir("DocumentReference", **params)
        discharge_summary = None

        entries = bundle.get("entry", [])
        if entries:
            resource = entries[0].get("resource", {})
            discharge_summary = {
                "id": resource.get("id", ""),
                "date": resource.get("date", ""),
                "author": "",
                "admission_date": "",
                "discharge_date": "",
                "admission_diagnosis": "",
                "discharge_diagnosis": "",
                "hospital_course": "",
                "discharge_medications": "",
                "follow_up": "",
                "full_text": "",
            }

            # Extract author
            if "author" in resource and resource["author"]:
                discharge_summary["author"] = resource["author"][0].get("display", "")

            # Extract content
            if "content" in resource and resource["content"]:
                content_item = resource["content"][0]
                if "attachment" in content_item and "data" in content_item["attachment"]:
                    import base64
                    text = base64.b64decode(content_item["attachment"]["data"]).decode("utf-8", errors="ignore")
                    discharge_summary["full_text"] = text

        return {
            "patient_id": patient_id,
            "discharge_summary": discharge_summary
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "discharge_summary": None
        }
