from langchain.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
from medster.tools.medical.api import search_fhir

####################################
# Input Schemas
####################################

class RadiologyReportsInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    modality: Optional[str] = Field(default=None, description="Imaging modality (e.g., 'XR' for X-ray, 'CT', 'MRI', 'US' for ultrasound, 'NM' for nuclear medicine). Leave empty for all modalities.")
    body_site: Optional[str] = Field(default=None, description="Body site imaged (e.g., 'chest', 'abdomen', 'head', 'spine'). Leave empty for all sites.")
    limit: int = Field(default=10, description="Maximum number of reports to retrieve.")
    date_start: Optional[str] = Field(default=None, description="Filter for studies performed after this date (YYYY-MM-DD).")
    date_end: Optional[str] = Field(default=None, description="Filter for studies performed before this date (YYYY-MM-DD).")


####################################
# Tools
####################################

@tool(args_schema=RadiologyReportsInput)
def get_radiology_reports(
    patient_id: str,
    modality: Optional[str] = None,
    body_site: Optional[str] = None,
    limit: int = 10,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None
) -> dict:
    """
    Retrieves radiology/imaging reports for a patient including X-rays, CT scans,
    MRIs, and ultrasounds with their interpretations and findings.
    Useful for reviewing imaging findings and radiologist impressions.
    """
    # Build FHIR search parameters for DiagnosticReport (imaging)
    params = {
        "patient": patient_id,
        "category": "imaging",
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

    try:
        bundle = search_fhir("DiagnosticReport", **params)
        reports = []

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "DiagnosticReport":
                report = {
                    "id": resource.get("id", ""),
                    "effectiveDateTime": resource.get("effectiveDateTime", ""),
                    "status": resource.get("status", ""),
                    "code": resource.get("code", {}).get("text", "Imaging Study"),
                    "conclusion": resource.get("conclusion", ""),
                    "performer": "",
                    "findings": [],
                }

                # Extract performer (radiologist)
                if "performer" in resource and resource["performer"]:
                    report["performer"] = resource["performer"][0].get("display", "")

                # Extract coded diagnoses if available
                if "codedDiagnosis" in resource:
                    for diag in resource["codedDiagnosis"]:
                        report["findings"].append(diag.get("text", ""))

                # Filter by modality if specified
                if modality:
                    code_text = report["code"].lower()
                    modality_lower = modality.lower()
                    if modality_lower not in code_text:
                        continue

                # Filter by body site if specified
                if body_site:
                    code_text = report["code"].lower()
                    conclusion_text = report["conclusion"].lower()
                    body_site_lower = body_site.lower()
                    if body_site_lower not in code_text and body_site_lower not in conclusion_text:
                        continue

                reports.append(report)

        return {
            "patient_id": patient_id,
            "report_count": len(reports),
            "modality_filter": modality,
            "body_site_filter": body_site,
            "reports": reports
        }
    except Exception as e:
        return {
            "patient_id": patient_id,
            "error": str(e),
            "reports": []
        }
