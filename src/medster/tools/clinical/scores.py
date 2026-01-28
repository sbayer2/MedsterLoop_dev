from langchain.tools import tool
from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

####################################
# Input Schemas
####################################

class ClinicalScoreInput(BaseModel):
    score_type: Literal[
        "wells_dvt", "wells_pe", "chadsvasc", "hasbled",
        "apache_ii", "sofa", "curb65", "meld", "child_pugh"
    ] = Field(description="The clinical scoring system to calculate.")
    parameters: dict = Field(description="Parameters required for the score calculation. Each score has specific required fields.")


class PatientClinicalScoreInput(BaseModel):
    patient_id: str = Field(description="The patient's unique identifier in the system.")
    score_type: Literal["chadsvasc", "wells_dvt", "curb65"] = Field(
        description="The clinical scoring system to calculate. Currently supports: chadsvasc (CHA2DS2-VASc for AFib stroke risk)."
    )


####################################
# SNOMED Code Mappings for CHA2DS2-VASc
####################################

# CHA2DS2-VASc component SNOMED codes
CHADSVASC_SNOMED_CODES = {
    # C - Congestive Heart Failure
    "chf": [
        "42343007",    # Congestive heart failure
        "88805009",    # Chronic congestive heart failure
        "84114007",    # Heart failure
        "85232009",    # Left heart failure
        "10091002",    # High output heart failure
        "417996009",   # Systolic heart failure
        "418304008",   # Diastolic heart failure
        "443253003",   # Chronic systolic heart failure
        "443254009",   # Chronic diastolic heart failure
        "698594003",   # Symptomatic congestive heart failure
    ],
    # H - Hypertension
    "hypertension": [
        "38341003",    # Hypertensive disorder
        "59621000",    # Essential hypertension
        "31992008",    # Secondary hypertension
        "70272006",    # Malignant hypertension
        "1201005",     # Benign hypertension
        "10725009",    # Benign essential hypertension
        "48146000",    # Diastolic hypertension
        "56218007",    # Systolic hypertension
        "194767001",   # Benign hypertensive heart disease
        "194779001",   # Hypertensive heart and renal disease
    ],
    # D - Diabetes Mellitus
    "diabetes": [
        "44054006",    # Diabetes mellitus type 2
        "46635009",    # Diabetes mellitus type 1
        "73211009",    # Diabetes mellitus
        "8801005",     # Secondary diabetes mellitus
        "237599002",   # Insulin treated diabetes mellitus
        "111552007",   # Diabetes mellitus without complication
        "422014003",   # Diabetic on insulin
        "190330002",   # Type 2 diabetes mellitus with hypoglycemia
        "314771006",   # Type 2 diabetes mellitus with hypoglycaemic coma
    ],
    # S2 - Stroke/TIA/Thromboembolism
    "stroke_tia": [
        "230690007",   # Stroke / Cerebrovascular accident
        "266257000",   # Transient ischemic attack
        "195206000",   # Ischemic stroke
        "195210001",   # Hemorrhagic stroke
        "230691006",   # Cerebrovascular accident
        "432504007",   # Cerebral infarction
        "413758000",   # Cardioembolic stroke
        "276219001",   # Lacunar infarction
        "373606000",   # Thromboembolic stroke
        "266253001",   # Transient cerebral ischemia
        "75543006",    # Cerebral embolism
        "723857007",   # Silent micro-hemorrhage of brain
    ],
    # V - Vascular Disease (MI, PAD, Aortic Plaque)
    "vascular_disease": [
        "22298006",    # Myocardial infarction
        "399211009",   # History of myocardial infarction
        "57054005",    # Acute myocardial infarction
        "129574000",   # Old myocardial infarction
        "233970002",   # Coronary artery bypass graft
        "41339005",    # Coronary atherosclerosis
        "399957001",   # Peripheral arterial occlusive disease
        "840580004",   # Peripheral artery disease
        "233970002",   # CABG history
        "233958001",   # Percutaneous coronary intervention
        "443502000",   # Atherosclerosis of aorta
        "128053003",   # Deep venous thrombosis
        "414545008",   # Ischemic heart disease
        "49436004",    # NOT AFib itself - but could indicate vascular burden
    ],
}


def calculate_age(birth_date_str: str) -> int:
    """Calculate age from birth date string (YYYY-MM-DD format)."""
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age
    except Exception:
        return 0


def extract_chadsvasc_params(demographics: dict, conditions: list) -> dict:
    """Extract CHA2DS2-VASc parameters from patient demographics and conditions."""
    params = {
        "chf": False,
        "hypertension": False,
        "age_75_or_older": False,
        "age_65_to_74": False,
        "diabetes": False,
        "stroke_tia": False,
        "vascular_disease": False,
        "female": False,
    }

    # Extract age from demographics
    birth_date = demographics.get("birth_date", "")
    if birth_date:
        age = calculate_age(birth_date)
        if age >= 75:
            params["age_75_or_older"] = True
        elif age >= 65:
            params["age_65_to_74"] = True

    # Extract gender
    gender = demographics.get("gender", "").lower()
    params["female"] = gender == "female"

    # Collect all SNOMED codes from conditions
    patient_codes = set()
    for condition in conditions:
        code = str(condition.get("code", ""))
        if code:
            patient_codes.add(code)

    # Match against CHA2DS2-VASc SNOMED codes
    for component, snomed_codes in CHADSVASC_SNOMED_CODES.items():
        for code in snomed_codes:
            if code in patient_codes:
                params[component] = True
                break

    return params


####################################
# Score Calculation Functions
####################################

def calculate_wells_dvt(params: dict) -> dict:
    """Calculate Wells' Criteria for DVT probability."""
    score = 0

    # Active cancer
    if params.get("active_cancer", False):
        score += 1

    # Paralysis, paresis, or recent plaster immobilization
    if params.get("paralysis_or_immobilization", False):
        score += 1

    # Recently bedridden >3 days or major surgery within 12 weeks
    if params.get("bedridden_or_surgery", False):
        score += 1

    # Localized tenderness along deep venous system
    if params.get("localized_tenderness", False):
        score += 1

    # Entire leg swelling
    if params.get("leg_swelling", False):
        score += 1

    # Calf swelling >3cm compared to asymptomatic leg
    if params.get("calf_swelling_3cm", False):
        score += 1

    # Pitting edema
    if params.get("pitting_edema", False):
        score += 1

    # Collateral superficial veins
    if params.get("collateral_veins", False):
        score += 1

    # Previously documented DVT
    if params.get("previous_dvt", False):
        score += 1

    # Alternative diagnosis at least as likely as DVT
    if params.get("alternative_diagnosis", False):
        score -= 2

    # Interpretation
    if score <= 0:
        risk = "Low"
        probability = "5%"
    elif score <= 2:
        risk = "Moderate"
        probability = "17%"
    else:
        risk = "High"
        probability = "53%"

    return {
        "score_name": "Wells' Criteria for DVT",
        "score": score,
        "risk_category": risk,
        "dvt_probability": probability,
        "recommendation": f"{risk} probability - consider D-dimer and/or ultrasound based on clinical judgment"
    }


def calculate_chadsvasc(params: dict) -> dict:
    """Calculate CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk."""
    score = 0

    # C - Congestive heart failure
    if params.get("chf", False):
        score += 1

    # H - Hypertension
    if params.get("hypertension", False):
        score += 1

    # A2 - Age >= 75
    if params.get("age_75_or_older", False):
        score += 2
    elif params.get("age_65_to_74", False):
        score += 1

    # D - Diabetes mellitus
    if params.get("diabetes", False):
        score += 1

    # S2 - Stroke/TIA/thromboembolism
    if params.get("stroke_tia", False):
        score += 2

    # V - Vascular disease
    if params.get("vascular_disease", False):
        score += 1

    # Sc - Sex category (female)
    if params.get("female", False):
        score += 1

    # Risk interpretation
    if score == 0:
        risk = "Low"
        recommendation = "No anticoagulation recommended"
    elif score == 1:
        risk = "Low-Moderate"
        recommendation = "Consider anticoagulation"
    else:
        risk = "Moderate-High"
        recommendation = "Anticoagulation recommended"

    return {
        "score_name": "CHA2DS2-VASc Score",
        "score": score,
        "risk_category": risk,
        "recommendation": recommendation
    }


def calculate_curb65(params: dict) -> dict:
    """Calculate CURB-65 Score for Pneumonia Severity."""
    score = 0

    # C - Confusion (new)
    if params.get("confusion", False):
        score += 1

    # U - Urea > 7 mmol/L (BUN > 19 mg/dL)
    if params.get("urea_elevated", False):
        score += 1

    # R - Respiratory rate >= 30
    if params.get("respiratory_rate_30", False):
        score += 1

    # B - Blood pressure (SBP < 90 or DBP <= 60)
    if params.get("low_blood_pressure", False):
        score += 1

    # 65 - Age >= 65
    if params.get("age_65_or_older", False):
        score += 1

    # Risk interpretation
    if score <= 1:
        risk = "Low"
        mortality = "1.5%"
        recommendation = "Consider outpatient treatment"
    elif score == 2:
        risk = "Moderate"
        mortality = "9.2%"
        recommendation = "Consider short inpatient stay or closely supervised outpatient"
    else:
        risk = "High"
        mortality = "22%"
        recommendation = "Hospitalize, consider ICU if score 4-5"

    return {
        "score_name": "CURB-65 Pneumonia Severity",
        "score": score,
        "risk_category": risk,
        "30_day_mortality": mortality,
        "recommendation": recommendation
    }


def calculate_meld(params: dict) -> dict:
    """Calculate MELD Score for End-Stage Liver Disease."""
    import math

    # Get values with defaults
    creatinine = max(1.0, min(4.0, params.get("creatinine", 1.0)))
    bilirubin = max(1.0, params.get("bilirubin", 1.0))
    inr = max(1.0, params.get("inr", 1.0))
    dialysis = params.get("dialysis", False)

    # If on dialysis, set creatinine to 4
    if dialysis:
        creatinine = 4.0

    # MELD formula
    meld_score = (
        0.957 * math.log(creatinine) +
        0.378 * math.log(bilirubin) +
        1.120 * math.log(inr) +
        0.643
    ) * 10

    meld_score = round(meld_score)
    meld_score = max(6, min(40, meld_score))

    # Mortality interpretation
    if meld_score < 10:
        mortality_3month = "1.9%"
    elif meld_score < 20:
        mortality_3month = "6.0%"
    elif meld_score < 30:
        mortality_3month = "19.6%"
    elif meld_score < 40:
        mortality_3month = "52.6%"
    else:
        mortality_3month = "71.3%"

    return {
        "score_name": "MELD Score",
        "score": meld_score,
        "3_month_mortality": mortality_3month,
        "note": "Higher scores indicate more urgent need for transplant"
    }


####################################
# Main Tool
####################################

@tool(args_schema=ClinicalScoreInput)
def calculate_clinical_score(
    score_type: str,
    parameters: dict
) -> dict:
    """
    Calculates clinical risk scores including Wells' Criteria, CHA2DS2-VASc,
    CURB-65, MELD, and others. Provides risk stratification and recommendations.
    IMPORTANT: These are decision support tools - always use clinical judgment.
    """
    calculators = {
        "wells_dvt": calculate_wells_dvt,
        "chadsvasc": calculate_chadsvasc,
        "curb65": calculate_curb65,
        "meld": calculate_meld,
    }

    if score_type not in calculators:
        return {
            "error": f"Score type '{score_type}' not implemented",
            "available_scores": list(calculators.keys())
        }

    try:
        result = calculators[score_type](parameters)
        result["disclaimer"] = "Clinical scores are decision support tools. Always use clinical judgment."
        return result
    except Exception as e:
        return {
            "score_type": score_type,
            "error": str(e)
        }


@tool(args_schema=PatientClinicalScoreInput)
def calculate_patient_score(
    patient_id: str,
    score_type: str
) -> dict:
    """
    Calculates clinical risk scores for a specific patient by automatically extracting
    risk factors from their FHIR data (demographics, conditions). Currently supports:
    - chadsvasc: CHA2DS2-VASc stroke risk score for atrial fibrillation patients

    This tool automatically:
    1. Fetches patient demographics (age, gender)
    2. Fetches all patient conditions
    3. Maps SNOMED codes to score components
    4. Calculates the score with proper component attribution

    Use this instead of calculate_clinical_score when you have a patient_id and want
    automatic risk factor extraction from their medical record.
    """
    from medster.tools.medical.api import get_fhir_resource, search_fhir, extract_conditions

    try:
        # Fetch patient demographics
        patient = get_fhir_resource("Patient", patient_id)

        demographics = {
            "patient_id": patient_id,
            "birth_date": patient.get("birthDate", ""),
            "gender": patient.get("gender", ""),
        }

        # Extract name for reporting
        name = "Unknown"
        if "name" in patient and patient["name"]:
            name_entry = patient["name"][0]
            given = " ".join(name_entry.get("given", []))
            family = name_entry.get("family", "")
            name = f"{given} {family}".strip()

        # Fetch patient conditions
        bundle = search_fhir("Condition", patient=patient_id, _count=500)
        conditions = extract_conditions(bundle)

        # Calculate based on score type
        if score_type == "chadsvasc":
            # Extract CHA2DS2-VASc parameters from patient data
            params = extract_chadsvasc_params(demographics, conditions)

            # Calculate score
            score_result = calculate_chadsvasc(params)

            # Calculate age for reporting
            age = calculate_age(demographics.get("birth_date", ""))

            # Build detailed component breakdown
            components_found = []
            if params["chf"]:
                components_found.append("Congestive heart failure (+1)")
            if params["hypertension"]:
                components_found.append("Hypertension (+1)")
            if params["age_75_or_older"]:
                components_found.append(f"Age ≥75 years (age {age}) (+2)")
            elif params["age_65_to_74"]:
                components_found.append(f"Age 65-74 years (age {age}) (+1)")
            if params["diabetes"]:
                components_found.append("Diabetes mellitus (+1)")
            if params["stroke_tia"]:
                components_found.append("Prior stroke/TIA (+2)")
            if params["vascular_disease"]:
                components_found.append("Vascular disease (+1)")
            if params["female"]:
                components_found.append("Female sex (+1)")

            return {
                "patient_id": patient_id,
                "patient_name": name,
                "age": age,
                "gender": demographics.get("gender", "Unknown"),
                "score_name": "CHA2DS2-VASc Score",
                "score": score_result["score"],
                "risk_category": score_result["risk_category"],
                "recommendation": score_result["recommendation"],
                "components_present": components_found,
                "extracted_parameters": params,
                "conditions_analyzed": len(conditions),
                "disclaimer": "Clinical scores are decision support tools. Always use clinical judgment."
            }
        else:
            return {
                "patient_id": patient_id,
                "error": f"Patient-aware calculation for '{score_type}' not yet implemented",
                "supported_scores": ["chadsvasc"]
            }

    except Exception as e:
        return {
            "patient_id": patient_id,
            "score_type": score_type,
            "error": str(e)
        }
