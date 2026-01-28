"""
Unit tests for clinical scoring tools.
Tests calculate_clinical_score and calculate_patient_score functions.
"""

import pytest
from unittest.mock import patch, MagicMock
from medster.tools.clinical.scores import (
    calculate_clinical_score,
    calculate_patient_score,
    calculate_wells_dvt,
    calculate_chadsvasc,
    calculate_curb65,
    calculate_meld,
    calculate_age,
    extract_chadsvasc_params,
)


class TestCalculateAge:
    """Tests for calculate_age helper function."""

    def test_calculate_age_valid_date(self):
        """Test age calculation with valid birth date."""
        # Someone born in 1990 should be around 35-36 in 2026
        age = calculate_age("1990-01-15")
        assert 35 <= age <= 36

    def test_calculate_age_invalid_date(self):
        """Test age calculation with invalid date returns 0."""
        age = calculate_age("invalid-date")
        assert age == 0

    def test_calculate_age_empty_string(self):
        """Test age calculation with empty string returns 0."""
        age = calculate_age("")
        assert age == 0


class TestCalculateWellsDVT:
    """Tests for Wells' DVT score calculation."""

    def test_wells_dvt_low_risk(self):
        """Test low risk DVT score (score <= 0)."""
        params = {"alternative_diagnosis": True}  # -2 points
        result = calculate_wells_dvt(params)
        assert result["score"] <= 0
        assert result["risk_category"] == "Low"
        assert result["dvt_probability"] == "5%"

    def test_wells_dvt_moderate_risk(self):
        """Test moderate risk DVT score (score 1-2)."""
        params = {
            "active_cancer": True,  # +1
            "leg_swelling": True,   # +1
        }
        result = calculate_wells_dvt(params)
        assert result["score"] == 2
        assert result["risk_category"] == "Moderate"
        assert result["dvt_probability"] == "17%"

    def test_wells_dvt_high_risk(self):
        """Test high risk DVT score (score >= 3)."""
        params = {
            "active_cancer": True,           # +1
            "paralysis_or_immobilization": True,  # +1
            "bedridden_or_surgery": True,    # +1
            "localized_tenderness": True,    # +1
        }
        result = calculate_wells_dvt(params)
        assert result["score"] >= 3
        assert result["risk_category"] == "High"
        assert result["dvt_probability"] == "53%"


class TestCalculateCHADSVASc:
    """Tests for CHA2DS2-VASc score calculation."""

    def test_chadsvasc_zero_score(self):
        """Test CHA2DS2-VASc with no risk factors."""
        params = {}
        result = calculate_chadsvasc(params)
        assert result["score"] == 0
        assert result["risk_category"] == "Low"
        assert "No anticoagulation" in result["recommendation"]

    def test_chadsvasc_age_75_plus(self):
        """Test CHA2DS2-VASc with age >= 75 (2 points)."""
        params = {"age_75_or_older": True}
        result = calculate_chadsvasc(params)
        assert result["score"] == 2
        assert result["risk_category"] == "Moderate-High"

    def test_chadsvasc_stroke_history(self):
        """Test CHA2DS2-VASc with stroke/TIA history (2 points)."""
        params = {"stroke_tia": True}
        result = calculate_chadsvasc(params)
        assert result["score"] == 2
        assert result["risk_category"] == "Moderate-High"

    def test_chadsvasc_multiple_factors(self):
        """Test CHA2DS2-VASc with multiple risk factors."""
        params = {
            "chf": True,              # +1
            "hypertension": True,     # +1
            "age_75_or_older": True,  # +2
            "diabetes": True,         # +1
            "vascular_disease": True, # +1
            "female": True,           # +1
        }
        result = calculate_chadsvasc(params)
        assert result["score"] == 7
        assert result["risk_category"] == "Moderate-High"


class TestCalculateCURB65:
    """Tests for CURB-65 pneumonia severity score."""

    def test_curb65_low_risk(self):
        """Test CURB-65 with low risk (score 0-1)."""
        params = {"age_65_or_older": True}  # +1
        result = calculate_curb65(params)
        assert result["score"] == 1
        assert result["risk_category"] == "Low"
        assert "outpatient" in result["recommendation"].lower()

    def test_curb65_high_risk(self):
        """Test CURB-65 with high risk (score 3+)."""
        params = {
            "confusion": True,           # +1
            "urea_elevated": True,       # +1
            "respiratory_rate_30": True, # +1
            "low_blood_pressure": True,  # +1
            "age_65_or_older": True,     # +1
        }
        result = calculate_curb65(params)
        assert result["score"] == 5
        assert result["risk_category"] == "High"
        assert "ICU" in result["recommendation"]


class TestCalculateMELD:
    """Tests for MELD score calculation."""

    def test_meld_low_score(self):
        """Test MELD with normal values."""
        params = {
            "creatinine": 1.0,
            "bilirubin": 1.0,
            "inr": 1.0,
        }
        result = calculate_meld(params)
        assert result["score"] >= 6
        assert result["score"] < 10

    def test_meld_dialysis(self):
        """Test MELD with dialysis (creatinine set to 4)."""
        params = {
            "creatinine": 1.0,
            "bilirubin": 2.0,
            "inr": 1.5,
            "dialysis": True,
        }
        result = calculate_meld(params)
        # With dialysis, creatinine is set to 4, increasing score
        assert result["score"] > 10


class TestCalculateClinicalScore:
    """Tests for the main calculate_clinical_score tool."""

    def test_calculate_clinical_score_wells_dvt(self):
        """Test calculate_clinical_score with Wells DVT."""
        result = calculate_clinical_score.invoke({
            "score_type": "wells_dvt",
            "parameters": {"active_cancer": True, "leg_swelling": True}
        })
        assert result["score_name"] == "Wells' Criteria for DVT"
        assert result["score"] == 2
        assert "disclaimer" in result

    def test_calculate_clinical_score_chadsvasc(self):
        """Test calculate_clinical_score with CHA2DS2-VASc."""
        result = calculate_clinical_score.invoke({
            "score_type": "chadsvasc",
            "parameters": {"chf": True, "hypertension": True}
        })
        assert result["score_name"] == "CHA2DS2-VASc Score"
        assert result["score"] == 2

    def test_calculate_clinical_score_invalid_type(self):
        """Test calculate_clinical_score with invalid score type raises ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            calculate_clinical_score.invoke({
                "score_type": "invalid_score",
                "parameters": {}
            })


class TestExtractCHADSVAScParams:
    """Tests for extract_chadsvasc_params helper."""

    def test_extract_params_with_chf(self):
        """Test extraction with CHF condition."""
        demographics = {"birth_date": "1950-01-01", "gender": "male"}
        conditions = [{"code": "42343007"}]  # CHF SNOMED code
        params = extract_chadsvasc_params(demographics, conditions)
        assert params["chf"] is True
        assert params["age_75_or_older"] is True  # Born 1950, now 76

    def test_extract_params_female(self):
        """Test extraction with female gender."""
        demographics = {"birth_date": "2000-01-01", "gender": "female"}
        conditions = []
        params = extract_chadsvasc_params(demographics, conditions)
        assert params["female"] is True
        assert params["age_75_or_older"] is False


# Run pytest if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
