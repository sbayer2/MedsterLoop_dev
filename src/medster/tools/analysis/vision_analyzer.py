"""
Vision analysis tool for medical images using Claude's vision API.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from medster.model import call_llm
from medster.tools.analysis.primitives import (
    load_ecg_image,
    load_dicom_image,
    load_dicom_image_by_filename,
    find_patient_images,
    get_dicom_metadata,
    get_dicom_metadata_from_path,
    scan_dicom_directory,
    analyze_ecg_for_rhythm
)


class PatientDICOMAnalysisInput(BaseModel):
    """Input schema for patient DICOM analysis."""

    patient_id: str = Field(
        description="Patient UUID to analyze DICOM images for"
    )
    clinical_question: str = Field(
        default="Analyze this medical image and describe any abnormal findings, masses, hemorrhage, or other clinically significant observations",
        description="Specific clinical question about the image"
    )
    image_index: int = Field(
        default=0,
        description="Which image to analyze if patient has multiple (0 for first)"
    )
    clinical_context: str = Field(
        default="",
        description="Optional clinical context (e.g., 'Patient with history of stroke')"
    )


class DICOMFileAnalysisInput(BaseModel):
    """Input schema for analyzing a specific DICOM file."""

    filename: str = Field(
        description="DICOM filename to analyze (from scan_dicom_directory or find_patient_images)"
    )
    clinical_question: str = Field(
        default="Analyze this medical image and describe any abnormal findings",
        description="Specific clinical question about the image"
    )
    clinical_context: str = Field(
        default="",
        description="Optional clinical context"
    )


@tool(args_schema=PatientDICOMAnalysisInput)
def analyze_patient_dicom(
    patient_id: str,
    clinical_question: str = "Analyze this medical image and describe any abnormal findings, masses, hemorrhage, or other clinically significant observations",
    image_index: int = 0,
    clinical_context: str = ""
) -> dict:
    """
    Analyze a patient's DICOM image using Claude's vision API.

    This is the RECOMMENDED tool for DICOM/MRI/CT analysis. It:
    1. Finds the patient's DICOM files
    2. Loads and converts the image (handles GCS automatically)
    3. Gets metadata (modality, body part, etc.)
    4. Sends to Claude vision API for analysis

    Use this instead of generate_and_run_analysis for simple DICOM queries.
    Much faster and more reliable than code generation.

    Returns structured analysis with findings and metadata.
    """
    try:
        # Find patient's images
        image_info = find_patient_images(patient_id)

        if image_info.get("error"):
            return {
                "status": "error",
                "patient_id": patient_id,
                "error": image_info["error"]
            }

        if image_info.get("dicom_count", 0) == 0:
            return {
                "status": "error",
                "patient_id": patient_id,
                "error": "No DICOM images found for this patient"
            }

        # Get metadata
        metadata = get_dicom_metadata(patient_id, image_index)

        # Load image as base64 PNG
        image_base64 = load_dicom_image(patient_id, image_index)

        if not image_base64:
            return {
                "status": "error",
                "patient_id": patient_id,
                "error": "Failed to load DICOM image - conversion failed",
                "metadata": metadata
            }

        # Build prompt
        context_str = f"\nClinical context: {clinical_context}" if clinical_context else ""
        modality = metadata.get("modality", "Unknown")
        body_part = metadata.get("body_part", "Unknown")
        study_desc = metadata.get("study_description", "Unknown")

        prompt = f"""Analyze this medical image for clinical decision support.

Patient ID: {patient_id}
Modality: {modality}
Body Part: {body_part}
Study Description: {study_desc}{context_str}

Clinical Question: {clinical_question}

Provide:
1. Image quality assessment
2. Key anatomical structures visible
3. Abnormal findings (masses, hemorrhage, fractures, etc.)
4. Direct answer to the clinical question
5. Any critical findings requiring immediate attention

Be specific and clinically relevant."""

        # Call vision API
        response = call_llm(
            prompt=prompt,
            images=[image_base64],
            model="claude-sonnet-4.5"
        )

        analysis_text = response.content if hasattr(response, 'content') else str(response)

        return {
            "status": "success",
            "patient_id": patient_id,
            "image_index": image_index,
            "total_images": image_info.get("dicom_count", 0),
            "modality": modality,
            "body_part": body_part,
            "study_description": study_desc,
            "clinical_question": clinical_question,
            "vision_analysis": analysis_text,
            "metadata": metadata
        }

    except Exception as e:
        return {
            "status": "error",
            "patient_id": patient_id,
            "error": f"DICOM analysis failed: {str(e)}"
        }


@tool(args_schema=DICOMFileAnalysisInput)
def analyze_dicom_file(
    filename: str,
    clinical_question: str = "Analyze this medical image and describe any abnormal findings",
    clinical_context: str = ""
) -> dict:
    """
    Analyze a specific DICOM file by filename.

    Use this when you have a specific filename from scan_dicom_directory()
    or find_patient_images() and want to analyze it directly.

    Faster than analyze_patient_dicom when you already know the filename.
    """
    try:
        # Get metadata
        metadata = get_dicom_metadata_from_path(filename)

        if metadata.get("error"):
            return {
                "status": "error",
                "filename": filename,
                "error": metadata["error"]
            }

        # Load image
        image_base64 = load_dicom_image_by_filename(filename)

        if not image_base64:
            return {
                "status": "error",
                "filename": filename,
                "error": "Failed to load DICOM image"
            }

        # Build prompt
        context_str = f"\nClinical context: {clinical_context}" if clinical_context else ""
        modality = metadata.get("modality", "Unknown")
        body_part = metadata.get("body_part", "Unknown")

        prompt = f"""Analyze this medical image.

Filename: {filename}
Modality: {modality}
Body Part: {body_part}{context_str}

{clinical_question}

Provide specific clinical findings."""

        response = call_llm(
            prompt=prompt,
            images=[image_base64],
            model="claude-sonnet-4.5"
        )

        analysis_text = response.content if hasattr(response, 'content') else str(response)

        return {
            "status": "success",
            "filename": filename,
            "modality": modality,
            "body_part": body_part,
            "vision_analysis": analysis_text,
            "metadata": metadata
        }

    except Exception as e:
        return {
            "status": "error",
            "filename": filename,
            "error": f"Analysis failed: {str(e)}"
        }


class PatientECGAnalysisInput(BaseModel):
    """Input schema for patient ECG analysis."""

    patient_id: str = Field(
        description="Patient UUID to analyze ECG for"
    )
    clinical_question: str = Field(
        default="Analyze the ECG tracing and describe the cardiac rhythm including rate, regularity, P waves, QRS complexes, and any abnormalities",
        description="Specific clinical question about the ECG"
    )
    clinical_context: str = Field(
        default="",
        description="Optional clinical context (e.g., 'Patient with hypertension and diabetes')"
    )


@tool(args_schema=PatientECGAnalysisInput)
def analyze_patient_ecg(
    patient_id: str,
    clinical_question: str = "Analyze the ECG tracing and describe the cardiac rhythm including rate, regularity, P waves, QRS complexes, and any abnormalities",
    clinical_context: str = ""
) -> dict:
    """
    Analyze a patient's ECG image using the LLM's vision API.

    This tool takes a patient_id and automatically loads their ECG image,
    then performs vision analysis to answer clinical questions about the ECG.

    Use this when you have a patient ID and want to visually analyze their ECG tracing.
    The tool handles image loading internally - you don't need base64 data.

    Returns structured analysis including rhythm classification, findings, and clinical significance.
    """
    try:
        # Use the structured ECG analysis primitive
        result = analyze_ecg_for_rhythm(patient_id, clinical_context)

        if not result.get("ecg_available", False):
            return {
                "status": "error",
                "patient_id": patient_id,
                "error": "No ECG image available for this patient"
            }

        # If custom question, do additional analysis
        if "atrial fibrillation" not in clinical_question.lower() and "rhythm" not in clinical_question.lower():
            # Load image for custom analysis
            ecg_image = load_ecg_image(patient_id)
            if ecg_image:
                context_str = f" (Clinical context: {clinical_context})" if clinical_context else ""
                prompt = f"""Analyze this ECG tracing for patient {patient_id}{context_str}.

{clinical_question}

Provide a detailed analysis with specific findings."""

                response = call_llm(
                    prompt=prompt,
                    images=[ecg_image],
                    model="claude-sonnet-4.5"  # Use Claude Sonnet 4.5 for vision
                )
                custom_analysis = response.content if hasattr(response, 'content') else str(response)
                result["custom_analysis"] = custom_analysis

        return {
            "status": "success",
            "patient_id": patient_id,
            "ecg_available": True,
            "rhythm": result.get("rhythm", "Unknown"),
            "afib_detected": result.get("afib_detected", False),
            "rr_intervals": result.get("rr_intervals", "Unknown"),
            "p_waves": result.get("p_waves", "Unknown"),
            "baseline": result.get("baseline", "Unknown"),
            "confidence": result.get("confidence", "Unknown"),
            "clinical_significance": result.get("clinical_significance", ""),
            "clinical_context": clinical_context,
            "detailed_analysis": result.get("raw_analysis", "")
        }

    except Exception as e:
        return {
            "status": "error",
            "patient_id": patient_id,
            "error": f"ECG analysis failed: {str(e)}"
        }


class VisionAnalysisInput(BaseModel):
    """Input schema for vision analysis."""

    analysis_prompt: str = Field(
        description="Specific clinical question to answer about the images (e.g., 'Does this ECG show atrial fibrillation pattern?', 'Identify any masses or hemorrhage in this brain MRI')"
    )
    image_data: List[Dict[str, Any]] = Field(
        description="List of image objects, each containing 'image_base64' (required), 'patient_id' (optional), 'modality' (optional), 'context' (optional)"
    )
    max_images: int = Field(
        default=3,
        description="Maximum number of images to analyze in a single call (for token efficiency)"
    )


@tool(args_schema=VisionAnalysisInput)
def analyze_medical_images(
    analysis_prompt: str,
    image_data: List[Dict[str, Any]],
    max_images: int = 3
) -> dict:
    """
    Analyze medical images using the LLM's vision API.

    Use this tool when you have loaded base64-encoded images (DICOM, ECG, etc.)
    and need to analyze them for clinical findings.

    The tool accepts a list of image objects with:
    - image_base64 (required): Base64-encoded PNG image string
    - patient_id (optional): Patient identifier
    - modality (optional): Imaging modality (MRI, CT, ECG, etc.)
    - context (optional): Clinical context for the image

    Example usage after generate_and_run_analysis loads images:
    - analysis_prompt: "Analyze these ECG waveforms for atrial fibrillation pattern"
    - image_data: List of dicts with patient_id, image_base64, and modality fields

    Returns a structured analysis with findings for each image.
    """
    try:
        # Limit images for token efficiency
        images_to_analyze = image_data[:max_images]

        # Extract base64 images
        base64_images = []
        patient_context = []

        for idx, img in enumerate(images_to_analyze):
            if "image_base64" not in img:
                continue

            base64_images.append(img["image_base64"])

            # Build context for each image
            context_parts = [f"Image {idx + 1}"]
            if "patient_id" in img:
                context_parts.append(f"Patient: {img['patient_id']}")
            if "modality" in img:
                context_parts.append(f"Modality: {img['modality']}")
            if "context" in img:
                context_parts.append(img["context"])

            patient_context.append(" | ".join(context_parts))

        if not base64_images:
            return {
                "status": "error",
                "error": "No valid images found in image_data (missing 'image_base64' key)"
            }

        # Build prompt with context
        full_prompt = f"""You are analyzing medical images for clinical decision support.

{analysis_prompt}

Context for each image:
{chr(10).join(f"- {ctx}" for ctx in patient_context)}

For each image, provide:
1. Patient ID (if provided)
2. Key visual findings
3. Direct answer to the clinical question
4. Any critical findings requiring immediate attention

Format your response as structured findings for each image."""

        # Call LLM vision API with Claude Sonnet 4.5
        response = call_llm(
            prompt=full_prompt,
            images=base64_images,
            model="claude-sonnet-4.5"
        )

        # Extract text content from response
        analysis_text = response.content if hasattr(response, 'content') else str(response)

        return {
            "status": "success",
            "images_analyzed": len(base64_images),
            "clinical_question": analysis_prompt,
            "vision_analysis": analysis_text,
            "patient_contexts": patient_context
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Vision analysis failed: {str(e)}",
            "images_attempted": len(image_data)
        }
