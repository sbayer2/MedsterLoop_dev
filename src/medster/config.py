"""
Configuration module for Medster.

Loads environment variables and provides centralized access to configuration values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Coherent Data Set paths
COHERENT_FHIR_PATH = os.getenv("COHERENT_DATA_PATH", "./coherent_data/fhir")
COHERENT_DICOM_PATH = os.getenv("COHERENT_DICOM_PATH", "./coherent_data/dicom")
COHERENT_DNA_PATH = os.getenv("COHERENT_DNA_PATH", "./coherent_data/dna")
COHERENT_CSV_PATH = os.getenv("COHERENT_CSV_PATH", "./coherent_data/csv")

# MCP Server configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
MCP_API_KEY = os.getenv("MCP_API_KEY")
MCP_DEBUG = os.getenv("MCP_DEBUG", "false").lower() == "true"

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def get_absolute_path(relative_path: str) -> Path:
    """
    Convert relative path to absolute path.

    Args:
        relative_path: Relative path string

    Returns:
        Absolute Path object
    """
    if os.path.isabs(relative_path):
        return Path(relative_path)

    # Use current working directory as base for relative paths
    # This ensures paths resolve correctly when running from project directory
    # regardless of where the package is installed
    return (Path(os.getcwd()) / relative_path).resolve()


# Convert all Coherent Data paths to absolute paths
COHERENT_FHIR_PATH_ABS = get_absolute_path(COHERENT_FHIR_PATH)
COHERENT_DICOM_PATH_ABS = get_absolute_path(COHERENT_DICOM_PATH)
COHERENT_DNA_PATH_ABS = get_absolute_path(COHERENT_DNA_PATH)
COHERENT_CSV_PATH_ABS = get_absolute_path(COHERENT_CSV_PATH)


def validate_paths():
    """
    Validate that all required data paths exist.

    Raises:
        FileNotFoundError: If required paths don't exist
    """
    paths_to_check = [
        ("FHIR", COHERENT_FHIR_PATH_ABS),
        ("DICOM", COHERENT_DICOM_PATH_ABS),
        ("DNA", COHERENT_DNA_PATH_ABS),
        ("CSV", COHERENT_CSV_PATH_ABS),
    ]

    missing_paths = []
    for name, path in paths_to_check:
        if not path.exists():
            missing_paths.append(f"{name}: {path}")

    if missing_paths:
        raise FileNotFoundError(
            f"Required Coherent Data Set paths not found:\n" +
            "\n".join(f"  - {p}" for p in missing_paths) +
            "\n\nPlease check your .env configuration and ensure the Coherent Data Set is extracted."
        )
