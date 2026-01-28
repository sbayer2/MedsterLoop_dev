"""
Google Cloud Storage client for Coherent Data Set access.

Provides unified access to FHIR bundles, DICOM files, and CSV data
whether stored locally or in GCS.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_COHERENT_BUCKET", "medster-coherent-data")
USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"

# Local fallback paths
LOCAL_FHIR_PATH = os.getenv("COHERENT_DATA_PATH", "./coherent_data/fhir")
LOCAL_DICOM_PATH = os.getenv("COHERENT_DICOM_PATH", "./coherent_data/dicom")
LOCAL_CSV_PATH = os.getenv("COHERENT_CSV_PATH", "./coherent_data/csv")
LOCAL_DNA_PATH = os.getenv("COHERENT_DNA_PATH", "./coherent_data/dna")

# Lazy-loaded GCS client
_gcs_client = None
_gcs_bucket = None

# In-memory cache for frequently accessed data
_patient_cache: Dict[str, dict] = {}
_patient_list_cache: Optional[List[str]] = None
_csv_cache: Dict[str, Any] = {}


def get_gcs_client():
    """Get or create GCS client (lazy initialization)."""
    global _gcs_client, _gcs_bucket

    if _gcs_client is None and USE_GCS:
        try:
            from google.cloud import storage
            _gcs_client = storage.Client()
            _gcs_bucket = _gcs_client.bucket(GCS_BUCKET_NAME)
            logger.info(f"GCS client initialized for bucket: {GCS_BUCKET_NAME}")
        except ImportError:
            logger.warning("google-cloud-storage not installed. Falling back to local storage.")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")

    return _gcs_client, _gcs_bucket


def load_patient_bundle_gcs(patient_id: str) -> Optional[dict]:
    """
    Load a patient's FHIR bundle from GCS.

    Args:
        patient_id: Patient ID or partial filename match

    Returns:
        FHIR Bundle dict or None if not found
    """
    # Check cache first
    if patient_id in _patient_cache:
        return _patient_cache[patient_id]

    client, bucket = get_gcs_client()
    if not bucket:
        return None

    try:
        # List blobs to find matching patient file
        blobs = list(bucket.list_blobs(prefix="fhir/", max_results=2000))

        for blob in blobs:
            if patient_id in blob.name and blob.name.endswith('.json'):
                content = blob.download_as_text()
                bundle = json.loads(content)
                _patient_cache[patient_id] = bundle
                return bundle

        return None
    except Exception as e:
        logger.error(f"Error loading patient {patient_id} from GCS: {e}")
        return None


def load_patient_bundle_local(patient_id: str) -> Optional[dict]:
    """
    Load a patient's FHIR bundle from local filesystem.

    Args:
        patient_id: Patient ID or partial filename match

    Returns:
        FHIR Bundle dict or None if not found
    """
    # Check cache first
    if patient_id in _patient_cache:
        return _patient_cache[patient_id]

    data_path = Path(LOCAL_FHIR_PATH)
    if not data_path.exists():
        return None

    # Try different matching patterns
    patterns = [
        f"{patient_id}.json",
        f"*{patient_id}*.json",
    ]

    for pattern in patterns:
        matches = list(data_path.glob(pattern))
        if matches:
            with open(matches[0], 'r') as f:
                bundle = json.load(f)
                _patient_cache[patient_id] = bundle
                return bundle

    return None


def load_patient_bundle(patient_id: str) -> Optional[dict]:
    """
    Load a patient's FHIR bundle (auto-selects GCS or local).

    Args:
        patient_id: Patient ID or partial filename match

    Returns:
        FHIR Bundle dict or None if not found
    """
    if USE_GCS:
        result = load_patient_bundle_gcs(patient_id)
        if result:
            return result
        # Fall back to local if GCS fails
        logger.warning(f"GCS load failed for {patient_id}, trying local")

    return load_patient_bundle_local(patient_id)


def list_patients_gcs(limit: Optional[int] = None) -> List[str]:
    """List available patient IDs from GCS."""
    global _patient_list_cache

    if _patient_list_cache is not None:
        return _patient_list_cache[:limit] if limit else _patient_list_cache

    client, bucket = get_gcs_client()
    if not bucket:
        return []

    try:
        patient_ids = []
        blobs = bucket.list_blobs(prefix="fhir/")

        for blob in blobs:
            if blob.name.endswith('.json'):
                # Extract patient ID from filename
                filename = blob.name.split('/')[-1].replace('.json', '')
                patient_ids.append(filename)

                if limit and len(patient_ids) >= limit:
                    break

        # Cache full list if we retrieved everything
        if not limit:
            _patient_list_cache = patient_ids

        return patient_ids
    except Exception as e:
        logger.error(f"Error listing patients from GCS: {e}")
        return []


def list_patients_local(limit: Optional[int] = None) -> List[str]:
    """List available patient IDs from local filesystem."""
    global _patient_list_cache

    if _patient_list_cache is not None:
        return _patient_list_cache[:limit] if limit else _patient_list_cache

    data_path = Path(LOCAL_FHIR_PATH)
    if not data_path.exists():
        return []

    patient_ids = []
    for json_file in data_path.glob("*.json"):
        patient_ids.append(json_file.stem)
        if limit and len(patient_ids) >= limit:
            break

    # Cache full list
    if not limit:
        _patient_list_cache = patient_ids

    return patient_ids


def list_patients(limit: Optional[int] = None) -> List[str]:
    """
    List available patient IDs (auto-selects GCS or local).

    Args:
        limit: Maximum number of patients to return

    Returns:
        List of patient IDs/filenames
    """
    if USE_GCS:
        result = list_patients_gcs(limit)
        if result:
            return result

    return list_patients_local(limit)


def load_csv_file_gcs(filename: str) -> Optional[str]:
    """Load a CSV file from GCS."""
    client, bucket = get_gcs_client()
    if not bucket:
        return None

    try:
        # Handle nested path from upload
        blob = bucket.blob(f"csv/csv/{filename}")
        if not blob.exists():
            blob = bucket.blob(f"csv/{filename}")

        return blob.download_as_text()
    except Exception as e:
        logger.error(f"Error loading CSV {filename} from GCS: {e}")
        return None


def load_csv_file_local(filename: str) -> Optional[str]:
    """Load a CSV file from local filesystem."""
    csv_path = Path(LOCAL_CSV_PATH) / filename
    if csv_path.exists():
        return csv_path.read_text()
    return None


def load_csv_file(filename: str) -> Optional[str]:
    """
    Load a CSV file (auto-selects GCS or local).

    Args:
        filename: CSV filename (e.g., 'patients.csv')

    Returns:
        CSV content as string or None
    """
    # Check cache
    if filename in _csv_cache:
        return _csv_cache[filename]

    content = None
    if USE_GCS:
        content = load_csv_file_gcs(filename)

    if content is None:
        content = load_csv_file_local(filename)

    if content:
        _csv_cache[filename] = content

    return content


def load_dicom_file_gcs(filename: str) -> Optional[bytes]:
    """Load a DICOM file from GCS."""
    client, bucket = get_gcs_client()
    if not bucket:
        return None

    try:
        # Handle nested path from upload
        blob = bucket.blob(f"dicom/{filename}")
        return blob.download_as_bytes()
    except Exception as e:
        logger.error(f"Error loading DICOM {filename} from GCS: {e}")
        return None


def load_dicom_file_local(filename: str) -> Optional[bytes]:
    """Load a DICOM file from local filesystem."""
    dicom_path = Path(LOCAL_DICOM_PATH) / filename
    if dicom_path.exists():
        return dicom_path.read_bytes()
    return None


def load_dicom_file(filename: str) -> Optional[bytes]:
    """
    Load a DICOM file (auto-selects GCS or local).

    Args:
        filename: DICOM filename

    Returns:
        DICOM file bytes or None
    """
    if USE_GCS:
        result = load_dicom_file_gcs(filename)
        if result:
            return result

    return load_dicom_file_local(filename)


def list_dicom_files_gcs(limit: Optional[int] = None) -> List[str]:
    """List DICOM files in GCS."""
    client, bucket = get_gcs_client()
    if not bucket:
        return []

    try:
        files = []
        blobs = bucket.list_blobs(prefix="dicom/")

        for blob in blobs:
            if blob.name.endswith('.dcm'):
                filename = blob.name.split('/')[-1]
                files.append(filename)

                if limit and len(files) >= limit:
                    break

        return files
    except Exception as e:
        logger.error(f"Error listing DICOM files from GCS: {e}")
        return []


def list_dicom_files_local(limit: Optional[int] = None) -> List[str]:
    """List DICOM files in local filesystem."""
    dicom_path = Path(LOCAL_DICOM_PATH)
    if not dicom_path.exists():
        return []

    files = []
    for dcm_file in dicom_path.glob("*.dcm"):
        files.append(dcm_file.name)
        if limit and len(files) >= limit:
            break

    return files


def list_dicom_files(limit: Optional[int] = None) -> List[str]:
    """
    List available DICOM files (auto-selects GCS or local).

    Args:
        limit: Maximum number of files to return

    Returns:
        List of DICOM filenames
    """
    if USE_GCS:
        result = list_dicom_files_gcs(limit)
        if result:
            return result

    return list_dicom_files_local(limit)


def get_dicom_metadata_from_gcs(filename: str) -> Dict[str, Any]:
    """
    Get DICOM metadata by downloading file from GCS and parsing it.

    Args:
        filename: DICOM filename

    Returns:
        Dictionary with DICOM metadata
    """
    try:
        import pydicom
        from io import BytesIO

        dicom_bytes = load_dicom_file(filename)
        if not dicom_bytes:
            return {"error": f"DICOM file not found: {filename}"}

        # Parse DICOM from bytes
        ds = pydicom.dcmread(BytesIO(dicom_bytes))

        return {
            "filename": filename,
            "modality": getattr(ds, 'Modality', 'Unknown'),
            "study_description": getattr(ds, 'StudyDescription', 'Unknown'),
            "body_part": getattr(ds, 'BodyPartExamined', 'Unknown'),
            "patient_name": str(getattr(ds, 'PatientName', 'Unknown')),
            "patient_id": getattr(ds, 'PatientID', 'Unknown'),
            "study_date": getattr(ds, 'StudyDate', 'Unknown'),
            "series_description": getattr(ds, 'SeriesDescription', 'Unknown'),
            "rows": getattr(ds, 'Rows', None),
            "columns": getattr(ds, 'Columns', None),
            "bits_stored": getattr(ds, 'BitsStored', None),
        }
    except Exception as e:
        logger.error(f"Error getting DICOM metadata from GCS: {e}")
        return {"error": str(e)}


def convert_dicom_to_png_from_gcs(filename: str, target_size: tuple = (800, 800)) -> Optional[str]:
    """
    Download DICOM from GCS, convert to optimized PNG, return base64.

    Args:
        filename: DICOM filename in GCS
        target_size: Target image dimensions

    Returns:
        Base64-encoded PNG string or None
    """
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        from io import BytesIO
        from PIL import Image
        import numpy as np
        import base64

        dicom_bytes = load_dicom_file(filename)
        if not dicom_bytes:
            logger.warning(f"Could not load DICOM bytes for: {filename}")
            return None

        # Parse DICOM
        ds = pydicom.dcmread(BytesIO(dicom_bytes))

        # Get pixel data
        pixel_array = ds.pixel_array

        # Handle multi-dimensional arrays (3D volumes, unusual shapes)
        # Squeeze single-frame dimensions: (1, 1, 256) → (256,) or (256, 256, 1) → (256, 256)
        while pixel_array.ndim > 2 and 1 in pixel_array.shape:
            pixel_array = np.squeeze(pixel_array)

        # If still 3D (true multi-frame), take middle slice
        if pixel_array.ndim == 3:
            middle_slice = pixel_array.shape[0] // 2
            pixel_array = pixel_array[middle_slice, :, :]

        # If 1D (unusual format), try to reshape to square
        if pixel_array.ndim == 1:
            size = int(np.sqrt(len(pixel_array)))
            if size * size == len(pixel_array):
                pixel_array = pixel_array.reshape(size, size)
            else:
                logger.warning(f"Cannot reshape 1D array of length {len(pixel_array)} to square")
                return None

        # Apply VOI LUT if available (improves contrast)
        try:
            pixel_array = apply_voi_lut(pixel_array, ds)
        except Exception:
            pass  # Use raw pixel data if VOI LUT fails

        # Normalize to 0-255 range
        if pixel_array.size > 0:
            pixel_array = pixel_array - pixel_array.min()
            if pixel_array.max() > 0:
                pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
            else:
                pixel_array = pixel_array.astype(np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(pixel_array)

        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize for token efficiency
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Convert to base64 PNG
        buffer = BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting DICOM to PNG from GCS: {e}")
        return None


def clear_cache():
    """Clear all in-memory caches."""
    global _patient_cache, _patient_list_cache, _csv_cache
    _patient_cache.clear()
    _patient_list_cache = None
    _csv_cache.clear()
    logger.info("Cleared all storage caches")


def load_dna_file_gcs(filename: str) -> Optional[str]:
    """Load a DNA/genomic CSV file from GCS."""
    client, bucket = get_gcs_client()
    if not bucket:
        return None

    try:
        blob = bucket.blob(f"dna/{filename}")
        return blob.download_as_text()
    except Exception as e:
        logger.error(f"Error loading DNA file {filename} from GCS: {e}")
        return None


def load_dna_file_local(filename: str) -> Optional[str]:
    """Load a DNA/genomic CSV file from local filesystem."""
    dna_path = Path(LOCAL_DNA_PATH) / filename
    if dna_path.exists():
        return dna_path.read_text()
    return None


def load_dna_file(filename: str) -> Optional[str]:
    """
    Load a DNA/genomic file (auto-selects GCS or local).

    Args:
        filename: DNA filename (e.g., 'patient_uuid.csv')

    Returns:
        DNA file content as string or None
    """
    if USE_GCS:
        result = load_dna_file_gcs(filename)
        if result:
            return result

    return load_dna_file_local(filename)


def list_dna_files_gcs(limit: Optional[int] = None) -> List[str]:
    """List DNA files in GCS."""
    client, bucket = get_gcs_client()
    if not bucket:
        return []

    try:
        files = []
        blobs = bucket.list_blobs(prefix="dna/")

        for blob in blobs:
            if blob.name.endswith('.csv'):
                filename = blob.name.split('/')[-1]
                files.append(filename)

                if limit and len(files) >= limit:
                    break

        return files
    except Exception as e:
        logger.error(f"Error listing DNA files from GCS: {e}")
        return []


def list_dna_files_local(limit: Optional[int] = None) -> List[str]:
    """List DNA files in local filesystem."""
    dna_path = Path(LOCAL_DNA_PATH)
    if not dna_path.exists():
        return []

    files = []
    for csv_file in dna_path.glob("*.csv"):
        files.append(csv_file.name)
        if limit and len(files) >= limit:
            break

    return files


def list_dna_files(limit: Optional[int] = None) -> List[str]:
    """
    List available DNA/genomic files (auto-selects GCS or local).

    Args:
        limit: Maximum number of files to return

    Returns:
        List of DNA filenames
    """
    if USE_GCS:
        result = list_dna_files_gcs(limit)
        if result:
            return result

    return list_dna_files_local(limit)


def get_storage_info() -> dict:
    """Get information about current storage configuration."""
    return {
        "use_gcs": USE_GCS,
        "gcs_bucket": GCS_BUCKET_NAME if USE_GCS else None,
        "local_fhir_path": LOCAL_FHIR_PATH,
        "local_dicom_path": LOCAL_DICOM_PATH,
        "local_csv_path": LOCAL_CSV_PATH,
        "local_dna_path": LOCAL_DNA_PATH,
        "cache_stats": {
            "patients_cached": len(_patient_cache),
            "csv_cached": len(_csv_cache),
            "patient_list_cached": _patient_list_cache is not None,
        }
    }
