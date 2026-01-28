"""
Image utilities for multimodal medical data analysis.

Provides token-efficient image conversion and optimization for DICOM, ECG, and other medical images.
"""

import base64
import io
from pathlib import Path
from typing import Optional, Tuple, List
import csv

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ImageConversionError(Exception):
    """Raised when image conversion fails."""
    pass


def dicom_to_base64_png(
    dicom_path: Path,
    target_size: Tuple[int, int] = (800, 800),
    quality: int = 85
) -> str:
    """
    Convert DICOM file to optimized base64-encoded PNG.

    Args:
        dicom_path: Path to DICOM file
        target_size: Target image size (width, height) for optimization
        quality: PNG compression quality (1-100)

    Returns:
        Base64-encoded PNG string

    Raises:
        ImageConversionError: If conversion fails
        ImportError: If pydicom or PIL not installed
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy not installed. Install with: uv add numpy")
    if not DICOM_AVAILABLE:
        raise ImportError("pydicom not installed. Install with: uv add pydicom")
    if not PIL_AVAILABLE:
        raise ImportError("Pillow not installed. Install with: uv add pillow")

    try:
        # Load DICOM file
        dicom = pydicom.dcmread(str(dicom_path))

        # Extract pixel data and apply VOI LUT (windowing) for proper visualization
        pixel_array = dicom.pixel_array

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
                # Can't reshape - use as 1D image (will fail gracefully)
                pass

        # Apply VOI LUT if available (improves contrast)
        try:
            pixel_array = apply_voi_lut(pixel_array, dicom)
        except Exception:
            pass  # Use raw pixel data if VOI LUT fails

        # Normalize to 0-255 range
        if pixel_array.size > 0:
            pixel_array = pixel_array - pixel_array.min()
            if pixel_array.max() > 0:
                pixel_array = (pixel_array / pixel_array.max() * 255).astype('uint8')
            else:
                pixel_array = pixel_array.astype('uint8')

        # Convert to PIL Image
        image = Image.fromarray(pixel_array)

        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize for token efficiency
        image.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Convert to PNG and encode as base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)

        base64_string = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_string

    except Exception as e:
        raise ImageConversionError(f"Failed to convert DICOM to PNG: {str(e)}") from e


def optimize_image(
    image_data: bytes,
    target_size: Tuple[int, int] = (800, 800),
    quality: int = 85
) -> str:
    """
    Optimize any image for token-efficient transmission.

    Args:
        image_data: Raw image bytes
        target_size: Target image size (width, height)
        quality: Compression quality (1-100)

    Returns:
        Base64-encoded optimized PNG string

    Raises:
        ImageConversionError: If optimization fails
        ImportError: If PIL not installed
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow not installed. Install with: uv add pillow")

    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        # Resize for token efficiency
        image.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Convert to PNG and encode as base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)

        base64_string = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_string

    except Exception as e:
        raise ImageConversionError(f"Failed to optimize image: {str(e)}") from e


def load_ecg_image_from_csv(
    csv_path: Path,
    patient_id: str
) -> Optional[str]:
    """
    Extract ECG image (base64 PNG) from observations.csv for a patient.

    Args:
        csv_path: Path to observations.csv
        patient_id: Patient UUID

    Returns:
        Base64-encoded PNG string if found, None otherwise

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Look for ECG observations for this patient
                if row.get('PATIENT') == patient_id and row.get('CODE') == '29303009':
                    # CODE 29303009 = Electrocardiographic procedure
                    ecg_base64 = row.get('VALUE', '')

                    # Verify it's a valid base64 image
                    if ecg_base64 and ecg_base64.startswith('iVBORw0KGgo'):
                        return ecg_base64

        return None

    except Exception as e:
        raise ImageConversionError(f"Failed to load ECG from CSV: {str(e)}") from e


def scan_all_dicom_files(dicom_dir: Path) -> List[Path]:
    """
    Scan DICOM directory and return all DICOM files.

    Args:
        dicom_dir: Directory containing DICOM files

    Returns:
        List of all DICOM file paths in the directory

    Raises:
        FileNotFoundError: If DICOM directory doesn't exist
    """
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Get all .dcm files in directory
    dicom_files = list(dicom_dir.glob("*.dcm"))

    return sorted(dicom_files)


def find_patient_dicom_files(
    dicom_dir: Path,
    patient_name: str
) -> List[Path]:
    """
    Find all DICOM files for a patient by name or UUID.

    Args:
        dicom_dir: Directory containing DICOM files
        patient_name: Patient name, UUID, or pattern to match in filename

    Returns:
        List of DICOM file paths for the patient

    Raises:
        FileNotFoundError: If DICOM directory doesn't exist
    """
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    # Extract base name (FirstName_LastName_UUID) without DICOM_ID suffix
    base_name = patient_name.split('[')[0] if '[' in patient_name else patient_name

    # Try matching UUID anywhere in filename (Coherent uses: FirstName_LastName_UUID_DICOMID.dcm)
    # First try exact start match, then try UUID anywhere in filename
    dicom_files = list(dicom_dir.glob(f"{base_name}*.dcm"))

    if not dicom_files:
        # If no match, try finding UUID anywhere in filename
        dicom_files = [f for f in dicom_dir.glob("*.dcm") if base_name in f.name]

    return sorted(dicom_files)


def get_image_metadata(image_path: Path) -> dict:
    """
    Extract metadata from a DICOM file.

    Args:
        image_path: Path to DICOM file

    Returns:
        Dictionary with image metadata

    Raises:
        ImportError: If pydicom not installed
        FileNotFoundError: If image doesn't exist
    """
    if not DICOM_AVAILABLE:
        raise ImportError("pydicom not installed. Install with: uv add pillow")

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        dicom = pydicom.dcmread(str(image_path))

        # Get dimensions from DICOM metadata tags (no pixel data access needed)
        rows = getattr(dicom, 'Rows', 'Unknown')
        cols = getattr(dicom, 'Columns', 'Unknown')
        dimensions = f"{cols}x{rows}" if rows != 'Unknown' and cols != 'Unknown' else 'Unknown'

        metadata = {
            'modality': str(getattr(dicom, 'Modality', 'Unknown')),
            'study_description': str(getattr(dicom, 'StudyDescription', 'Unknown')),
            'series_description': str(getattr(dicom, 'SeriesDescription', 'Unknown')),
            'body_part': str(getattr(dicom, 'BodyPartExamined', 'Unknown')),
            'patient_id': str(getattr(dicom, 'PatientID', 'Unknown')),
            'study_date': str(getattr(dicom, 'StudyDate', 'Unknown')),
            'dimensions': dimensions,
            'file_size_mb': round(image_path.stat().st_size / (1024 * 1024), 2)
        }

        return metadata

    except Exception as e:
        return {'error': f"Failed to read metadata: {str(e)}"}


def verify_dependencies() -> dict:
    """
    Check if required image processing dependencies are installed.

    Returns:
        Dictionary with dependency status
    """
    return {
        'numpy': NUMPY_AVAILABLE,
        'pydicom': DICOM_AVAILABLE,
        'pillow': PIL_AVAILABLE,
        'required_for': {
            'dicom_conversion': 'numpy, pydicom and pillow',
            'image_optimization': 'pillow',
            'ecg_extraction': 'none (uses standard library)'
        }
    }
