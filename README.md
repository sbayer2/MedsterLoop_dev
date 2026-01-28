# Medster - Autonomous Clinical Case Analysis Agent

An autonomous agent for deep clinical case analysis, inspired by [Dexter](https://github.com/virattt/dexter) and adapted for medical domain.

> **Research & Educational Use**: This is an open-source research project, not a production medical system. Always verify findings with appropriate clinical resources.

## Overview

Medster "thinks, plans, and learns as it works" - performing clinical analysis through task planning, self-reflection, and real-time medical data. It leverages SYNTHEA/FHIR data sources, supports multimodal analysis (DICOM, ECG), and can analyze uploaded clinical documents.

### Healthcare Data Compliance

**Anthropic offers HIPAA-ready infrastructure** for Claude when used with appropriate Business Associate Agreements (BAAs). This means Medster can be configured for workflows involving protected health information (PHI) when proper compliance measures are in place.

**Important clarifications:**
- Claude is **not** FDA-approved as a medical device
- HIPAA compliance requires a BAA between your organization and Anthropic
- This is a research tool, not a certified clinical decision support system

**When using with real patient data:**
- Execute a BAA with Anthropic for HIPAA-covered workflows
- Follow your institution's data handling and privacy policies
- Ensure appropriate access controls and audit logging
- Always have qualified clinicians review AI-generated analysis

## Core Capabilities

- **Intelligent Task Planning**: Breaks down complex clinical questions into structured diagnostic and therapeutic steps
- **Autonomous Execution**: Automatically selects and runs appropriate tools for data gathering (labs, notes, vitals, imaging, medications)
- **Self-Validation**: Verifies its own work and iterates until tasks are complete
- **Real-Time Medical Data**: Patient notes, lab results, vital sign trends, medication lists
- **Safety Mechanisms**: Loop detection, critical value flagging, drug interaction checking

## Primary Use Cases

1. **Clinical Case Analysis** - Comprehensive review of patient cases with risk stratification
2. **Differential Diagnosis Workup** - Prioritized differentials with optimal diagnostic sequences

## Architecture

```
                    MEDSTER CLI
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Coherent │  │   MCP    │  │ Claude   │
    │ Data Set │  │  Server  │  │ Sonnet   │
    │          │  │          │  │   4.5    │
    │ FHIR     │  │ Analyze  │  │ Planning │
    │ DICOM    │  │ Complex  │  │ Reasoning│
    │ ECG/Notes│  │ Notes    │  │ Synthesis│
    └──────────┘  └──────────┘  └──────────┘
```

## Requirements

- Python 3.11+
- Node.js 18+ (for frontend)
- Anthropic API key (for Claude Sonnet 4.5)
- Google Cloud Platform account (for cloud deployment)
- Optional: Coherent Data Set (9GB synthetic medical data)
- Optional: Your MCP medical analysis server for complex note analysis

## Quick Start (Local Development)

### 1. Clone the Repository

```bash
git clone https://github.com/sbayer2/Medster_dev.git
cd Medster_dev
```

### 2. Set Up Anthropic API

1. Create an account at [Anthropic Console](https://console.anthropic.com/)
2. Navigate to **API Keys** in the dashboard
3. Click **Create Key** and copy the key (starts with `sk-ant-`)
4. Save this key securely - you'll need it for the `.env` file

### 3. Configure Environment

```bash
cp env.example .env
```

Edit `.env` with your credentials:
```bash
# Required: Your Anthropic API key
ANTHROPIC_API_KEY=sk-ant-your_actual_key_here

# Coherent Data Set paths (if using synthetic data)
COHERENT_DATA_PATH=./coherent_data/fhir
COHERENT_DICOM_PATH=./coherent_data/dicom
COHERENT_CSV_PATH=./coherent_data/csv

# Optional: Your MCP server for advanced analysis
MCP_SERVER_URL=<YOUR_MCP_SERVER_URL>
```

### 4. Install Dependencies

**Backend (Python):**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

**Frontend (Node.js):**
```bash
cd frontend
npm install
cd ..
```

### 5. Run Locally

**CLI Mode:**
```bash
uv run medster-agent
```

**Web Interface:**
```bash
# Terminal 1: Backend API
uv run uvicorn medster.api:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Access the web UI at http://localhost:3000

**⚠️ Security Note:** Never commit your `.env` file to git. It's already in `.gitignore`.

---

## Cloud Deployment (Google Cloud Run)

### Prerequisites

1. [Google Cloud Platform](https://cloud.google.com/) account with billing enabled
2. [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed

### Step 1: Create a GCP Project

```bash
# Login to GCP
gcloud auth login

# Create a new project (or use existing)
gcloud projects create your-medster-project --name="Medster"

# Set as active project
gcloud config set project your-medster-project

# Enable billing (required for Cloud Run)
# Visit: https://console.cloud.google.com/billing/linkedaccount?project=your-medster-project
```

### Step 2: Enable Required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 3: Set Up Secrets

```bash
# Create secrets for API keys
echo -n "sk-ant-your-key" | gcloud secrets create anthropic-api-key --data-file=-

# Grant Cloud Run access to secrets
gcloud secrets add-iam-policy-binding anthropic-api-key \
    --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### Step 4: Deploy

```bash
# Deploy both frontend and backend
gcloud builds submit --config cloudbuild.yaml
```

This will:
- Build Docker images for frontend and backend
- Push to Google Container Registry
- Deploy to Cloud Run with auto-scaling

### Step 5: Get Service URLs

```bash
gcloud run services list
```

Your services will be available at URLs like:
- Backend: `https://medster-backend-xxxxx.run.app`
- Frontend: `https://medster-frontend-xxxxx.run.app`

---

## Coherent Data Set Setup (Optional)

The Coherent Data Set is a 9GB synthetic medical dataset with linked FHIR, DICOM, genomics, and ECG data.

### Download

1. Visit https://synthea.mitre.org/downloads
2. Download "Coherent Data Set" (~9GB)
3. Extract to `./coherent_data/`

### Directory Structure

```
coherent_data/
├── fhir/          # FHIR patient bundles (1,278 patients)
├── dicom/         # Medical images (298 brain MRIs)
├── csv/           # ECG waveforms, observations
└── dna/           # Genomic data (889 files)
```

### For Cloud Deployment

Upload to Google Cloud Storage:
```bash
gsutil -m cp -r coherent_data gs://your-bucket/coherent_data
```

Set environment variable:
```bash
USE_GCS=true
GCS_COHERENT_BUCKET=your-bucket
```

---

## File Upload Feature

Medster supports uploading clinical documents for analysis via the web interface.

### Supported Formats
- TXT, PDF, CSV, JSON, MD, XML, HL7
- Maximum file size: 5MB
- Large files are automatically truncated to ~150K characters to fit Claude's context window

### How It Works
1. Click the paperclip icon in the chat interface
2. Select a clinical document
3. Optionally add a question about the document
4. Medster analyzes the uploaded content using specialized primitives:
   - `search_uploaded_content(pattern)` - Regex search through the document
   - `extract_sections(start, end)` - Extract sections between patterns

### Example Use Cases
- "Search this medical record for hypertension and diabetes diagnoses"
- "Extract all medication mentions from this discharge summary"
- "Find lab values in this clinical note and flag abnormals"

---

## Usage

Run the interactive CLI:
```bash
uv run medster-agent
```

Or:
```bash
python -m medster.cli
```

### Example Queries

**Clinical Case Analysis:**
```
medster>> Analyze this patient - 58yo male with chest pain, elevated troponins, and new ECG changes. What's the diagnostic workup and risk stratification?
```

**Lab Review:**
```
medster>> Get the last 7 days of labs for patient 12345 and identify any critical values or concerning trends
```

**Medication Safety:**
```
medster>> Review the medication list for patient 12345 and check for potential drug interactions
```

**Differential Diagnosis:**
```
medster>> Patient presents with fatigue, weight loss, and night sweats. Generate a prioritized differential and optimal workup sequence.
```

## Available Tools

### Medical Data (SYNTHEA/FHIR)
- `get_patient_labs` - Laboratory results with reference ranges
- `get_vital_signs` - Vital sign measurements and trends
- `get_demographics` - Patient demographic information
- `get_clinical_notes` - Progress notes, H&P, consultations
- `get_soap_notes` - SOAP-formatted progress notes
- `get_discharge_summary` - Hospital discharge summaries
- `get_medication_list` - Current and historical medications
- `check_drug_interactions` - Drug-drug interaction screening
- `get_radiology_reports` - Imaging studies and interpretations

### Clinical Scores
- `calculate_clinical_score` - Wells' Criteria, CHA2DS2-VASc, CURB-65, MELD, etc.

### Complex Analysis
- `analyze_complex_note` - Multi-step clinical reasoning via MCP server (Claude/Anthropic)

## Data Sources

### Coherent Data Set
Medster uses the **Coherent Data Set** - a comprehensive synthetic dataset that includes:
- FHIR resources (patient records, labs, vitals, medications, notes)
- DICOM images (X-rays, CT scans)
- Genomic data
- Physiological data (ECGs)
- Clinical notes

All data types are linked together via FHIR references.

**Download**: https://synthea.mitre.org/downloads (9 GB)

**Citation**:
> Walonoski J, et al. The "Coherent Data Set": Combining Patient Data and Imaging in a Comprehensive, Synthetic Health Record. Electronics. 2022; 11(8):1199. https://doi.org/10.3390/electronics11081199

### MCP Medical Analysis Server (Optional)

For complex document analysis, Medster can integrate with a FastMCP medical analysis server. This enables a "recursive AI architecture" where:
- Local Medster agent orchestrates data retrieval
- Remote MCP server provides specialist-level clinical reasoning

To use this feature:
1. Deploy your own FastMCP server
2. Set `MCP_SERVER_URL=<YOUR_MCP_SERVER_URL>` in `.env`

The MCP server is optional - Medster works fully without it using local Claude analysis.

## Configuration Reference

See `env.example` for all available configuration options. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key for Claude Sonnet 4.5 |
| `COHERENT_DATA_PATH` | No | Path to FHIR data (for database queries) |
| `MCP_SERVER_URL` | No | Your FastMCP server URL |
| `USE_GCS` | No | Set to `true` for Cloud Run with GCS data |

## Safety & Disclaimer

**IMPORTANT**: Medster is for research and educational purposes only.

- Not intended for clinical decision-making without physician review
- Always verify findings with appropriate clinical resources
- Critical values and drug interactions are simplified checks
- Use clinical judgment for all patient care decisions

## Architecture Details

Medster preserves Dexter's proven multi-agent architecture:

1. **Planning Module** - Decomposes clinical queries into tasks
2. **Action Module** - Selects appropriate tools for data retrieval
3. **Validation Module** - Verifies task completion
4. **Synthesis Module** - Generates comprehensive clinical analysis

Safety mechanisms include:
- Global step limits (default: 20)
- Per-task step limits (default: 5)
- Loop detection (prevents repetitive actions)
- Critical value flagging

## License

MIT License

## Acknowledgments

- [Dexter](https://github.com/virattt/dexter) by @virattt - The original financial research agent that inspired this architecture. Medster adapts Dexter's proven multi-agent loop (planning → action → validation → synthesis) for the medical domain. A local reference copy of the Dexter codebase is maintained in `dexter-reference/` for architectural consultation during development.
- [SYNTHEA](https://synthetichealth.github.io/synthea/) - Synthetic patient data generator
- [HAPI FHIR](https://hapifhir.io/) - FHIR server implementation
- [Coherent Data Set](https://synthea.mitre.org/downloads) - 9GB synthetic dataset integrating FHIR, DICOM, genomics, and ECG data for comprehensive multimodal medical AI research
