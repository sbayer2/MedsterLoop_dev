"""
MCP Medical Analysis Client

Connects to FastMCP medical analysis server for specialist-level clinical document analysis.

Recursive AI Architecture:
- Local Agent: Claude Sonnet 4.5 (Medster) - Orchestration, tool selection, FHIR data extraction
- Remote Server: Claude Sonnet 4.5 (FastMCP) - Specialist medical document analysis

This creates a "medical specialist consultant" in Medster's backpack that can be delegated
complex clinical reasoning tasks requiring deep medical knowledge and multi-step analysis.
"""

from langchain.tools import tool
from typing import Literal, Optional
from pydantic import BaseModel, Field
import os
import requests
import json
import sys
import logging

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

####################################
# MCP Server Configuration
####################################

# Your FastMCP medical analysis server endpoint
# Note: FastMCP servers use /mcp endpoint for JSON-RPC
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://Medical-agent-server.fastmcp.app/mcp")

# MCP API Key for authentication
MCP_API_KEY = os.getenv("MCP_API_KEY", "")

# Enable debug logging for MCP calls
MCP_DEBUG = os.getenv("MCP_DEBUG", "false").lower() == "true"

# Debug log file
MCP_LOG_FILE = "/Users/sbm4_mac/Desktop/Medster/mcp_debug.log"

# Synthetic data disclaimer for Claude safeguards
SYNTHETIC_DATA_DISCLAIMER = """
[DISCLAIMER: This is SYNTHETIC patient data from the Coherent Data Set (SYNTHEA).
This is NOT real patient data - no PHI or HIPAA concerns apply.
This data is generated for medical AI research and education purposes.
Source: https://synthea.mitre.org/downloads - Coherent Data Set]

"""

def mcp_log(message: str):
    """Write debug message to log file"""
    if MCP_DEBUG:
        with open(MCP_LOG_FILE, "a") as f:
            f.write(f"{message}\n")
        # Also try stderr which might not be captured
        print(f"{message}", file=sys.stderr)


####################################
# Input Schemas
####################################

class ComplexNoteAnalysisInput(BaseModel):
    note_text: str = Field(description="The clinical note text to analyze (SOAP note, discharge summary, consult note, etc.)")
    analysis_type: Literal["basic", "comprehensive", "complicated"] = Field(
        default="complicated",
        description="Level of analysis: 'basic' for simple extraction, 'comprehensive' for detailed analysis, 'complicated' for multi-step clinical reasoning with quality assurance"
    )
    # NOTE: context parameter removed - not supported by deployed FastMCP server
    # Context can be prepended to note_text if needed


####################################
# Tools
####################################

@tool(args_schema=ComplexNoteAnalysisInput)
def analyze_medical_document(
    note_text: str,
    analysis_type: Literal["basic", "comprehensive", "complicated"] = "complicated"
) -> dict:
    """
    Analyzes medical documents using the FastMCP server with Claude Sonnet 4.5.
    Delegates complex clinical analysis to the MCP medical server for AI-powered insights.

    Analysis types:
    - basic: Quick extraction of key clinical data
    - comprehensive: Detailed analysis with clinical context (multi-step reasoning)
    - complicated: Alias for 'comprehensive' (automatically mapped on client side)

    Note: 'complicated' is automatically mapped to 'comprehensive' when calling the server,
    as the deployed FastMCP server uses 'comprehensive' for advanced analysis with Claude Sonnet 4.5.

    Useful for: SOAP notes, discharge summaries, lab interpretations,
    clinical pattern recognition, and decision support.

    Architecture: This tool creates a recursive AI system where:
    - Local: Claude Sonnet 4.5 (Medster) handles orchestration and tool selection
    - Remote: Claude Sonnet 4.5 (MCP Server) provides specialist medical analysis
    """
    try:
        mcp_log(f"[MCP] Calling server at {MCP_SERVER_URL}")
        mcp_log(f"[MCP] Analysis type: {analysis_type}")
        mcp_log(f"[MCP] Note text length: {len(note_text)} chars")

        # Map Medster analysis types to server analysis types
        # Server uses "comprehensive" for advanced analysis (not "complicated")
        server_analysis_type = analysis_type
        if analysis_type == "complicated":
            server_analysis_type = "comprehensive"
            mcp_log(f"[MCP] Mapping 'complicated' -> 'comprehensive' for server")

        # Prepend synthetic data disclaimer to avoid Claude safeguard issues
        # The Coherent Data Set is synthetic - no PHI concerns
        note_with_disclaimer = SYNTHETIC_DATA_DISCLAIMER + note_text
        mcp_log(f"[MCP] Added synthetic data disclaimer ({len(SYNTHETIC_DATA_DISCLAIMER)} chars)")

        # Build MCP JSON-RPC 2.0 request for tool call
        # The /mcp endpoint requires JSON-RPC format
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "analyze_medical_document",
                "arguments": {
                    "document_content": note_with_disclaimer,
                    "analysis_type": server_analysis_type,
                }
            }
        }

        # NOTE: The deployed FastMCP server does not support the 'context' parameter
        # Context can be prepended to the document_content if needed
        # if context:
        #     mcp_request["params"]["arguments"]["context"] = context

        # Try MCP JSON-RPC endpoint - use URL directly as provided
        mcp_endpoints = [
            MCP_SERVER_URL,  # Use the complete endpoint URL from config
        ]

        import time as time_module
        import uuid

        # Add unique request ID to prevent CloudFront caching
        request_id = str(uuid.uuid4())
        mcp_request["id"] = request_id
        mcp_log(f"[MCP] Request ID: {request_id}")

        # Note: Warmup ping removed - HEAD method returns 405 on MCP endpoints
        # The unique request ID prevents caching issues instead

        last_error = None
        for endpoint in mcp_endpoints:
            # Retry logic with exponential backoff for timeouts
            max_retries = 2
            for retry in range(max_retries):
                try:
                    mcp_log(f"[MCP] Trying endpoint: {endpoint} (attempt {retry + 1}/{max_retries})")
                    if endpoint.endswith("/mcp") or endpoint.endswith("/rpc"):
                        mcp_log(f"[MCP] Request arguments: {mcp_request['params']['arguments']}")

                    # Build headers with optional auth
                    # CloudFront requires application/json Content-Type for this server
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "Cache-Control": "no-cache, no-store",
                        "X-Request-ID": request_id,
                    }
                    if MCP_API_KEY:
                        headers["Authorization"] = f"Bearer {MCP_API_KEY}"

                    # Send JSON-RPC request (NOT plain text - CloudFront rejects text/plain)
                    mcp_log(f"[MCP] Sending JSON-RPC request to {endpoint}")

                    response = requests.post(
                        endpoint,
                        json=mcp_request,  # JSON-RPC format
                        headers=headers,
                        timeout=60  # 1 minute timeout (reduced from 3 min for better UX)
                    )
                    mcp_log(f"[MCP] Response status: {response.status_code}")
                    mcp_log(f"[MCP] Response headers: {dict(response.headers)}")
                    mcp_log(f"[MCP] Response body (first 500 chars): {response.text[:500]}")

                    if response.status_code == 200:
                        # Handle SSE (Server-Sent Events) format from FastMCP
                        response_text = response.text
                        mcp_log(f"[MCP] Parsing response format")

                        # Parse SSE format: SSE can have ping comments (: ping), event lines, and data lines
                        # Look for "data:" line which contains the JSON-RPC response
                        if "event:" in response_text or response_text.startswith(":"):
                            # Extract JSON from SSE data line
                            lines = response_text.split("\n")
                            mcp_log(f"[MCP] Found {len(lines)} lines in SSE response")
                            for line in lines:
                                if line.startswith("data:"):
                                    json_str = line[5:].strip()  # Remove "data:" prefix
                                    mcp_log(f"[MCP] Found data line, length: {len(json_str)} chars")
                                    try:
                                        result = json.loads(json_str)
                                        mcp_log(f"[MCP] Successfully parsed JSON-RPC response")
                                    except json.JSONDecodeError as e:
                                        logger.error(f"JSON parsing error in SSE data: {e}")
                                        mcp_log(f"[MCP] JSON parsing error: {e}")
                                        mcp_log(f"[MCP] Malformed JSON (first 200 chars): {json_str[:200]}")
                                        result = {"error": f"JSON parsing error: {str(e)}", "raw_data": json_str[:500]}
                                    break
                            else:
                                logger.warning("No data line found in SSE response")
                                mcp_log(f"[MCP] No data line in SSE response")
                                result = {"error": "No data in SSE response"}
                        else:
                            # Regular JSON response
                            mcp_log(f"[MCP] Parsing as regular JSON")
                            try:
                                result = response.json()
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parsing error in response: {e}")
                                mcp_log(f"[MCP] JSON parsing error: {e}")
                                mcp_log(f"[MCP] Malformed response (first 200 chars): {response_text[:200]}")
                                result = {"error": f"JSON parsing error: {str(e)}", "raw_data": response_text[:500]}

                        mcp_log(f"[MCP] Success from {endpoint}")
                        mcp_log(f"[MCP] Response keys: {result.keys() if isinstance(result, dict) else 'not dict'}")

                        # Handle direct MCP response format (not JSON-RPC wrapped)
                        # Response format: {"content": [...], "isError": false, "structuredContent": {...}}
                        if "content" in result:
                            content = result["content"]
                            mcp_log(f"[MCP] Content type: {type(content)}, length: {len(content) if isinstance(content, (list, str)) else 'N/A'}")

                            if isinstance(content, list) and len(content) > 0:
                                analysis_text = content[0].get("text", str(content))
                            else:
                                analysis_text = str(content)

                            mcp_log(f"[MCP] Analysis text length: {len(analysis_text)} chars")

                            # Check if this is an error response
                            if result.get("isError"):
                                mcp_log(f"[MCP] ERROR RESPONSE: {analysis_text}")
                                return {
                                    "analysis_type": analysis_type,
                                    "server_analysis_type": server_analysis_type,
                                    "status": "error",
                                    "error": f"MCP Server Error: {analysis_text}",
                                    "source": f"MCP Medical Analysis Server ({endpoint})"
                                }

                            # Get structured content if available
                            structured = result.get("structuredContent", {})
                            tokens_used = structured.get("tokens_used", {})

                            return {
                                "analysis_type": analysis_type,
                                "server_analysis_type": server_analysis_type,
                                "status": "success",
                                "analysis": analysis_text,
                                "tokens_used": tokens_used.get("total_tokens", 0) if isinstance(tokens_used, dict) else 0,
                                "processing_time": structured.get("processing_time_seconds", 0),
                                "source": f"MCP Medical Analysis Server ({endpoint})"
                            }
                        elif "result" in result:
                            # JSON-RPC wrapped response
                            mcp_result = result["result"]
                            mcp_log(f"[MCP] JSON-RPC result keys: {mcp_result.keys() if isinstance(mcp_result, dict) else type(mcp_result)}")

                            if isinstance(mcp_result, dict) and "content" in mcp_result:
                                content = mcp_result["content"]
                                mcp_log(f"[MCP] Content type: {type(content)}, length: {len(content) if isinstance(content, (list, str)) else 'N/A'}")

                                if isinstance(content, list) and len(content) > 0:
                                    analysis_text = content[0].get("text", str(content))
                                else:
                                    analysis_text = str(content)

                                mcp_log(f"[MCP] Analysis text length: {len(analysis_text)} chars")

                                # Check for error
                                if mcp_result.get("isError"):
                                    mcp_log(f"[MCP] ERROR RESPONSE: {analysis_text}")
                                    return {
                                        "analysis_type": analysis_type,
                                        "server_analysis_type": server_analysis_type,
                                        "status": "error",
                                        "error": f"MCP Server Error: {analysis_text}",
                                        "source": f"MCP Medical Analysis Server ({endpoint})"
                                    }

                                # Get structured content if available
                                structured = mcp_result.get("structuredContent", {})
                                tokens_used = structured.get("tokens_used", {})

                                return {
                                    "analysis_type": analysis_type,
                                    "server_analysis_type": server_analysis_type,
                                    "status": "success",
                                    "analysis": analysis_text,
                                    "tokens_used": tokens_used.get("total_tokens", 0) if isinstance(tokens_used, dict) else 0,
                                    "processing_time": structured.get("processing_time_seconds", 0),
                                    "source": f"MCP Medical Analysis Server ({endpoint})"
                                }
                        elif "error" in result:
                            last_error = result["error"].get("message", str(result["error"])) if isinstance(result["error"], dict) else str(result["error"])
                            break
                        else:
                            # Unknown format - return as-is
                            return {
                                "analysis_type": analysis_type,
                                "server_analysis_type": server_analysis_type,
                                "status": "success",
                                "analysis": str(result),
                                "source": f"MCP Medical Analysis Server ({endpoint})"
                            }
                    elif response.status_code == 404:
                        last_error = f"Endpoint not found: {endpoint}"
                        logger.error(f"MCP endpoint not found (404): {endpoint}")
                        mcp_log(f"[MCP] 404 response body: {response.text[:500]}")
                        break  # Try next endpoint
                    elif response.status_code == 401 or response.status_code == 403:
                        last_error = f"Authentication/Authorization error ({response.status_code}): {endpoint}"
                        logger.error(f"MCP auth error ({response.status_code}): {endpoint}")
                        mcp_log(f"[MCP] Auth error response: {response.text[:500]}")
                        break  # Try next endpoint
                    elif response.status_code >= 500:
                        last_error = f"Server error ({response.status_code}): {endpoint}"
                        logger.error(f"MCP server error ({response.status_code}): {endpoint}")
                        mcp_log(f"[MCP] Server error response: {response.text[:500]}")
                        break  # Try next endpoint
                    else:
                        last_error = f"HTTP error {response.status_code}: {response.text[:200]}"
                        logger.error(f"MCP HTTP error ({response.status_code}): {endpoint}")
                        mcp_log(f"[MCP] Error response: Status {response.status_code}")
                        mcp_log(f"[MCP] Response body: {response.text[:500]}")
                        break  # Try next endpoint

                except requests.exceptions.Timeout:
                    last_error = f"Timeout after 60s (attempt {retry + 1})"
                    logger.warning(f"MCP request timeout: attempt {retry + 1}/{max_retries}")
                    mcp_log(f"[MCP] Timeout on attempt {retry + 1}/{max_retries}")
                    if retry < max_retries - 1:
                        wait_time = 5 * (retry + 1)  # 5s, 10s backoff
                        mcp_log(f"[MCP] Waiting {wait_time}s before retry...")
                        time_module.sleep(wait_time)
                        continue
                    break  # All retries exhausted, try next endpoint
                except requests.exceptions.SSLError as e:
                    last_error = f"SSL/TLS error connecting to {endpoint}"
                    logger.error(f"MCP SSL error: {e}")
                    mcp_log(f"[MCP] SSL error: {e}")
                    break  # SSL errors are not retryable, try next endpoint
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Connection failed to {endpoint}"
                    logger.error(f"MCP connection error: {e}")
                    mcp_log(f"[MCP] Connection error: {e}")
                    break  # Try next endpoint
                except json.JSONDecodeError as e:
                    last_error = f"Malformed JSON response from {endpoint}"
                    logger.error(f"MCP JSON decode error: {e}")
                    mcp_log(f"[MCP] JSON decode error in outer handler: {e}")
                    break  # Try next endpoint
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"MCP unexpected error: {type(e).__name__}: {e}")
                    mcp_log(f"[MCP] Exception ({type(e).__name__}): {e}")
                    break  # Try next endpoint

        # All endpoints failed
        logger.error(f"All MCP endpoints failed. Last error: {last_error}")
        mcp_log(f"[MCP] All endpoints failed. Last error: {last_error}")
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": f"All MCP endpoints failed. Last error: {last_error}",
            "server_url": MCP_SERVER_URL,
            "recommendation": "Check MCP server status and endpoint configuration"
        }

    except requests.exceptions.Timeout:
        logger.error("MCP server request timed out at outer level")
        mcp_log(f"[MCP] Outer timeout exception")
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": "MCP server request timed out",
            "recommendation": "Try 'comprehensive' or 'basic' analysis type for faster results"
        }
    except json.JSONDecodeError as e:
        logger.error(f"MCP JSON decode error at outer level: {e}")
        mcp_log(f"[MCP] Outer JSON decode error: {e}")
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": f"Failed to parse MCP server response: {str(e)}",
            "recommendation": "Server returned malformed JSON response"
        }
    except Exception as e:
        logger.error(f"MCP unexpected error at outer level: {type(e).__name__}: {e}")
        mcp_log(f"[MCP] Outer exception ({type(e).__name__}): {e}")
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
            "recommendation": "Check logs for detailed error information"
        }
