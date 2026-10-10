
import os
import re
import sys
import traceback
from io import StringIO
from contextlib import redirect_stdout
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI(title="AI Code Interpreter")


# Enable CORS for testing from browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and capture stdout or the full traceback.
    """
    stdout_buffer = StringIO()

    try:
        compiled_code = compile(code, "<string>", "exec")

        with redirect_stdout(stdout_buffer):
            exec(compiled_code, {"__name__": "__main__"})

        return {
            "success": True,
            "output": stdout_buffer.getvalue(),
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc(),
        }


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Analyze a traceback using Gemini's structured JSON output.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Analyze the Python execution error.

Identify the 1-based line number(s) in the USER-SUBMITTED CODE
that caused the error. Do not report lines from Python libraries
or internal interpreter frames.

For SyntaxError, use the indicated source line.
For runtime errors, identify the failing statement.
Return only valid structured output matching the schema.

USER CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ErrorAnalysis,
            temperature=0,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    analysis = ErrorAnalysis.model_validate_json(response.text)

    # Keep only valid, unique source-code line numbers.
    valid_lines = sorted({
        line for line in analysis.error_lines
        if 1 <= line <= len(code.splitlines())
    })

    return valid_lines


def fallback_error_lines(code: str, traceback_text: str) -> List[int]:
    """
    Recover a source line number from Python's traceback if AI analysis
    is unavailable. This is a fallback, not an AI analysis.
    """
    lines = code.splitlines()
    found = set()

    for match in re.finditer(
        r'File "<string>", line (\d+)', traceback_text
    ):
        line_number = int(match.group(1))
        if 1 <= line_number <= len(lines):
            found.add(line_number)

    # SyntaxError and some compile-time errors expose lineno directly
    # on the exception, but not always in the formatted traceback.
    # The endpoint's main execution path can be enhanced to preserve
    # that attribute if compile-time line recovery is needed.

    return sorted(found)


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    # Successful execution: never call the AI.
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    # Failed execution: analyze the traceback with AI.
    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"],
        )
    except Exception:
        # Keep the endpoint usable if AI analysis fails.
        error_lines = fallback_error_lines(
            request.code,
            execution["output"],
        )

    return {
        "error": error_lines,
        "result": execution["output"],
    }
