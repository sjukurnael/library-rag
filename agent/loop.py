"""
The hand-written Anthropic tool-use loop: system prompt, tool schemas, and
the run() function that drives the conversation until the model stops
calling tools and produces its final recommendation.
"""
import json
import os
import sys

from anthropic import Anthropic

from agent import assumptions, tools
from drive.client import DriveAuthError

ROOT_FOLDER_ID = "1jOO-7ZAEosq2mAtuVzTTq9uAekrypVfq"
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_ITERATIONS = 25

TOOL_SCHEMAS = [
    {
        "name": "list_folder",
        "description": (
            "List the immediate children of a Google Drive folder. ONE level "
            "only -- never recursive. Returns the folder's own name/url, its "
            "direct subfolders (id/name/url), and aggregate stats over the "
            "PDFs directly inside it (count, total/min/median/max size in MB, "
            "count of non-PDF junk files, and up to 8 sample PDFs). Automatically "
            "cached to disk, so calling this again on the same folder is free."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_id": {
                    "type": "string",
                    "description": "The Google Drive folder ID to list.",
                }
            },
            "required": ["folder_id"],
        },
    },
    {
        "name": "estimate_pipeline",
        "description": (
            "Deterministically estimate cost and time for the FULL ingestion "
            "pipeline (download -> extract/OCR -> chunk -> embed -> store) for "
            "a candidate set of PDFs. All arithmetic happens here -- do not try "
            "to compute costs yourself, always call this tool. Returns a "
            "stage-by-stage breakdown as (low, high) ranges, including both an "
            "OCR-API cost path and a rented-GPU cost path for the scanned-page "
            "extraction stage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pdf_count": {
                    "type": "integer",
                    "description": "Number of PDFs in the candidate set.",
                },
                "total_mb": {
                    "type": "number",
                    "description": "Total size in MB of the candidate set's PDFs.",
                },
                "scanned_ratio": {
                    "type": "number",
                    "description": (
                        "Your judgment: fraction (0.0-1.0) of the content that is "
                        "scanned images vs digital-native text, inferred from the "
                        "file-size distribution seen in list_folder results "
                        "(median file size 50MB+ suggests scans; <10MB suggests "
                        "digital-native text)."
                    ),
                },
            },
            "required": ["pdf_count", "total_mb", "scanned_ratio"],
        },
    },
]

SYSTEM_PROMPT = f"""You are a Drive-exploration agent. Your job: scout a shared \
Google Drive library (a future RAG "library tutor" chatbot's source corpus, \
~130GB of PDF books) to pick ONE pilot folder for testing a PDF -> RAG \
ingestion pipeline (download -> extract/OCR -> chunk -> embed -> store -> \
ready for retrieval), and to budget that pilot run in time and dollars.

You have exactly two tools: list_folder (one level, never recursive, metadata \
only -- you must NEVER download or read file content) and estimate_pipeline \
(deterministic math -- you must NEVER compute costs yourself, always call it).

The root folder ID is {ROOT_FOLDER_ID}. Known top level: folders Articles, \
Books, Dissertations, Transcripts, plus junk files (.DS_Store, index.html, \
index.php). Expect junk throughout the tree -- only mimeType='application/pdf' \
files matter; list_folder already filters/aggregates this for you.

PILOT CAPS:
- Hard cap A: candidate folder's total PDF size must be UNDER {assumptions.PILOT_MAX_GB} GB.
- Hard cap B: estimated TOTAL pipeline cost for the pilot must be UNDER \
${assumptions.PILOT_MAX_COST_USD}, judged at the HIGH end of the estimate range \
(use the cheaper of the OCR-API-path vs GPU-path totals when checking this).
- Ideal candidate: 10-100 PDFs, real books (representative of the main corpus).

If a candidate folder busts the $5 cap on the OCR API path, check whether the \
rented-GPU/local-Marker path fits under $5 before giving up on it. You MUST \
ALWAYS produce a recommendation -- never conclude "nothing fits" or "no \
suitable folder found". If literally no folder satisfies BOTH caps, recommend \
the SMALLEST folder that actually contains PDFs (by total GB) as the primary \
pick, and state plainly which cap it violates and by how much.

SCANNED-RATIO HEURISTIC: PDFs where the median file size is large (50MB+) are \
likely image scans (large per-page raster images); PDFs with small median file \
size (<10MB) are likely digital-native text. Infer a scanned_ratio (0.0-1.0) \
per candidate from the size distribution (min/median/max, sample_pdfs) returned \
by list_folder, and explicitly state what you assumed and why.

HOW TO EXPLORE: Start at the root. Drill into promising folders (Books is the \
most representative of the real corpus). Skip folders that are obviously wrong \
(e.g. pure-junk folders, folders of Articles/Transcripts if Books gives you \
better candidates) -- do not exhaustively enumerate the whole tree. Stop once \
you're confident in a recommendation. Call estimate_pipeline for every serious \
candidate (2-4 folders), not just the winner, so alternatives are comparable. \
Also call estimate_pipeline ONCE with totals summed across the top-level \
folders (the whole ~130GB corpus) so your final answer includes a full-corpus \
extrapolation for planning purposes.

FINAL ANSWER (required, in this exact block format, all numbers as estimated \
ranges):

==================== RECOMMENDATION ====================
Option A (RECOMMENDED): <folder name/path> -- <size>, <PDF count>, ~<pages> pages
  <one line: why this option>
  URL: <drive url>

  Download          <time>        <cost>
  Extract + OCR     <time>        <cost range, API path> or <cost, GPU path>
  Chunk + embed     <time>        <cost>
  Store (Postgres)  <time>        <GB>, laptop-friendly y/n
  TOTAL             <time range>  <cost range>

Option B: <folder> -- <size>, <count>, est. <cost>, <time>
  <one line>
Option C: <folder> -- <size>, <count>, est. <cost>, <time>
  <one line>

Full-corpus extrapolation (all ~130 GB, for planning only):
  est. <pages> pages -> OCR <cost API> / <cost GPU>,
  embedding <cost>, total wall-clock <time>.
=========================================================

Before that block, briefly show your reasoning: which folders you checked, the \
scanned_ratio you assumed for each and why, and how each candidate compares \
against the two caps. Keep this brief -- the block above is the actual \
deliverable.
"""


def log(msg: str, verbose: bool):
    if verbose:
        print(f"[agent] {msg}")


def summarize_result(tool_name: str, result: dict) -> str:
    if tool_name == "list_folder":
        f = result["files"]
        return (
            f"{len(result['subfolders'])} subfolders, {f['pdf_count']} PDFs, "
            f"{f['pdf_total_mb']:.1f} MB, {f['other_count']} junk files"
        )
    if tool_name == "estimate_pipeline":
        t = result["totals"]
        return (
            f"API path {t['api_path']['cost_label']} / {t['api_path']['time_label']}; "
            f"GPU path {t['gpu_path']['cost_label']} / {t['gpu_path']['time_label']}"
        )
    return "ok"


def execute_tool(name: str, tool_input: dict, fresh: bool, verbose: bool) -> dict:
    if name == "list_folder":
        log(f"list_folder({tool_input['folder_id']})", verbose)
        result = tools.list_folder(tool_input["folder_id"], fresh=fresh)
    elif name == "estimate_pipeline":
        log(
            f"estimate_pipeline(pdf_count={tool_input['pdf_count']}, "
            f"total_mb={tool_input['total_mb']}, "
            f"scanned_ratio={tool_input['scanned_ratio']})",
            verbose,
        )
        result = tools.estimate_pipeline(
            tool_input["pdf_count"], tool_input["total_mb"], tool_input["scanned_ratio"]
        )
    else:
        result = {"error": f"unknown tool: {name}"}
    log(f"  -> {summarize_result(name, result)}", verbose)
    return result


def run(fresh: bool, verbose: bool):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": (
                f"Explore the Drive library starting at root folder "
                f"{ROOT_FOLDER_ID} and produce your recommendation."
            ),
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "\n".join(
                block.text for block in response.content if block.type == "text"
            )
            print(final_text)
            return 0

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = execute_tool(block.name, block.input, fresh, verbose)
            except DriveAuthError as e:
                print(f"\nDrive auth error: {e}", file=sys.stderr)
                sys.exit(1)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

    print(
        f"\n[agent] Hit the {MAX_ITERATIONS}-iteration leash without a final "
        "answer. Partial findings from this run are cached in cache.json; "
        "re-run to let the agent continue reasoning over them.",
        file=sys.stderr,
    )
    return 1
