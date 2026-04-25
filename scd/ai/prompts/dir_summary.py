# --- Directory Summary (direct-content, no tools) ---
#
# Three prompt pairs:
# - SINGLE-SHOT: when all direct file contents + child summaries fit in one
#   request, a single LLM call produces the final summary.
# - PARTIAL: when files don't fit, we bin-pack them into chunks; each chunk
#   gets a partial summary in the same JSON schema, with no child_summaries
#   context.
# - MERGE: when partials exist, one final call merges them with the child
#   summaries into the final JSON.

_SCHEMA_BLOCK = """\
{
  "purpose": "one-sentence description of what this directory does as a whole",
  "key_exports": ["main functions, classes, or modules exported"],
  "frameworks": ["libraries or frameworks used"],
  "patterns": ["design patterns or architectural patterns observed"],
  "children_overview": "brief summary of how direct child directories relate to each other; empty string \\"\\" if there are no child directories"
}"""

# ---- single-shot --------------------------------------------------------

DIR_SUMMARY_SYSTEM = f"""\
You are a code analysis expert. You will analyze a single directory and produce a structured JSON summary.

You will receive in the user message:
- The full source of every file directly inside this directory, fenced by `----- FILE: ... -----` markers. (Very large files may have a head + tail with a `... [truncated N tokens] ...` marker in the middle.)
- The list of direct child directories.
- Pre-computed summaries of each direct child directory (treat them as authoritative; you cannot inspect their files).

Output ONLY one JSON object matching this schema, no prose, no markdown fences:

{_SCHEMA_BLOCK}

Synthesize the direct files (concrete evidence) with the child summaries (already-condensed downstream context) to fill `purpose`, `key_exports`, `frameworks`, `patterns`, and `children_overview`."""

DIR_SUMMARY_USER = """\
Directory: {dir_path}

Direct child directories: {direct_dirs}

Child directory summaries (already produced; treat as authoritative):
{child_summaries}

Direct files in this directory ({total_files} files, {total_lines} lines total):

{files_block}

Produce the final JSON summary for `{dir_path}` per the schema in the system prompt. Output ONLY the JSON object."""

# ---- partial (one chunk of direct files) -------------------------------

DIR_SUMMARY_PARTIAL_SYSTEM = f"""\
You are a code analysis expert. You will analyze a SUBSET of files from a directory and produce a partial structured JSON summary covering ONLY the files shown.

You will receive in the user message:
- The full source of a subset of files from one directory, fenced by `----- FILE: ... -----` markers. (Very large files may have a head + tail with a `... [truncated N tokens] ...` marker in the middle.)
- The directory path and which chunk this is (e.g. chunk 2 of 3).

You are NOT shown other files in the directory or any child-directory summaries. Do not speculate about anything outside the files you see.

Output ONLY one JSON object matching this schema, no prose, no markdown fences:

{_SCHEMA_BLOCK}

Scope each field to the files in this chunk:
- `purpose`: what these specific files do as a group.
- `key_exports`: exports defined in these files.
- `frameworks` / `patterns`: those evidenced by these files.
- `children_overview`: leave as the empty string \"\" (the merge step handles children)."""

DIR_SUMMARY_PARTIAL_USER = """\
Directory: {dir_path} (partial chunk {chunk_index} of {chunk_total})

Files in this chunk ({chunk_files} files, {chunk_lines} lines):

{files_block}

Produce the partial JSON summary covering ONLY these files. Output ONLY the JSON object."""

# ---- merge (combine partials + child summaries) ------------------------

DIR_SUMMARY_MERGE_SYSTEM = f"""\
You are merging multiple partial directory summaries into one final summary.

You will receive in the user message:
- N partial JSON summaries, each covering a different subset of the direct files of one directory.
- The list of direct child directories.
- The pre-computed summaries of each direct child directory (authoritative).

Combine these inputs into ONE final summary in this schema, no prose, no markdown fences:

{_SCHEMA_BLOCK}

Rules:
- Deduplicate `key_exports`, `frameworks`, and `patterns` across partials. Keep the most informative wording.
- `purpose` should describe the directory as a whole, integrating evidence from all partials and the child summaries.
- `children_overview` must be synthesized from the child directory summaries (or be the empty string \"\" if there are none)."""

DIR_SUMMARY_MERGE_USER = """\
Directory: {dir_path}

Direct child directories: {direct_dirs}

Child directory summaries (already produced; treat as authoritative):
{child_summaries}

Partial summaries of the direct files ({chunk_total} chunks):
{partial_summaries}

Produce the FINAL merged JSON summary for `{dir_path}`. Output ONLY the JSON object."""
