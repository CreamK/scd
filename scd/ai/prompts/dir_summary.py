# --- Directory Summary (reduce step over file summaries) ---
#
# In the new pipeline the LLM never sees raw file content for directory
# summarization. Instead it receives:
# - the JSON summary of every direct file in the directory (already produced
#   by the file-level map step), and
# - the JSON summary of every direct child directory (already produced by
#   earlier passes of this same step).
#
# Three prompt pairs:
# - SINGLE-SHOT: when all direct file summaries + child summaries fit in one
#   request, produce the final JSON in one call.
# - PARTIAL: when they don't fit, bin-pack file summaries into chunks; each
#   chunk yields a partial JSON in the same schema, no child summaries.
# - MERGE: combine N partials with the child summaries into the final JSON.

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
You are a code analysis expert. You will synthesize a structured JSON summary for a single directory.

You will receive in the user message:
- The pre-computed JSON summary of every direct file in this directory, fenced by `----- FILE SUMMARY: ... -----` markers. Each file summary already contains `purpose`, `exports`, `imports`, `frameworks`, `patterns`, and `key_snippets`.
- The list of direct child directories.
- Pre-computed JSON summaries of each direct child directory (treat them as authoritative; you cannot inspect their files).

You do NOT see any raw source code. Trust the file summaries: they are the only ground truth for files in this directory.

Output ONLY one JSON object matching this schema, no prose, no markdown fences:

{_SCHEMA_BLOCK}

Synthesize across the file summaries and child summaries:
- `purpose`: what this directory does as a whole, deduced from grouped file purposes and child-directory roles.
- `key_exports`: deduplicate exports across all direct files; favor symbols that appear in multiple files or look central.
- `frameworks` / `patterns`: union across files and children, deduplicated.
- `children_overview`: synthesize the child-directory summaries into one sentence; use \"\" if there are no child directories."""

DIR_SUMMARY_USER = """\
Directory: {dir_path}

Direct child directories: {direct_dirs}

Child directory summaries (already produced; treat as authoritative):
{child_summaries}

Direct file summaries ({total_files} files, {total_lines} lines total):

{file_summaries_block}

Produce the final JSON summary for `{dir_path}` per the schema in the system prompt. Output ONLY the JSON object."""

# ---- partial (one chunk of file summaries) -----------------------------

DIR_SUMMARY_PARTIAL_SYSTEM = f"""\
You are a code analysis expert. You will synthesize a PARTIAL structured JSON summary covering a SUBSET of the file summaries from one directory.

You will receive in the user message:
- The pre-computed JSON summary of a subset of direct files from one directory, fenced by `----- FILE SUMMARY: ... -----` markers.
- The directory path and which chunk this is (e.g. chunk 2 of 3).

You are NOT shown other files in the directory or any child-directory summaries. Do not speculate about anything outside the file summaries you see.

Output ONLY one JSON object matching this schema, no prose, no markdown fences:

{_SCHEMA_BLOCK}

Scope each field to the file summaries in this chunk:
- `purpose`: what these specific files do as a group.
- `key_exports`: union of exports across these files (dedupe).
- `frameworks` / `patterns`: those evidenced by these files.
- `children_overview`: leave as the empty string \"\" (the merge step handles children)."""

DIR_SUMMARY_PARTIAL_USER = """\
Directory: {dir_path} (partial chunk {chunk_index} of {chunk_total})

File summaries in this chunk ({chunk_files} files, {chunk_lines} lines):

{file_summaries_block}

Produce the partial JSON summary covering ONLY these files. Output ONLY the JSON object."""

# ---- merge (combine partials + child summaries) ------------------------

DIR_SUMMARY_MERGE_SYSTEM = f"""\
You are merging multiple partial directory summaries into one final summary.

You will receive in the user message:
- N partial JSON summaries, each derived from a different subset of the file summaries of one directory.
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

Partial summaries derived from the file summaries ({chunk_total} chunks):
{partial_summaries}

Produce the FINAL merged JSON summary for `{dir_path}`. Output ONLY the JSON object."""
