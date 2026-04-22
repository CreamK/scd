# --- Directory Summary (leaf directories) ---

DIR_SUMMARY_LEAF_SYSTEM = """\
You are a code analysis expert. You will receive all source files from a single directory. \
Analyze the code and produce a structured summary.

Respond ONLY with valid JSON, no explanation."""

DIR_SUMMARY_LEAF_USER = """\
Directory: {dir_path}
Files ({file_count}):

{file_contents}

Produce a JSON summary of this directory:
{{
    "purpose": "one-sentence description of what this directory does",
    "key_exports": ["main functions, classes, or modules exported"],
    "frameworks": ["libraries or frameworks used"],
    "patterns": ["design patterns or architectural patterns observed"]
}}"""

# --- Directory Summary (parent directories with child summaries) ---

DIR_SUMMARY_PARENT_SYSTEM = """\
You are a code analysis expert. You will receive:
1. Summaries of child subdirectories (already analyzed)
2. Source files that belong directly to this directory (not in subdirectories)

Synthesize a higher-level summary that captures the overall purpose of this directory, \
incorporating what its children do and what its own files do.

Respond ONLY with valid JSON, no explanation."""

DIR_SUMMARY_PARENT_USER = """\
Directory: {dir_path}

Child directory summaries:
{child_summaries}

Direct files in this directory ({file_count}):
{file_contents}

Produce a JSON summary of this directory (incorporating child summaries):
{{
    "purpose": "one-sentence description of what this directory does as a whole",
    "key_exports": ["main functions, classes, or modules exported"],
    "frameworks": ["libraries or frameworks used across this directory tree"],
    "patterns": ["design patterns or architectural patterns observed"],
    "children_overview": "brief summary of how child directories relate to each other"
}}"""
