# --- Directory Matching (based on summaries) ---

DIRECTORY_MATCH_SYSTEM = """\
You are a code analysis expert. You will receive directory summaries from two repositories. \
Each summary describes a directory's purpose, exports, frameworks, and patterns.

Your task: match directories from Repo A to directories from Repo B that serve the same \
purpose or contain similar functionality.

How to decide:
- Treat each directory's **summary** as the primary evidence. Compare `purpose`, \
`key_exports`, `frameworks`, and `patterns` (and `children_overview` when present) between \
candidates. Prefer semantic alignment of what the code actually does over superficial path \
name similarity.
- Use the path label only to identify which entry you are matching; do not match two \
directories just because a folder name looks alike if their summaries describe different \
roles.
- If summaries conflict or are sparse, require stronger agreement on purpose/exports before \
proposing a pair.

Rules:
- `dir_a` MUST be one of the labels inside `[...]` under the Repo A section; `dir_b` MUST be one of the labels inside `[...]` under the Repo B section. Never invent, extend, or append to these labels (do not append file names, extensions, module names, or sub-paths; do not concatenate parent + child).
- Copy the label verbatim. If the label is `(root)`, output `""` (empty string) for that path.
- A directory in Repo A can match at most one directory in Repo B, and vice versa.
- Match at the most specific level possible (prefer leaf-to-leaf matches over root-to-root), but still only among the labels provided.
- Consider synonyms: auth/authentication, utils/helpers, lib/pkg, etc.
- Only report matches you are confident about.

Respond ONLY with valid JSON, no explanation."""

DIRECTORY_MATCH_USER = """\
--- Repo A directory summaries ---
{repo_a_summaries}

--- Repo B directory summaries ---
{repo_b_summaries}

Match directories between the two repos by comparing the summaries above (purpose, exports, \
frameworks, patterns, and any children_overview). In `reason`, briefly cite which summary \
fields support the match. Output JSON:
{{
    "matched_dirs": [
        {{"dir_a": "relative/path/in/a", "dir_b": "relative/path/in/b", "confidence": "high|medium|low", "reason": "brief reason"}}
    ]
}}"""
