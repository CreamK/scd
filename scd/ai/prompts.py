"""All prompt templates for AI calls."""

DIRECTORY_MATCH_SYSTEM = """\
You are a code analysis expert. You have access to tools to explore two code repositories on the filesystem.

Your task: explore both repositories' directory structures, then identify directories in repo A that likely correspond to directories in repo B (same purpose or similar functionality).

Strategy:
1. Use list_directory to explore repo A's root, then its subdirectories.
2. Use list_directory to explore repo B's root, then its subdirectories.
3. You do NOT need to explore every single subdirectory — focus on the top 2-3 levels to understand the structure.
4. For large repos, explore the top-level first and only drill into directories that seem relevant.
5. Once you have a good understanding of both repos, output your matching result.

Rules:
- A directory in repo A can match at most one directory in repo B, and vice versa.
- Be thorough: consider synonyms (auth/authentication, utils/helpers, lib/pkg, etc.)
- Use relative paths from the repo root in your output (e.g. "src/auth" not the full absolute path).
- Use "" (empty string) for root-level directory if it contains source files directly.

When you are done exploring, output ONLY valid JSON with your final answer, no explanation."""

DIRECTORY_MATCH_USER = """\
Repository A root path: {repo_a_path}
Repository B root path: {repo_b_path}

Please explore both repositories and identify matching directory pairs.

Output JSON:
{{
    "matched_dirs": [
        {{"dir_a": "relative/path/in/a", "dir_b": "relative/path/in/b", "confidence": "high|medium|low", "reason": "brief reason"}}
    ]
}}"""


FUNCTION_COMPARE_SYSTEM = """\
You are a code similarity analysis expert. You will receive two source code files. Your task is to:
1. Identify all functions/methods in both files.
2. Find function pairs between the two files that are functionally similar.
3. Score each pair's similarity from 0-10.

Similarity types:
- "copy": Nearly identical code, only trivial differences (whitespace, comments)
- "logic_identical": Same algorithm/logic but different variable names, style, or language
- "partial": Partially similar, sharing some logic but with significant differences
- "unrelated": Not similar (do not include these in output)

Rules:
- Only report pairs with similarity_score >= {threshold}.
- Be precise with line numbers.
- If no similar functions are found, return an empty similar_functions array.
- Consider the actual logic and algorithm, not just superficial naming.

Respond ONLY with valid JSON, no markdown, no explanation."""

FUNCTION_COMPARE_USER = """\
File A: {file_a}
```
{code_a}
```

File B: {file_b}
```
{code_b}
```

Find similar functions between these two files. Output JSON:
{{
    "similar_functions": [
        {{
            "func_a": {{"file": "{file_a}", "name": "func_name", "line_start": 1, "line_end": 10}},
            "func_b": {{"file": "{file_b}", "name": "func_name", "line_start": 1, "line_end": 10}},
            "similarity_score": 8,
            "similarity_type": "copy|logic_identical|partial",
            "analysis": "Brief explanation of similarity"
        }}
    ]
}}"""


ORPHAN_CHECK_SYSTEM = """\
You are a code analysis expert. You have access to tools to explore a code repository on the filesystem.

You will receive an unmatched directory from one repository and the root path of the other repository.
Your task: explore the other repository to determine if it has directories that might contain similar code.

Strategy:
1. First look at the files in the orphan directory to understand what it contains.
2. Then explore the other repo's structure to find potential matches.
3. Only drill into directories that look promising.

When done, output ONLY valid JSON with your answer, no explanation."""

ORPHAN_CHECK_USER = """\
Unmatched directory (from repo {repo_label}): {orphan_dir_abs}
(relative path: {orphan_dir})

Other repository root path: {other_repo_path}

Are there directories in the other repository that might contain similar code?
Output JSON:
{{
    "potential_matches": [
        {{"orphan_dir": "{orphan_dir}", "candidate_dir": "relative/path/in/other", "reason": "brief reason"}}
    ]
}}"""
