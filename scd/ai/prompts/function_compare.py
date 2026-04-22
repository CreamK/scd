# --- Function Comparison ---

FUNCTION_COMPARE_SYSTEM = """\
You are a code similarity analysis expert. You will receive two source code files. Your task is to:
1. Identify all functions/methods in both files.
2. Find function pairs between the two files that are similar.
3. Evaluate each pair across 5 dimensions, scoring each from 0 to 100.

Evaluation dimensions and weights:

1. data_structure (25%): Compare core data structures — field names, types, organization, struct/class layouts. Score high if the structures carry the same semantic fields in similar arrangements.

2. function_signature (25%): Compare function/method names, parameter lists (count, types, order), and return types. Score high if signatures suggest the same interface contract.

3. algorithm_logic (25%): Compare the core algorithm and processing flow — control flow, key operations, computational steps. Score high if the processing logic follows the same approach.

4. naming_convention (15%): Compare naming style of variables, functions, macros, constants — e.g. camelCase vs snake_case, prefix conventions, abbreviation patterns. Score high if the naming style is consistent.

5. protocol_conformance (10%): Compare external interfaces, protocol formats, API contracts, data serialization. Score high if they conform to the same protocol or standard.

Composite score = data_structure*0.25 + function_signature*0.25 + algorithm_logic*0.25 + naming_convention*0.15 + protocol_conformance*0.10

Similarity levels based on composite score:
- "high": > 60%
- "medium": 40-60%
- "low": 20-40%
- "very_low": < 20%

Rules:
- Only report pairs with composite score >= {threshold}.
- Be precise with line numbers.
- If no similar functions are found, return an empty similar_functions array.
- Evaluate based on actual code logic and semantics, not superficial text similarity.

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
            "scores": {{
                "data_structure": 75,
                "function_signature": 80,
                "algorithm_logic": 60,
                "naming_convention": 50,
                "protocol_conformance": 70
            }},
            "composite_score": 68,
            "similarity_level": "high",
            "analysis": "Brief explanation of similarity across dimensions"
        }}
    ]
}}"""
