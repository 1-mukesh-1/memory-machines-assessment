"""
Judge Prompt Templates - Zero-shot, Chain-of-Thought, Few-shot strategies.
"""

SYSTEM_PROMPT = """You are a historical accuracy evaluator comparing first-person accounts by Abraham Lincoln with third-party biographer accounts of the same events.

Your task is to assess CONSISTENCY - how well the biographer's claims align with Lincoln's own words.

CONTRADICTION TYPES:
- FACTUAL: Disagreement on verifiable facts (dates, numbers, names, locations, sequences)
- INTERPRETIVE: Different characterization of motivations, emotions, or significance  
- OMISSION: Important information in one source that contradicts or undermines the other

SCORING RUBRIC:
- 0-20: Major factual contradictions that fundamentally misrepresent the event
- 21-40: Multiple factual errors or significant interpretive disagreements
- 41-60: Some factual discrepancies or notable interpretive differences
- 61-80: Minor discrepancies, mostly aligned with some interpretive variance
- 81-100: Strong alignment, differences are trivial or complementary details

NOTE: Biographers adding context Lincoln didn't mention is NOT a contradiction.
Only flag omissions when one source contradicts or undermines the other."""


def format_claims(lincoln_claims: list[str], other_claims: list[str], author: str) -> str:
    """Format claims for prompt input."""
    lincoln_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(lincoln_claims))
    other_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(other_claims))
    
    return f"""LINCOLN'S CLAIMS:
{lincoln_text}

{author.upper()}'S CLAIMS:
{other_text}"""


def get_judge_prompt(
    event_name: str,
    lincoln_claims: list[str],
    other_claims: list[str],
    other_author: str,
    strategy: str = "cot"
) -> list[dict]:
    """Build messages for judgment based on strategy."""
    
    if strategy == "zero_shot":
        return _zero_shot_prompt(event_name, lincoln_claims, other_claims, other_author)
    elif strategy == "cot":
        return _cot_prompt(event_name, lincoln_claims, other_claims, other_author)
    elif strategy == "few_shot":
        return _few_shot_prompt(event_name, lincoln_claims, other_claims, other_author)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _zero_shot_prompt(
    event_name: str,
    lincoln_claims: list[str],
    other_claims: list[str],
    other_author: str
) -> list[dict]:
    """Direct judgment without reasoning."""
    claims_text = format_claims(lincoln_claims, other_claims, other_author)
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Compare these accounts of "{event_name}":

{claims_text}

Respond with JSON only:
{{
  "consistency_score": <0-100>,
  "contradictions": [
    {{
      "type": "factual|interpretive|omission",
      "lincoln_claim": "...",
      "other_claim": "...",
      "explanation": "..."
    }}
  ]
}}

If no contradictions, return empty array for contradictions."""}
    ]


def _cot_prompt(
    event_name: str,
    lincoln_claims: list[str],
    other_claims: list[str],
    other_author: str
) -> list[dict]:
    """Chain-of-thought judgment with reasoning."""
    claims_text = format_claims(lincoln_claims, other_claims, other_author)
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Compare these accounts of "{event_name}":

{claims_text}

---

Think step by step:

1. ALIGNMENT CHECK: Which claims from {other_author} align with Lincoln's account?

2. CONTRADICTION SCAN: Are there any direct factual disagreements? 
   Look for conflicting dates, names, numbers, or sequences.

3. INTERPRETATION DIFFERENCES: Does {other_author} characterize motivations 
   or significance differently than Lincoln's account suggests?

4. SIGNIFICANT OMISSIONS: Does either source omit something that contradicts 
   or undermines the other? (Adding context is NOT an omission.)

5. SCORE: Based on the above, assign a consistency score (0-100).

After your analysis, provide your final answer as JSON:
```json
{{
  "consistency_score": <0-100>,
  "contradictions": [
    {{
      "type": "factual|interpretive|omission",
      "lincoln_claim": "exact claim or null",
      "other_claim": "exact claim or null", 
      "explanation": "why this is a contradiction"
    }}
  ],
  "reasoning": "brief summary of your analysis"
}}
```

If no contradictions found, return empty array for contradictions."""}
    ]


def _few_shot_prompt(
    event_name: str,
    lincoln_claims: list[str],
    other_claims: list[str],
    other_author: str
) -> list[dict]:
    """Judgment with examples."""
    
    # Example 1: High consistency
    example1_input = """LINCOLN'S CLAIMS:
  1. The address was delivered at Gettysburg.
  2. The nation was engaged in a great civil war.
  3. The ground was consecrated by the brave men who fought there.

BIOGRAPHER'S CLAIMS:
  1. Lincoln delivered the address on November 19, 1863.
  2. Edward Everett spoke for two hours before Lincoln.
  3. The speech emphasized that the nation was testing whether liberty could endure.
  4. Lincoln stated the ground was consecrated by the soldiers' sacrifice."""

    example1_output = """{
  "consistency_score": 92,
  "contradictions": [],
  "reasoning": "All facts align. Biographer adds date and context (Everett) that Lincoln's summary doesn't include, but nothing contradicts. The themes of civil war and consecration match perfectly."
}"""

    # Example 2: With contradiction
    example2_input = """LINCOLN'S CLAIMS:
  1. Chew arrived in Charleston on April 8, 1861.
  2. Fort Sumter had not been attacked at that time.

BIOGRAPHER'S CLAIMS:
  1. The messenger arrived on April 6, 1861.
  2. Lincoln sent the notification before the attack.
  3. The attack began April 12, 1861."""

    example2_output = """{
  "consistency_score": 65,
  "contradictions": [
    {
      "type": "factual",
      "lincoln_claim": "Chew arrived in Charleston on April 8, 1861.",
      "other_claim": "The messenger arrived on April 6, 1861.",
      "explanation": "Two-day discrepancy in arrival date."
    }
  ],
  "reasoning": "One factual disagreement on the arrival date. Other claims about the timeline (attack on April 12) are consistent with Lincoln's claim that the fort had not been attacked when Chew arrived."
}"""

    claims_text = format_claims(lincoln_claims, other_claims, other_author)
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Compare these accounts of "Gettysburg Address":

{example1_input}

Respond with JSON containing consistency_score, contradictions array, and reasoning."""},
        {"role": "assistant", "content": example1_output},
        {"role": "user", "content": f"""Compare these accounts of "Fort Sumter":

{example2_input}

Respond with JSON containing consistency_score, contradictions array, and reasoning."""},
        {"role": "assistant", "content": example2_output},
        {"role": "user", "content": f"""Compare these accounts of "{event_name}":

{claims_text}

Respond with JSON containing consistency_score, contradictions array, and reasoning."""}
    ]