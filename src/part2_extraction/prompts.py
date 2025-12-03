"""
Prompt Templates - Different strategies for event extraction.

Strategies:
- zero_shot: Direct extraction without examples
- cot: Chain-of-thought reasoning
- few_shot: With examples (requires example data)
"""

from src.part2_extraction.config import EventConfig

SYSTEM_PROMPT = """You are a historian analyzing primary and secondary sources about Abraham Lincoln. 
Your task is to extract factual claims about specific historical events from the provided text.

Be precise and factual. Only extract claims explicitly stated or strongly implied in the text.
Do not infer or add information not present in the source."""


def get_extraction_prompt(
    event: EventConfig,
    text: str,
    strategy: str = "cot"
) -> list[dict]:
    """Build messages for extraction based on strategy."""
    
    if strategy == "zero_shot":
        return _zero_shot_prompt(event, text)
    elif strategy == "cot":
        return _cot_prompt(event, text)
    elif strategy == "few_shot":
        return _few_shot_prompt(event, text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _zero_shot_prompt(event: EventConfig, text: str) -> list[dict]:
    """Direct extraction without reasoning."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Extract information about "{event.name}" from the following text.

TEXT:
{text}

Respond with JSON only:
{{
  "claims": ["claim 1", "claim 2", ...],
  "temporal_details": {{"date": "if mentioned", "time": "if mentioned"}},
  "tone": "Sympathetic | Critical | Neutral | Admiring | Defensive | Other"
}}

If the event is not mentioned, return {{"claims": [], "temporal_details": null, "tone": "Neutral"}}"""}
    ]


def _cot_prompt(event: EventConfig, text: str) -> list[dict]:
    """Chain-of-thought extraction with reasoning."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Extract information about "{event.name}" from the following text.

TEXT:
{text}

---

Think step by step:

1. SCAN: Does this text discuss "{event.name}"? Look for relevant names, dates, locations.

2. EXTRACT CLAIMS: List each factual claim about this event. A claim should be:
   - A single, specific assertion
   - Directly stated or strongly implied in the text
   - About actions, decisions, people, or outcomes related to the event

3. TEMPORAL DETAILS: Note any dates or times mentioned for this event.

4. ASSESS TONE: How does the author portray Lincoln in relation to this event?
   - Sympathetic: Defends or justifies Lincoln's actions
   - Critical: Questions or criticizes Lincoln's decisions
   - Neutral: Factual without judgment
   - Admiring: Praises Lincoln's character/wisdom
   - Defensive: Responds to criticism of Lincoln
   - Other: None of the above fit

After your analysis, provide your final answer as JSON:
```json
{{
  "claims": ["claim 1", "claim 2", ...],
  "temporal_details": {{"date": "...", "time": "..."}},
  "tone": "..."
}}
```

If the event is not discussed in this text, return:
```json
{{"claims": [], "temporal_details": null, "tone": "Neutral"}}
```"""}
    ]


def _few_shot_prompt(event: EventConfig, text: str) -> list[dict]:
    """Extraction with examples."""
    
    example_text = """The attack on Fort Sumter came at half past four in the morning on April 12, 1861. 
Lincoln had deliberately chosen to resupply rather than reinforce, knowing this would force 
the Confederacy to either back down or fire the first shot. His cabinet had been divided, 
with Seward opposing any action that might provoke war."""

    example_output = """{
  "claims": [
    "The attack on Fort Sumter occurred at 4:30 AM on April 12, 1861",
    "Lincoln chose to resupply rather than reinforce Fort Sumter",
    "Lincoln's strategy was designed to force the Confederacy to fire first",
    "Lincoln's cabinet was divided on the Fort Sumter decision",
    "Seward opposed actions that might provoke war"
  ],
  "temporal_details": {"date": "April 12, 1861", "time": "4:30 AM"},
  "tone": "Sympathetic"
}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Extract information about "Fort Sumter Decision" from this text:

TEXT:
{example_text}

Respond with JSON containing claims, temporal_details, and tone."""},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": f"""Now extract information about "{event.name}" from this text:

TEXT:
{text}

Respond with JSON only:
{{
  "claims": ["claim 1", "claim 2", ...],
  "temporal_details": {{"date": "if mentioned", "time": "if mentioned"}},
  "tone": "Sympathetic | Critical | Neutral | Admiring | Defensive | Other"
}}

If the event is not mentioned, return {{"claims": [], "temporal_details": null, "tone": "Neutral"}}"""}
    ]