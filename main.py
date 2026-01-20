import json
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from tqdm import tqdm

from dashboard import run as run_dashboard
from event_tree import create_event_tree

FILE_PATH = "hiad.xlsx"
GPTOSS = "gpt-oss:20b"
COUNT = 2000

LLM_PROVIDER = "ollama"  # Set to "gemini" to use the Gemini API.
SELECTED_LLM = GPTOSS
LLM_THINKING = True
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_API_KEY = ""
_GEMINI_CLIENT = None
_GEMINI_TYPES = None

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "continuous_release": {"type": "boolean"},
        "continuous_release_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "immediate_ignition": {"type": "boolean"},
        "immediate_ignition_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "barrier_stopped_immediate_ignition": {"type": "boolean"},
        "delayed_ignition": {"type": "boolean"},
        "delayed_ignition_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "barrier_stopped_delayed_ignition": {"type": "boolean"},
        "confined_space": {"type": "boolean"},
        "confined_space_confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "exclude_not_pure_h2": {"type": "boolean"},
        "exclude_not_gaseous_h2": {"type": "boolean"},
        "exclude_no_loc": {"type": "boolean"},
    },
    "required": [
        "continuous_release",
        "continuous_release_confidence",
        "immediate_ignition",
        "immediate_ignition_confidence",
        "barrier_stopped_immediate_ignition",
        "delayed_ignition",
        "delayed_ignition_confidence",
        "barrier_stopped_delayed_ignition",
        "confined_space",
        "confined_space_confidence",
        "exclude_not_pure_h2",
        "exclude_not_gaseous_h2",
        "exclude_no_loc",
    ],
}


def _suppress_openpyxl_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="openpyxl",
        message="Conditional Formatting extension is not supported",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="openpyxl",
        message="Data Validation extension is not supported",
    )


def _clean_value(value):
    """Normalize cell values and drop empty content."""
    if pd.isna(value):
        return None
    text = str(value).replace("_x000D_", "\n").strip()
    return text or None


def _gemini_client():
    global _GEMINI_CLIENT, _GEMINI_TYPES
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT, _GEMINI_TYPES
    if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_GEMINI_API_KEY_HERE":
        raise ValueError("Set GEMINI_API_KEY before using Gemini.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Gemini selected but google-genai is not installed. Install with: pip install google-genai"
        ) from exc
    _GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    _GEMINI_TYPES = types
    return _GEMINI_CLIENT, _GEMINI_TYPES


def _selected_model():
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL
    return SELECTED_LLM


def _gemini_config(types, system_prompt: str | None):
    thinking_level = "HIGH" if LLM_THINKING else "NONE"
    return types.GenerateContentConfig(
        system_instruction=system_prompt or "",
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )


def ask(prompt: str, system_prompt: str | None = None):
    if LLM_PROVIDER == "ollama":
        import ollama

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=SELECTED_LLM,
            think=LLM_THINKING,
            format=RESPONSE_SCHEMA,
            messages=messages,
        )

        message = response.get("message", {})
        content = message.get("content", "")
        reasoning = (
            message.get("thinking")
            or message.get("reasoning")
            or message.get("metadata", {}).get("thinking")
            or response.get("thinking")
            or response.get("metadata", {}).get("thinking")
            or ""
        )
    elif LLM_PROVIDER == "gemini":
        client, types = _gemini_client()
        config = _gemini_config(types, system_prompt)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        content = response.text or ""
        if not content and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            parts = getattr(candidate.content, "parts", []) if candidate.content else []
            content = "".join(part.text for part in parts if getattr(part, "text", None))
        reasoning = ""
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON content: {content}") from exc
    return parsed, reasoning


def read_enriched_events(path: str, n: int):
    _suppress_openpyxl_warnings()
    sheets = [
        "EVENTS",
        "FACILITY",
        "CONSEQUENCES",
        "LESSONS LEARNT",
        "EVENT NATURE",
        "REFERENCES",
        "Classes  & Labels",
    ]

    merged = None
    merged_cols: set[str] = set()

    for sheet in sheets:
        df = pd.read_excel(path, sheet_name=sheet)
        if "Event ID" not in df.columns:
            continue  # Skip sheets that cannot be merged.
        if merged is None:
            merged = df
        else:
            # Drop duplicate columns (keep the first occurrence seen).
            dup_cols = [col for col in df.columns if col in merged_cols and col != "Event ID"]
            df = df.drop(columns=dup_cols)
            merged = merged.merge(df, on="Event ID", how="left")
        merged_cols = set(merged.columns)

    if merged is None:
        raise ValueError("No sheets loaded from Excel file.")

    merged = merged[merged["Event full description"].notna()].head(n)
    return merged


def build_event_markdown(row: Series) -> str:
    fields = []

    def add(label, value):
        cleaned = _clean_value(value)
        if cleaned:
            fields.append(f"- **{label}:** {cleaned}")

    title = _clean_value(row.get("Event Title")) or "Untitled Event"
    header = f"# {title}"

    # Include all available fields except the full description (which is shown separately)
    # and the title (already used as the header).
    for col in row.index:
        if col in ("Event full description", "Event Title"):
            continue
        add(col, row.get(col))

    description = _clean_value(row.get("Event full description")) or "No description available."

    sections = [header, ""]
    if fields:
        sections.append("## Key Facts")
        sections.extend(fields)
        sections.append("")
    sections.append("## Description")
    sections.append(description)
    return "\n".join(sections)



def process_events(log = True, save_path=None):
    events = read_enriched_events(FILE_PATH, COUNT)
    results = []

    system_prompt = """Fill every field in the JSON schema. For each question not about barriers or exclusion, provide a boolean answer and an integer confidence from 0-10 (0 = no information in the description to decide; 10 = the description makes the chosen answer unquestionably clear). Schema: {continuous_release:boolean, continuous_release_confidence:int, immediate_ignition:boolean, immediate_ignition_confidence:int, barrier_stopped_immediate_ignition:boolean, delayed_ignition:boolean, delayed_ignition_confidence:int, barrier_stopped_delayed_ignition:boolean, confined_space:boolean, confined_space_confidence:int, exclude_not_pure_h2:boolean, exclude_not_gaseous_h2:boolean, exclude_no_loc:boolean}. Use the provided event details to decide.

Continuous release rubric: Mark true if hydrogen flow persisted over time rather than a single brief discharge.
Immediate ignition rubric: Mark true if ignition occurred at the moment of release or within seconds without delay.
Delayed ignition rubric: Mark true if a flammable cloud formed and ignited after a noticeable delay from the release.
Barrier (immediate) rubric: If immediate_ignition is true, barrier_stopped_immediate_ignition must be false. If immediate_ignition is false, set barrier_stopped_immediate_ignition to true only when a barrier meaningfully prevented immediate ignition (e.g., ESD systems, isolation valves, emergency shutdowns); otherwise set it to false.
Barrier (delayed) rubric: If delayed_ignition is true, barrier_stopped_delayed_ignition must be false. If delayed_ignition is false, set barrier_stopped_delayed_ignition to true only when a barrier meaningfully prevented delayed ignition (e.g., ESD, inerting, venting, isolation); otherwise set it to false.
Confined space rubric: Mark true if the release occurred in an enclosed or poorly ventilated area that limits dispersion.
Exclude not pure H2 rubric: Mark true if the release substance is a hydrogen mixture with significant non-hydrogen components or is not primarily hydrogen.
Exclude not gaseous H2 rubric: Mark true if hydrogen was released in a non-gaseous state (e.g., liquid or solid hydrogen) or the release medium is not gaseous H2.
Exclude no loss of containment rubric: Mark true when no hydrogen was released; set false if any amount of hydrogen actually leaked.
"""

    for _, row in tqdm(events.iterrows(), total=len(events), desc="Processing events"):
        markdown = build_event_markdown(row)
        description = _clean_value(row.get("Event full description")) or "No description available."
        prompt = f"""Use the event details to determine the answers for each question.

Event details:

{markdown}
"""
        if log:
            tqdm.write(markdown + "\n")
            tqdm.write("==========================================")
        result, reasoning = ask(prompt=prompt, system_prompt=system_prompt)
        record = {
            "event_id": _clean_value(row.get("Event ID")),
            "title": _clean_value(row.get("Event Title")) or "Untitled Event",
            "description": description,
            "user_prompt": prompt,
            "system_prompt": system_prompt,
            "model": _selected_model(),
            "thinking": LLM_THINKING,
            "reasoning": reasoning,
        }
        record.update(result)
        results.append(record)

        if log:
            tqdm.write(str(result))
            tqdm.write("==========================================")

    if save_path is not None:
        save_events(results, path=save_path)

    return results


def save_events(events, path = "events.json"):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(events, handle, ensure_ascii=True, indent=2)


def load_events(path = "events.json"):
    with open(path, "r", encoding="utf-8") as handle:
        events = json.load(handle)
    return events


if __name__ == "__main__":
    EVENTS_PATH = "events.json"
    
    #events = process_events(save_path=EVENTS_PATH)

    events = load_events(path=EVENTS_PATH)
    SAVE_FIGS = True
    create_event_tree(events, save=SAVE_FIGS, show_exclusion=False, filename="event_tree.png")
    create_event_tree(events, save=SAVE_FIGS, show_exclusion=True, filename="event_tree_exclusions.png")
    run_dashboard(events_path=EVENTS_PATH, port=4000)
