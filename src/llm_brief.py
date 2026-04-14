"""
llm_brief.py
-------------
Generates a human-readable AI operations brief using the Google Gemini API.

If GEMINI_API_KEY is not set, falls back to a template-based brief so the
app still works for offline demos.

The brief contains:
  - executive_summary  : 2-3 sentence overview
  - top_risks          : list of top 5 risk nodes with explanations
  - immediate_actions  : list of 4-5 recommended actions
  - confidence         : model confidence (high/medium/low)
  - estimated_impact   : business impact estimate
  - timeline           : suggested response timeline
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars

GEMINI_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")
GROQ_KEY_ENV_VARS = ("GROQ_API_KEY",)


def _gemini_generate(api_key: str, prompt: str, temperature: float = 0.3, max_tokens: int = 8192) -> str:
    """
    Call Gemini, trying the new google-genai SDK first, then falling back to
    the legacy google-generativeai SDK so the app works in any environment.

    Uses response_mime_type="application/json" to force pure JSON output and
    a higher token budget because gemini-2.5-flash is a thinking model that
    consumes output tokens for internal reasoning before writing the response.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        cfg = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        # Disable thinking for structured JSON tasks (saves tokens & latency).
        try:
            cfg["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
        except AttributeError:
            pass  # older SDK version — skip
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(**cfg),
        )
        return response.text.strip()
    except ImportError:
        import google.generativeai as _genai
        _genai.configure(api_key=api_key)
        model = _genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
        )
        return response.text.strip()


def _resolve_api_key(explicit_key: Optional[str], env_names: Tuple[str, ...]) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    for env_name in env_names:
        candidate = os.getenv(env_name, "").strip()
        if candidate and not candidate.lower().startswith("your_"):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Template brief (offline fallback)
# ---------------------------------------------------------------------------
def template_brief(
    disruption_info:     Dict[str, Any],
    risk_df:             pd.DataFrame,
    reroute_suggestions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a structured operations brief WITHOUT calling any API.
    Used when no Gemini API key is configured.
    """
    event_text = disruption_info.get("event_text", "Unknown disruption")
    severity   = disruption_info.get("severity", "medium").upper()
    category   = disruption_info.get("category", "other").replace("_", " ").title()
    n_affected = len(risk_df)

    top_nodes = risk_df.head(5)

    # Build risk bullets
    risk_bullets = []
    for _, row in top_nodes.iterrows():
        risk_bullets.append(
            f"{row['city_name']}, {row['country']} "
            f"[{row['product_category']}] — Risk Score: {row['risk_score']:.2f} "
            f"({row['risk_level']})"
        )

    # Build action items
    actions = [
        f"🚨 Activate emergency protocols for {severity} severity {category} event.",
        f"📊 Monitor {n_affected} affected supply chain nodes in real-time.",
    ]
    if reroute_suggestions:
        found = [r for r in reroute_suggestions if r["status"] == "✅ Alternate Found"]
        if found:
            actions.append(
                f"🔁 Immediately reroute {len(found)} critical shipment lanes "
                f"via identified alternate paths."
            )
    if not risk_df.empty:
        top_country = risk_df.iloc[0]["country"]
        actions.append(f"📞 Contact backup suppliers outside {top_country} for critical components.")
    actions.append("📋 Issue formal supplier risk assessment within 24 hours.")

    # Confidence based on severity
    confidence_map = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "High"}
    confidence = confidence_map.get(severity, "Medium")

    # Impact estimate
    impact = {
        "HIGH":   "Severe — major production slowdowns expected within 24-48 hours.",
        "MEDIUM": "Moderate — partial disruption with 3-7 day recovery window.",
        "LOW":    "Minor — localised delays of 1-3 days expected.",
    }.get(severity, "To be assessed.")

    return {
        "executive_summary": (
            f"A {severity.lower()}-severity {category.lower()} event has been detected: "
            f"'{event_text}'. "
            f"Analysis indicates {n_affected} supply chain nodes are at risk across "
            f"{risk_df['country'].nunique()} countries. "
            f"Immediate action is recommended to mitigate cascading disruptions."
        ),
        "top_risks":         risk_bullets,
        "immediate_actions": actions,
        "confidence":        confidence,
        "estimated_impact":  impact,
        "timeline":          _get_timeline(severity),
        "source":            "template",
    }


def _get_timeline(severity: str) -> str:
    timelines = {
        "HIGH":   "0-6 hours: Activate crisis team | 6-24h: Reroute critical lanes | 1-7 days: Supplier negotiations",
        "MEDIUM": "0-24 hours: Assessment | 1-3 days: Alternate sourcing | 1-2 weeks: Recovery",
        "LOW":    "1-3 days: Monitor & assess | 1-2 weeks: Contingency planning",
    }
    return timelines.get(severity.upper(), "Assess within 48 hours.")


# ---------------------------------------------------------------------------
# Gemini API brief
# ---------------------------------------------------------------------------
def _build_prompt(
    disruption_info:     Dict[str, Any],
    risk_df:             pd.DataFrame,
    reroute_suggestions: List[Dict[str, Any]],
    shap_context:        Optional[str] = None,
) -> str:
    """Build the prompt string sent to Gemini."""
    event        = disruption_info.get("event_text", "Unknown")
    severity     = disruption_info.get("severity", "medium")
    category     = disruption_info.get("category", "other")
    n_nodes      = len(risk_df)
    top5         = risk_df.head(5)[["city_name", "country", "product_category", "risk_score", "risk_level"]].to_string(index=False)
    n_alternates = len([r for r in reroute_suggestions if r["status"] == "✅ Alternate Found"])

    shap_section = ""
    if shap_context:
        shap_section = f"""

ML EXPLAINABILITY — SHAP FEATURE ANALYSIS (why the top node is at risk):
{shap_context}
Use these specific feature drivers when explaining risks in your brief.
"""

    prompt = f"""You are a senior supply chain risk analyst. Generate a structured operations brief.

DISRUPTION EVENT:
  Description : {event}
  Severity    : {severity}
  Category    : {category}
  Nodes at risk: {n_nodes}
  Alternate routes found: {n_alternates}

TOP 5 HIGHEST-RISK NODES:
{top5}
{shap_section}
Generate a JSON operations brief with EXACTLY these fields:
{{
  "executive_summary": "<2-3 sentences summarising the situation and urgency. If SHAP data is provided, reference the specific feature drivers.>",
  "top_risks": [
    "<risk 1 with node name, country, and WHY it matters — cite SHAP features if available>",
    "<risk 2>",
    "<risk 3>",
    "<risk 4>",
    "<risk 5>"
  ],
  "immediate_actions": [
    "<action 1 with clear owner and deadline>",
    "<action 2>",
    "<action 3>",
    "<action 4>",
    "<action 5>"
  ],
  "confidence": "<High | Medium | Low>",
  "estimated_impact": "<business impact in plain English>",
  "timeline": "<response timeline with phases>"
}}

Be specific, actionable, and professional. Use the actual node names and countries from the data.
Return ONLY valid JSON, no extra text."""
    return prompt


def generate_brief(
    disruption_info:     Dict[str, Any],
    risk_df:             pd.DataFrame,
    reroute_suggestions: List[Dict[str, Any]],
    api_key:             Optional[str] = None,
    shap_context:        Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate the AI operations brief.

    Tries Gemini API first; falls back to template if API key is missing or call fails.

    Parameters
    ----------
    disruption_info     : dict from disruption_input.parse_disruption()
    risk_df             : DataFrame from risk_scoring.score_nodes()
    reroute_suggestions : list from reroute.find_alternates()
    api_key             : Gemini API key (defaults to GEMINI_API_KEY env var)
    shap_context        : Optional plain-English SHAP summary string for prompt injection

    Returns
    -------
    dict with keys: executive_summary, top_risks, immediate_actions,
                    confidence, estimated_impact, timeline, source
    """
    # Resolve API key — try Gemini first, then Groq
    gemini_key = _resolve_api_key(api_key, GEMINI_KEY_ENV_VARS)
    groq_key = _resolve_api_key(None, GROQ_KEY_ENV_VARS)

    # Try Gemini (single call — reliable even on free tier)
    if gemini_key:
        try:
            import json
            import re

            prompt = _build_prompt(disruption_info, risk_df, reroute_suggestions, shap_context=shap_context)
            print("  [llm_brief] Calling Gemini 2.5 Flash …")
            raw_text = _gemini_generate(gemini_key, prompt, temperature=0.3)
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                brief = json.loads(json_match.group(0))
                brief["source"] = "gemini-2.5-flash"
                if shap_context:
                    brief["shap_explanation"] = shap_context
                print("  [llm_brief] Gemini brief generated successfully")
                return brief
            else:
                raise ValueError("No JSON in Gemini response")
        except Exception as e:
            print(f"  [llm_brief] Gemini error: {e} — trying Groq")

    # Try Groq as fallback
    if groq_key:
        try:
            from groq import Groq
            import json
            import re

            client = Groq(api_key=groq_key)
            prompt = _build_prompt(disruption_info, risk_df, reroute_suggestions, shap_context=shap_context)

            print("  [llm_brief] Calling Groq GPT-OSS 120B …")
            req = {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {"role": "system", "content": "You are a senior supply chain risk analyst. Return only valid JSON."},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0.3,
                "top_p": 1,
                "max_completion_tokens": 4096,
                "reasoning_effort": "medium",
                "stream": False,
            }
            try:
                response = client.chat.completions.create(**req)
            except TypeError:
                # Backward compatibility for older Groq SDKs.
                req.pop("reasoning_effort", None)
                req["max_tokens"] = req.pop("max_completion_tokens")
                response = client.chat.completions.create(**req)
            raw_text = ((response.choices[0].message.content if response.choices else "") or "").strip()
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                brief = json.loads(json_match.group(0))
                brief["source"] = "groq-openai-gpt-oss-120b"
                if shap_context:
                    brief["shap_explanation"] = shap_context
                print("  [llm_brief] Groq brief generated successfully")
                return brief
            raise ValueError("No JSON in Groq response")
        except Exception as e:
            print(f"  [llm_brief] Groq error: {e} — falling back to template")

    print("  [llm_brief] No working AI API key — using template brief")
    brief = template_brief(disruption_info, risk_df, reroute_suggestions)
    if shap_context:
        brief["shap_explanation"] = shap_context
    return brief


# ---------------------------------------------------------------------------
# Agentic Action Execution (Mock ERP)
# ---------------------------------------------------------------------------
def generate_execution_payloads(
    route: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an email to the new supplier and a JSON payload mocking an
    ERP (e.g. SAP/Oracle) system update to execute the reroute.
    """
    gemini_key = _resolve_api_key(api_key, GEMINI_KEY_ENV_VARS)
    
    source_city = route.get("source_name", "Unknown Source")
    dest_city = route.get("destination_name", "Unknown Destination")
    cost = route.get("alt_cost_usd", 50000)
    
    prompt = f"""You are an Autonomous Logistics Execution Agent.
An alternate supply route has been approved to bypass a disruption.
Route details:
- Original Origin: Disrupted
- New Origin: {source_city}
- Destination: {dest_city}
- Estimated Cost: ${cost:,.0f}
- Units required: 500

Generate EXACTLY the following JSON:
{{
  "email_draft": "<A professional email to a generic supplier in {source_city} requesting 500 units of capacity urgently.>",
  "erp_json_payload": {{
     "transaction_type": "PURCHASE_ORDER_UPDATE",
     "po_number": "PO-99482-EMG",
     "supplier": "Generic Supplier {source_city}",
     "origin": "{source_city}",
     "destination": "{dest_city}",
     "units": 500,
     "cost_estimate_usd": {cost},
     "status": "Awaiting Vendor Confirmation"
  }}
}}
Return ONLY valid JSON, no extra text."""

    if gemini_key:
        try:
            import json, re
            raw_text = _gemini_generate(gemini_key, prompt, temperature=0.2)
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                payloads = json.loads(json_match.group(0))
                payloads["erp_json_payload"] = json.dumps(payloads["erp_json_payload"], indent=2)
                return payloads
        except Exception as e:
            print(f"  [llm_brief] Agentic Execution via Gemini failed: {e}")

    # Fallback if no API or error
    import json
    fallback_email = (
        f"Subject: URGENT: Capacity Request for 500 Units\n\n"
        f"Dear Supplier Team in {source_city},\n\n"
        f"We are writing to urgently request your assistance. Due to an active supply chain disruption, "
        f"we require an immediate capacity allocation of 500 units to be routed to {dest_city}.\n\n"
        f"Please find the preliminary order details below:\n"
        f"  - Origin: {source_city}\n"
        f"  - Destination: {dest_city}\n"
        f"  - Units Required: 500\n"
        f"  - Estimated Cost: ${cost:,.0f}\n"
        f"  - Priority: CRITICAL / Emergency Procurement\n\n"
        f"Kindly acknowledge receipt of this request and confirm your earliest available shipment window "
        f"at your earliest convenience. Our procurement team is standing by.\n\n"
        f"Best regards,\n"
        f"Autonomous Supply Chain Agent\n"
        f"SupplAI — Enterprise Logistics Intelligence Platform"
    )
    return {
        "email_draft": fallback_email,
        "erp_json_payload": json.dumps({
            "transaction_type": "PURCHASE_ORDER_UPDATE",
            "po_number": "PO-99482-EMG",
            "supplier": f"Generic Supplier {source_city}",
            "origin": source_city,
            "destination": dest_city,
            "units": 500,
            "cost_estimate_usd": cost,
            "status": "Awaiting Vendor Confirmation"
        }, indent=2)
    }

# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Mock inputs for quick local test
    disruption_info = {
        "event_text":     "Factory shutdown in China affecting electronics supply chain",
        "severity":       "high",
        "category":       "industrial_accident",
        "affected_nodes": ["City_1", "City_2", "City_12"],
    }

    risk_df = pd.DataFrame([
        {"city_name": "Shenzhen", "country": "China", "product_category": "Electronics",
         "risk_score": 0.91, "risk_level": "🔴 Critical"},
        {"city_name": "Shanghai", "country": "China", "product_category": "Semiconductors",
         "risk_score": 0.83, "risk_level": "🔴 Critical"},
    ])

    reroutes = [{"status": "✅ Alternate Found"}]

    brief = generate_brief(disruption_info, risk_df, reroutes)
    for k, v in brief.items():
        print(f"\n=== {k.upper()} ===")
        if isinstance(v, list):
            for item in v:
                print(f"  • {item}")
        else:
            print(f"  {v}")
