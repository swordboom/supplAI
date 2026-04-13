"""
news_fetcher.py
----------------
Fetches latest geopolitical / supply chain news from public RSS feeds,
then uses Gemini to extract structured disruption events from headlines.

Returns a list of disruption dicts compatible with parse_disruption() output,
so the existing cascade / risk / reroute pipeline works unchanged.
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import feedparser

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# RSS feeds — public, no auth required
# ---------------------------------------------------------------------------
RSS_FEEDS = [
    {"name": "Reuters World",   "url": "https://feeds.reuters.com/reuters/worldNews"},
    {"name": "BBC World",       "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "Al Jazeera",      "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"name": "AP Top News",     "url": "https://rsshub.app/apnews/topics/apf-topnews"},
    {"name": "Guardian World",  "url": "https://www.theguardian.com/world/rss"},
]

# Keywords to filter headlines before sending to Gemini (saves tokens)
STRONG_KEYWORDS = {
    "port", "shipping", "cargo", "container", "freight", "logistics", "supply chain",
    "factory", "manufacturing", "shutdown", "plant", "closure", "outage",
    "export", "import", "tariff", "sanction", "embargo", "trade", "blockade",
    "pipeline", "canal", "strait", "railway",
    "earthquake", "tsunami", "typhoon", "hurricane", "flood", "wildfire", "storm",
    "strike", "protest", "riot"
}

WEAK_KEYWORDS = {
    "semiconductor", "chip", "electronics", "oil", "gas", "steel", "pharma",
    "automotive", "textile", "energy", "fuel", "mineral", "rare earth",
    "china", "russia", "ukraine", "taiwan", "iran", "north korea", "middle east", "red sea", "suez",
    "india", "pakistan", "israel", "gaza",
    "war", "conflict", "military", "missile", "nuclear", "attack", "threat", "border",
    "explosion", "accident", "fire", "collapse"
}

NEGATIVE_KEYWORDS = {
    "soldier", "officer", "person", "biography", "memoir", "troop",
    "court", "trial", "lawsuit", "allegation", "case", "war crime", "prosecutor",
    "murder", "prison", "sentenced", "guilty", "suspect", "arrest"
}

GEMINI_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")

# ---------------------------------------------------------------------------
# Gemini prompt
# ---------------------------------------------------------------------------
_EXTRACT_PROMPT = """Extract supply chain disruption events from these headlines. Return ONLY a JSON array (no markdown).

Headlines:
{headlines}

Each item: {{"title":"...","affected_countries":["..."],"affected_sectors":["..."],"severity":"high/medium/low","category":"geopolitical/natural_disaster/labor/industrial_accident/logistics/other","event_text":"one sentence like Factory shutdown in China affecting electronics","source_headline":"..."}}

Rules: only physical supply chain impact, max 4 events, return [] if none. JSON array only."""



def is_supply_chain_relevant(title: str, summary: str) -> bool:
    """
    Returns True ONLY if the headline has DIRECT or STRONG INDIRECT supply chain impact.
    Uses positive/negative signals and heuristics to score relevance.
    """
    title_lower = title.lower()
    combined = (title + " " + summary).lower()

    # NLP Heuristics - Reject
    if title_lower.startswith(("who ", "what ", "he's ", "she's ", "he is ", "she is ", "he’s ", "she’s ")):
        return False

    if "opinion:" in title_lower or "editorial:" in title_lower:
        return False

    score = 0

    for kw in STRONG_KEYWORDS:
        if kw in combined:
            score += 2

    for kw in WEAK_KEYWORDS:
        if kw in combined:
            score += 1

    for kw in NEGATIVE_KEYWORDS:
        if kw in combined:
            score -= 3

    return score >= 2

# ---------------------------------------------------------------------------
# Fetch raw headlines from RSS
# ---------------------------------------------------------------------------
def fetch_headlines(max_per_feed: int = 15) -> List[Dict[str, str]]:
    """
    Pull headlines from all RSS feeds, filter by supply-chain keywords.

    Returns list of dicts: {title, summary, source}
    """
    headlines = []

    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:max_per_feed]:
                title   = getattr(entry, "title",   "") or ""
                summary = getattr(entry, "summary", "") or ""

                if is_supply_chain_relevant(title, summary):
                    headlines.append({
                        "title":   title.strip(),
                        "summary": summary.strip()[:200],
                        "source":  feed_info["name"],
                    })
                else:
                    if title.strip():
                        print(f"  [FILTERED OUT] {title.strip()}")
        except Exception as e:
            print(f"  [news_fetcher] Failed to fetch {feed_info['name']}: {e}")

    # Deduplicate by title similarity (simple: exact title match)
    seen_titles = set()
    unique = []
    for h in headlines:
        t = h["title"].lower()
        if t not in seen_titles:
            seen_titles.add(t)
            unique.append(h)

    print(f"  [news_fetcher] Fetched {len(unique)} unique relevant headlines")
    return unique


# ---------------------------------------------------------------------------
# Gemini API call — supports both google-genai (new) and google-generativeai (old)
# ---------------------------------------------------------------------------
def _gemini_generate(api_key: str, prompt: str, temperature: float = 0.1, max_tokens: int = 8192) -> str:
    """
    Call Gemini, trying the new google-genai SDK first, then falling back to
    the legacy google-generativeai SDK so the app works in any environment.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()
    except ImportError:
        import google.generativeai as _genai
        _genai.configure(api_key=api_key)
        model = _genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# API-key resolution and offline extraction fallback
# ---------------------------------------------------------------------------
def _resolve_gemini_key(explicit_key: Optional[str] = None) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    for env_name in GEMINI_KEY_ENV_VARS:
        candidate = os.getenv(env_name, "").strip()
        if candidate and not candidate.lower().startswith("your_"):
            return candidate
    return ""


def _extract_disruptions_offline(
    headlines: List[Dict[str, str]],
    max_events: int = 4,
) -> List[Dict[str, Any]]:
    """
    Offline fallback: parse headlines with the keyword-based disruption parser.
    Returns an event schema compatible with Gemini extraction output.
    """
    if not headlines:
        return []

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from disruption_input import parse_disruption

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    parsed_events: List[Dict[str, Any]] = []
    seen_signatures = set()

    for h in headlines[:30]:
        title = (h.get("title") or "").strip()
        if not title:
            continue

        parsed = parse_disruption(title)
        countries = parsed.get("country_hit", [])
        sectors = parsed.get("product_hit", [])
        keywords = parsed.get("keywords_hit", [])

        # Skip weak generic matches when we have better candidates.
        if not countries and not sectors and keywords == ["general"]:
            continue

        signature = (
            parsed.get("severity", "medium"),
            parsed.get("category", "other"),
            tuple(sorted(countries)),
            tuple(sorted(sectors)),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        parsed_events.append({
            "title": title,
            "affected_countries": countries,
            "affected_sectors": sectors,
            "severity": parsed.get("severity", "medium"),
            "category": parsed.get("category", "other"),
            "event_text": parsed.get("event_text", title),
            "source_headline": f"[{h.get('source', 'Unknown')}] {title}",
        })

    if not parsed_events:
        # Last resort: keep the first few headlines as medium severity events.
        for h in headlines[:max_events]:
            title = (h.get("title") or "").strip()
            if not title:
                continue
            parsed_events.append({
                "title": title,
                "affected_countries": [],
                "affected_sectors": [],
                "severity": "medium",
                "category": "other",
                "event_text": title,
                "source_headline": f"[{h.get('source', 'Unknown')}] {title}",
            })

    parsed_events.sort(key=lambda e: severity_rank.get(e.get("severity", "medium"), 1))
    return parsed_events[:max_events]


# ---------------------------------------------------------------------------
# Extract disruption events using Gemini
# ---------------------------------------------------------------------------
def extract_disruptions_with_gemini(
    headlines: List[Dict[str, str]],
    api_key:   str = None,
) -> List[Dict[str, Any]]:
    """
    Send filtered headlines to Gemini and extract structured disruption events.

    Returns list of disruption dicts compatible with parse_disruption() output.
    """
    effective_key = _resolve_gemini_key(api_key)
    if not effective_key:
        print("  [news_fetcher] No Gemini API key — using keyword extraction fallback")
        fallback_events = _extract_disruptions_offline(headlines)
        print(f"  [news_fetcher] Fallback extracted {len(fallback_events)} disruption event(s)")
        return fallback_events

    if not headlines:
        return []

    # Format headlines for the prompt
    headline_block = "\n".join(
        f"{i+1}. [{h['source']}] {h['title']}"
        + (f" — {h['summary']}" if h.get("summary") else "")
        for i, h in enumerate(headlines[:30])   # cap at 30 to avoid token overflow
    )

    try:
        prompt = _EXTRACT_PROMPT.format(headlines=headline_block)
        raw = _gemini_generate(effective_key, prompt, temperature=0.1, max_tokens=8192)
        # Extract JSON array robustly — find first [ ... ] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in Gemini response")

        gemini_events = json.loads(match.group(0))
        if not isinstance(gemini_events, list):
            return []

        print(f"  [news_fetcher] Gemini extracted {len(gemini_events)} disruption events")
        return gemini_events

    except Exception as e:
        print(f"  [news_fetcher] Gemini extraction failed: {e} — using keyword fallback")
        fallback_events = _extract_disruptions_offline(headlines)
        print(f"  [news_fetcher] Fallback extracted {len(fallback_events)} disruption event(s)")
        return fallback_events


# ---------------------------------------------------------------------------
# Map Gemini output → city node IDs (using existing keyword maps)
# ---------------------------------------------------------------------------
def _map_to_nodes(countries: List[str], sectors: List[str]) -> List[str]:
    """Map country/sector names to City_XX IDs using disruption_input maps."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from disruption_input import COUNTRY_CITY_MAP, PRODUCT_CITY_MAP

    nodes = []
    for country in countries:
        key = country.lower().strip()
        if key in COUNTRY_CITY_MAP:
            nodes.extend(COUNTRY_CITY_MAP[key])
        # Try partial match
        else:
            for map_key, cities in COUNTRY_CITY_MAP.items():
                if map_key in key or key in map_key:
                    nodes.extend(cities)
                    break

    for sector in sectors:
        key = sector.lower().strip()
        if key in PRODUCT_CITY_MAP:
            nodes.extend(PRODUCT_CITY_MAP[key])
        else:
            for map_key, cities in PRODUCT_CITY_MAP.items():
                if map_key in key or key in map_key:
                    nodes.extend(cities)
                    break

    # Deduplicate
    seen = set()
    return [n for n in nodes if not (n in seen or seen.add(n))]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def get_live_disruptions(api_key: str = None) -> List[Dict[str, Any]]:
    """
    Full pipeline: fetch news → Gemini extract → map to nodes.

    Returns list of disruption dicts, each with:
        event_text, affected_nodes, severity, category,
        keywords_hit, country_hit, product_hit,
        title, source_headline
    """
    headlines = fetch_headlines()
    if not headlines:
        return []

    gemini_events = extract_disruptions_with_gemini(headlines, api_key=api_key)
    if not gemini_events:
        return []

    disruptions = []
    for evt in gemini_events:
        countries = evt.get("affected_countries", [])
        sectors   = evt.get("affected_sectors",   [])
        nodes     = _map_to_nodes(countries, sectors)

        if not nodes:
            nodes = ["City_1", "City_2", "City_13", "City_24"]   # fallback

        disruptions.append({
            "event_text":     evt.get("event_text", evt.get("title", "Unknown event")),
            "affected_nodes": nodes,
            "severity":       evt.get("severity",  "medium"),
            "category":       evt.get("category",  "other"),
            "keywords_hit":   countries + sectors,
            "country_hit":    countries,
            "product_hit":    sectors,
            "title":          evt.get("title", ""),
            "source_headline": evt.get("source_headline", ""),
        })

    return disruptions


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Fetching live news disruptions …\n")
    events = get_live_disruptions()
    if not events:
        print("No disruption events found.")
    else:
        for i, e in enumerate(events, 1):
            print(f"[{i}] {e['title']}")
            print(f"     Severity : {e['severity']}")
            print(f"     Category : {e['category']}")
            print(f"     Countries: {e['country_hit']}")
            print(f"     Sectors  : {e['product_hit']}")
            print(f"     Nodes    : {e['affected_nodes'][:5]} …")
            print(f"     Text     : {e['event_text']}\n")
