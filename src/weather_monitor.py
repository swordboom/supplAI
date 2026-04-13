"""
weather_monitor.py
-------------------
Monitors real-time weather and earthquake data near supply chain nodes.

Data sources:
  - OpenWeatherMap API  : current weather per lat/lon (API key required)
  - USGS GeoJSON        : significant earthquakes in the past 7 days (free)

Returns a list of disruption dicts compatible with parse_disruption() output,
so the existing pipeline (cascade → risk → reroute → brief) works unchanged.
"""

import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPLY_PATH  = PROJECT_ROOT / "data" / "supply_chain.csv"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
WIND_SEVERE_MS    = 15.0    # m/s  (~54 km/h)
WIND_EXTREME_MS   = 25.0    # m/s  (~90 km/h)
PRECIP_HEAVY_MM   = 10.0    # mm/3h
PRECIP_EXTREME_MM = 30.0    # mm/3h

# OWM condition IDs that flag severe weather
# https://openweathermap.org/weather-conditions
SEVERE_OWM_IDS = {
    # Thunderstorm
    200, 201, 202, 210, 211, 212, 221, 230, 231, 232,
    # Heavy rain / extreme rain
    502, 503, 504, 511, 522, 531,
    # Heavy snow / sleet
    602, 611, 612, 613, 615, 616, 621, 622,
    # Atmosphere: tornado, squall
    781, 771,
}

EARTHQUAKE_MIN_MAG = 5.5
OPENWEATHER_KEY_ENV_VARS = ("OPENWEATHER_API_KEY", "OPENWEATHERMAP_API_KEY", "OWM_API_KEY")

# Open-Meteo weather codes that indicate significant disruption risk.
OPEN_METEO_SEVERE_CODES = {65, 75, 82, 86, 95, 96, 99}
OPEN_METEO_MODERATE_CODES = {63, 66, 67, 73, 81, 85}
OPEN_METEO_CODE_LABELS = {
    63: "moderate rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    73: "moderate snowfall",
    75: "heavy snowfall",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
}
OPEN_METEO_HEAVY_PRECIP_MM = 7.0
OPEN_METEO_EXTREME_PRECIP_MM = 15.0


# ---------------------------------------------------------------------------
def _load_cities(supply_path: Path = SUPPLY_PATH) -> pd.DataFrame:
    df = pd.read_csv(supply_path)
    df = df[(df["lat"] != 0) & (df["lon"] != 0)].copy()
    return df


# ---------------------------------------------------------------------------
def _resolve_openweather_key(explicit_key: Optional[str] = None) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    for env_name in OPENWEATHER_KEY_ENV_VARS:
        candidate = os.getenv(env_name, "").strip()
        if candidate and not candidate.lower().startswith("your_"):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# OpenWeatherMap: current weather for a single location
# ---------------------------------------------------------------------------
def _fetch_owm(lat: float, lon: float, api_key: str, timeout: int = 6) -> Optional[Dict]:
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
def _fetch_open_meteo(lat: float, lon: float, timeout: int = 6) -> Optional[Dict]:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=weather_code,wind_speed_10m,precipitation,rain,showers,snowfall"
        "&forecast_days=1&timezone=UTC"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# USGS: recent significant earthquakes
# ---------------------------------------------------------------------------
def _fetch_earthquakes(timeout: int = 8) -> List[Dict]:
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.geojson"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read()).get("features", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Check weather for supply nodes via OpenWeatherMap
# ---------------------------------------------------------------------------
def check_weather_events(
    cities_df:    pd.DataFrame,
    api_key:      str,
    sample_every: int = 4,   # check every Nth city to stay within free-tier limits
) -> List[Dict[str, Any]]:
    events = []
    city_list = cities_df.reset_index(drop=True)

    for i, row in city_list.iterrows():
        if i % sample_every != 0:
            continue

        lat, lon = float(row["lat"]), float(row["lon"])
        data = _fetch_owm(lat, lon, api_key)
        if not data:
            continue

        weather_list = data.get("weather", [{}])
        owm_id       = weather_list[0].get("id",          800)
        description  = weather_list[0].get("description", "clear")
        wind_ms      = data.get("wind",  {}).get("speed", 0.0)
        rain_mm      = data.get("rain",  {}).get("3h",    0.0)
        snow_mm      = data.get("snow",  {}).get("3h",    0.0)
        precip_mm    = rain_mm + snow_mm

        is_severe_code  = owm_id in SEVERE_OWM_IDS
        is_severe_wind  = wind_ms  >= WIND_SEVERE_MS
        is_extreme_wind = wind_ms  >= WIND_EXTREME_MS
        is_heavy_rain   = precip_mm >= PRECIP_HEAVY_MM
        is_extreme_rain = precip_mm >= PRECIP_EXTREME_MM

        if not (is_severe_code or is_severe_wind or is_heavy_rain):
            continue

        city_name = row.get("city_name", "Unknown")
        country   = row.get("country",  "Unknown")
        product   = row.get("product_category", "General")
        city_id   = row.get("city_id",  str(i))

        if is_extreme_wind or is_extreme_rain or owm_id in {202, 212, 504, 781}:
            severity = "high"
        elif is_severe_wind or is_heavy_rain or is_severe_code:
            severity = "medium"
        else:
            severity = "low"

        conditions = [description]
        if is_severe_wind:
            conditions.append(f"{wind_ms:.0f} m/s winds")
        if is_heavy_rain:
            conditions.append(f"{precip_mm:.0f} mm precipitation")
        condition_str = ", ".join(conditions)

        events.append({
            "event_text":      f"{description.title()} in {city_name}, {country} disrupting {product} supply chain",
            "affected_nodes":  [city_id],
            "severity":        severity,
            "category":        "natural_disaster",
            "keywords_hit":    [country.lower(), product.lower()],
            "country_hit":     [country],
            "product_hit":     [product],
            "title":           f"Weather alert: {condition_str} at {city_name}",
            "source_headline": f"OpenWeatherMap: {condition_str} at {city_name} ({lat:.1f}°N, {lon:.1f}°E)",
            "source":          "OpenWeatherMap",
            "wind_ms":         round(wind_ms,   1),
            "precip_mm":       round(precip_mm, 1),
            "owm_id":          owm_id,
        })

    return events


# ---------------------------------------------------------------------------
def check_weather_events_open_meteo(
    cities_df: pd.DataFrame,
    sample_every: int = 4,
) -> List[Dict[str, Any]]:
    """
    No-key fallback weather monitor using Open-Meteo public API.
    """
    events = []
    city_list = cities_df.reset_index(drop=True)
    severe_wind_kmh = WIND_SEVERE_MS * 3.6
    extreme_wind_kmh = WIND_EXTREME_MS * 3.6

    for i, row in city_list.iterrows():
        if i % sample_every != 0:
            continue

        lat, lon = float(row["lat"]), float(row["lon"])
        data = _fetch_open_meteo(lat, lon)
        if not data:
            continue

        current = data.get("current", {})
        if not current:
            continue

        weather_code = int(current.get("weather_code", 0) or 0)
        wind_kmh = float(current.get("wind_speed_10m", 0.0) or 0.0)
        precip_mm = max(
            float(current.get("precipitation", 0.0) or 0.0),
            float(current.get("rain", 0.0) or 0.0)
            + float(current.get("showers", 0.0) or 0.0)
            + float(current.get("snowfall", 0.0) or 0.0),
        )

        is_severe_code = weather_code in OPEN_METEO_SEVERE_CODES
        is_moderate_code = weather_code in OPEN_METEO_MODERATE_CODES
        is_severe_wind = wind_kmh >= severe_wind_kmh
        is_extreme_wind = wind_kmh >= extreme_wind_kmh
        is_heavy_precip = precip_mm >= OPEN_METEO_HEAVY_PRECIP_MM
        is_extreme_precip = precip_mm >= OPEN_METEO_EXTREME_PRECIP_MM

        if not (is_severe_code or is_moderate_code or is_severe_wind or is_heavy_precip):
            continue

        city_name = row.get("city_name", "Unknown")
        country = row.get("country", "Unknown")
        product = row.get("product_category", "General")
        city_id = row.get("city_id", str(i))
        description = OPEN_METEO_CODE_LABELS.get(weather_code, "adverse weather")

        if is_severe_code or is_extreme_wind or is_extreme_precip:
            severity = "high"
        elif is_moderate_code or is_severe_wind or is_heavy_precip:
            severity = "medium"
        else:
            severity = "low"

        conditions = [description]
        if is_severe_wind:
            conditions.append(f"{wind_kmh:.0f} km/h winds")
        if is_heavy_precip:
            conditions.append(f"{precip_mm:.1f} mm precipitation")
        condition_str = ", ".join(conditions)

        events.append({
            "event_text": f"{description.title()} in {city_name}, {country} disrupting {product} supply chain",
            "affected_nodes": [city_id],
            "severity": severity,
            "category": "natural_disaster",
            "keywords_hit": [country.lower(), product.lower()],
            "country_hit": [country],
            "product_hit": [product],
            "title": f"Weather alert: {condition_str} at {city_name}",
            "source_headline": f"Open-Meteo: {condition_str} at {city_name} ({lat:.1f}°N, {lon:.1f}°E)",
            "source": "Open-Meteo",
            "wind_kmh": round(wind_kmh, 1),
            "precip_mm": round(precip_mm, 1),
            "weather_code": weather_code,
        })

    return events


# ---------------------------------------------------------------------------
# Check USGS earthquake feed, map quakes to nearby supply nodes
# ---------------------------------------------------------------------------
def check_earthquake_events(
    cities_df: pd.DataFrame,
    radius_km: float = 800,
) -> List[Dict[str, Any]]:
    quakes = _fetch_earthquakes()
    if not quakes:
        return []

    events = []
    seen_ids = set()

    for quake in quakes:
        props  = quake.get("properties", {})
        geo    = quake.get("geometry",   {})
        mag    = props.get("mag", 0) or 0
        place  = props.get("place", "Unknown location")
        qid    = quake.get("id", "")

        if mag < EARTHQUAKE_MIN_MAG or qid in seen_ids:
            continue
        seen_ids.add(qid)

        coords = geo.get("coordinates", [])
        if len(coords) < 2:
            continue
        q_lon, q_lat = float(coords[0]), float(coords[1])

        nearby_nodes     = []
        nearby_countries = []
        for _, row in cities_df.iterrows():
            dist = _haversine_km(q_lat, q_lon, float(row["lat"]), float(row["lon"]))
            if dist <= radius_km:
                nearby_nodes.append(row.get("city_id", ""))
                nearby_countries.append(row.get("country", ""))

        if not nearby_nodes:
            continue

        severity = "high" if mag >= 7.0 else ("medium" if mag >= 6.0 else "low")
        countries = list(dict.fromkeys(c for c in nearby_countries if c))[:3]
        country_str = ", ".join(countries) if countries else "affected region"

        events.append({
            "event_text":      f"Magnitude {mag:.1f} earthquake near {place} disrupting supply chains in {country_str}",
            "affected_nodes":  nearby_nodes,
            "severity":        severity,
            "category":        "natural_disaster",
            "keywords_hit":    ["earthquake"] + [c.lower() for c in countries],
            "country_hit":     countries,
            "product_hit":     [],
            "title":           f"M{mag:.1f} Earthquake — {place}",
            "source_headline": f"USGS: M{mag:.1f} earthquake near {place}",
            "source":          "USGS",
            "magnitude":       mag,
            "epicenter":       place,
        })

    return events


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------
def get_weather_disruptions(
    api_key:     str = None,
    supply_path: Path = SUPPLY_PATH,
) -> List[Dict[str, Any]]:
    """
    Full pipeline: load cities → check OWM weather → check USGS earthquakes.
    Returns list of disruption dicts sorted by severity.
    """
    effective_key = _resolve_openweather_key(api_key)

    print("  [weather_monitor] Loading city data …")
    cities_df = _load_cities(supply_path)

    quake_events   = []
    weather_events = []

    print("  [weather_monitor] Fetching USGS earthquake data …")
    quake_events = check_earthquake_events(cities_df)
    print(f"  [weather_monitor] {len(quake_events)} earthquake event(s) detected")

    if effective_key:
        print("  [weather_monitor] Fetching OpenWeatherMap conditions …")
        weather_events = check_weather_events(cities_df, effective_key)
        print(f"  [weather_monitor] {len(weather_events)} weather alert(s) detected")
    else:
        print("  [weather_monitor] No OpenWeather API key — using Open-Meteo fallback")
        weather_events = check_weather_events_open_meteo(cities_df)
        print(f"  [weather_monitor] {len(weather_events)} weather alert(s) detected (Open-Meteo)")

    all_events = quake_events + weather_events
    priority   = {"high": 0, "medium": 1, "low": 2}
    all_events.sort(key=lambda e: priority.get(e["severity"], 1))
    return all_events


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    events = get_weather_disruptions()
    if not events:
        print("No weather/earthquake disruptions detected.")
    else:
        for i, e in enumerate(events, 1):
            print(f"[{i}] [{e.get('source','?')}] {e['title']}")
            print(f"     Severity : {e['severity']}")
            print(f"     Nodes    : {e['affected_nodes'][:5]}")
            print(f"     Text     : {e['event_text']}\n")
