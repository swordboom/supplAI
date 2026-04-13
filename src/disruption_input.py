"""
disruption_input.py
--------------------
Parses a free-text disruption event into a structured object containing:
  - event_text     : original text
  - affected_nodes : list of City_XX IDs likely impacted
  - severity       : "high" / "medium" / "low"
  - category       : type of disruption (natural_disaster, strike, etc.)
  - keywords_hit   : which keywords / countries triggered the match
  - reasoning      : LLM explanation of WHY those nodes were chosen (LLM path only)
  - llm_source     : "gemini" | "keyword-matching"

Strategy
--------
1. Try Gemini LLM first — it understands geopolitical context, implied effects,
   unusual phrasing (e.g. "Hormuz tensions" → oil routes through Middle East).
2. Fall back to fast keyword matching when the API key is missing or the call fails.
   The fallback is deterministic and works fully offline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

GEMINI_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")


# ---------------------------------------------------------------------------
# Country / city mapping  (City_1–City_70 original + City_71–City_159 new)
# ---------------------------------------------------------------------------
COUNTRY_CITY_MAP: Dict[str, List[str]] = {
    # ── China ─────────────────────────────────────────────────────────────────
    "china":          ["City_61","City_1","City_2","City_3","City_4","City_5","City_6",
                       "City_7","City_8","City_9","City_10","City_11","City_12",
                       "City_71","City_72","City_73","City_74","City_75","City_76"],
    "chinese":        ["City_61","City_1","City_2","City_3","City_4","City_5","City_6",
                       "City_7","City_8","City_9","City_10","City_11","City_12",
                       "City_71","City_72","City_73","City_74","City_75","City_76"],
    "shenzhen":       ["City_1"],
    "shanghai":       ["City_2"],
    "beijing":        ["City_3"],
    "guangzhou":      ["City_4"],
    "chengdu":        ["City_5"],
    "wuhan":          ["City_6"],
    "xian":           ["City_7"],
    "tianjin":        ["City_8"],
    "chongqing":      ["City_9"],
    "nanjing":        ["City_10"],
    "hangzhou":       ["City_11"],
    "dongguan":       ["City_12"],
    "suzhou":         ["City_71"],
    "qingdao":        ["City_72"],
    "zhengzhou":      ["City_73"],
    "hefei":          ["City_74"],
    "ningbo":         ["City_75"],
    "wuxi":           ["City_76"],
    "hanoi":          ["City_61"],
    "vietnam":        ["City_55","City_61","City_134"],
    "vietnamese":     ["City_55","City_61","City_134"],
    "da nang":        ["City_134"],

    # ── Taiwan ────────────────────────────────────────────────────────────────
    "taiwan":         ["City_48","City_51","City_77","City_78"],
    "taiwanese":      ["City_48","City_51","City_77","City_78"],
    "taipei":         ["City_48"],
    "kaohsiung":      ["City_51"],
    "hsinchu":        ["City_77"],
    "taichung":       ["City_78"],
    "tsmc":           ["City_77","City_48"],

    # ── Japan ─────────────────────────────────────────────────────────────────
    "japan":          ["City_43","City_44","City_45","City_50","City_52","City_79","City_80"],
    "japanese":       ["City_43","City_44","City_45","City_50","City_52","City_79","City_80"],
    "tokyo":          ["City_43"],
    "osaka":          ["City_44"],
    "nagoya":         ["City_45"],
    "yokohama":       ["City_50"],
    "hiroshima":      ["City_52"],
    "kobe":           ["City_79"],
    "fukuoka":        ["City_80"],

    # ── South Korea ───────────────────────────────────────────────────────────
    "korea":          ["City_46","City_47","City_49","City_81","City_82"],
    "korean":         ["City_46","City_47","City_49","City_81","City_82"],
    "south korea":    ["City_46","City_47","City_49","City_81","City_82"],
    "seoul":          ["City_46"],
    "busan":          ["City_47"],
    "incheon":        ["City_49"],
    "ulsan":          ["City_81"],
    "suwon":          ["City_82"],
    "samsung":        ["City_82","City_46"],
    "hyundai":        ["City_81"],

    # ── India ─────────────────────────────────────────────────────────────────
    "india":          ["City_13","City_14","City_15","City_16","City_17","City_18",
                       "City_19","City_20","City_21","City_22","City_128","City_129"],
    "indian":         ["City_13","City_14","City_15","City_16","City_17","City_18",
                       "City_19","City_20","City_21","City_22","City_128","City_129"],
    "mumbai":         ["City_13"],
    "bangalore":      ["City_14"],
    "chennai":        ["City_15"],
    "pune":           ["City_16"],
    "delhi":          ["City_17"],
    "hyderabad":      ["City_18"],
    "kolkata":        ["City_19"],
    "surat":          ["City_20"],
    "ahmedabad":      ["City_21"],
    "coimbatore":     ["City_22"],
    "kochi":          ["City_128"],
    "visakhapatnam":  ["City_129"],

    # ── Southeast Asia ────────────────────────────────────────────────────────
    "southeast asia": ["City_53","City_54","City_55","City_56","City_57","City_58",
                       "City_61","City_130","City_131","City_132","City_133","City_134","City_135"],
    "singapore":      ["City_53"],
    "thailand":       ["City_54"],
    "bangkok":        ["City_54"],
    "malaysia":       ["City_57","City_130"],
    "kuala lumpur":   ["City_57"],
    "penang":         ["City_130"],
    "indonesia":      ["City_56","City_132","City_135"],
    "jakarta":        ["City_56"],
    "surabaya":       ["City_132"],
    "batam":          ["City_135"],
    "philippines":    ["City_58","City_131"],
    "manila":         ["City_58"],
    "cebu":           ["City_131"],
    "myanmar":        ["City_133"],
    "yangon":         ["City_133"],
    "sri lanka":      ["City_62"],
    "colombo":        ["City_62"],

    # ── USA ───────────────────────────────────────────────────────────────────
    "usa":            ["City_23","City_24","City_25","City_26","City_27","City_28",
                       "City_29","City_30","City_31","City_32",
                       "City_102","City_103","City_104","City_105","City_106"],
    "america":        ["City_23","City_24","City_25","City_26","City_27","City_28",
                       "City_29","City_30","City_31","City_32",
                       "City_102","City_103","City_104","City_105","City_106"],
    "american":       ["City_23","City_24","City_25","City_26","City_27","City_28",
                       "City_29","City_30","City_31","City_32",
                       "City_102","City_103","City_104","City_105","City_106"],
    "united states":  ["City_23","City_24","City_25","City_26","City_27","City_28",
                       "City_29","City_30","City_31","City_32",
                       "City_102","City_103","City_104","City_105","City_106"],
    "new york":       ["City_23"],
    "los angeles":    ["City_24"],
    "houston":        ["City_25"],
    "chicago":        ["City_26"],
    "san jose":       ["City_27"],
    "seattle":        ["City_28"],
    "atlanta":        ["City_29"],
    "detroit":        ["City_30"],
    "dallas":         ["City_31"],
    "phoenix":        ["City_32"],
    "boston":         ["City_102"],
    "austin":         ["City_103"],
    "san diego":      ["City_104"],
    "minneapolis":    ["City_105"],
    "portland":       ["City_106"],

    # ── Canada ────────────────────────────────────────────────────────────────
    "canada":         ["City_107","City_108","City_109"],
    "canadian":       ["City_107","City_108","City_109"],
    "toronto":        ["City_107"],
    "vancouver":      ["City_108"],
    "montreal":       ["City_109"],

    # ── Mexico ────────────────────────────────────────────────────────────────
    "mexico":         ["City_110","City_111","City_112","City_113"],
    "mexican":        ["City_110","City_111","City_112","City_113"],
    "mexico city":    ["City_110"],
    "monterrey":      ["City_111"],
    "guadalajara":    ["City_112"],
    "tijuana":        ["City_113"],

    # ── Latin America ─────────────────────────────────────────────────────────
    "latin america":  ["City_114","City_115","City_116","City_117","City_118"],
    "brazil":         ["City_114","City_115"],
    "brazilian":      ["City_114","City_115"],
    "sao paulo":      ["City_114"],
    "são paulo":      ["City_114"],
    "rio de janeiro": ["City_115"],
    "argentina":      ["City_116"],
    "buenos aires":   ["City_116"],
    "chile":          ["City_117"],
    "santiago":       ["City_117"],
    "colombia":       ["City_118"],
    "bogota":         ["City_118"],

    # ── Germany ───────────────────────────────────────────────────────────────
    "germany":        ["City_33","City_34","City_35","City_36","City_37"],
    "german":         ["City_33","City_34","City_35","City_36","City_37"],
    "berlin":         ["City_33"],
    "munich":         ["City_34"],
    "hamburg":        ["City_35"],
    "frankfurt":      ["City_36"],
    "stuttgart":      ["City_37"],

    # ── Europe (broad) ────────────────────────────────────────────────────────
    "europe":         ["City_33","City_34","City_35","City_36","City_37","City_38",
                       "City_39","City_40","City_41","City_42","City_83","City_84",
                       "City_85","City_86","City_87","City_88","City_89","City_90",
                       "City_91","City_92","City_93","City_94","City_95","City_96",
                       "City_97","City_98","City_99","City_100","City_101","City_143",
                       "City_144","City_150","City_151","City_152","City_153","City_154",
                       "City_155","City_156","City_157","City_158","City_159"],
    "european":       ["City_33","City_34","City_35","City_36","City_37","City_38",
                       "City_39","City_40","City_41","City_42","City_83","City_84",
                       "City_85","City_86","City_87","City_88","City_89","City_90",
                       "City_91","City_92","City_93","City_94","City_95","City_96",
                       "City_97","City_98","City_99","City_100","City_101","City_143",
                       "City_144","City_150","City_151","City_152","City_153","City_154",
                       "City_155","City_156","City_157","City_158","City_159"],

    # ── Netherlands ───────────────────────────────────────────────────────────
    "netherlands":    ["City_38","City_85"],
    "rotterdam":      ["City_38"],
    "amsterdam":      ["City_85"],

    # ── France ────────────────────────────────────────────────────────────────
    "france":         ["City_39","City_40","City_150"],
    "french":         ["City_39","City_40","City_150"],
    "paris":          ["City_39"],
    "lyon":           ["City_40"],
    "marseille":      ["City_150"],

    # ── Italy ─────────────────────────────────────────────────────────────────
    "italy":          ["City_41","City_151","City_152"],
    "italian":        ["City_41","City_151","City_152"],
    "milan":          ["City_41"],
    "turin":          ["City_151"],
    "genoa":          ["City_152"],

    # ── Spain ─────────────────────────────────────────────────────────────────
    "spain":          ["City_42","City_83","City_159"],
    "spanish":        ["City_42","City_83","City_159"],
    "madrid":         ["City_42"],
    "barcelona":      ["City_83"],
    "valencia":       ["City_159"],

    # ── Belgium ───────────────────────────────────────────────────────────────
    "belgium":        ["City_84"],
    "antwerp":        ["City_84"],

    # ── Sweden ────────────────────────────────────────────────────────────────
    "sweden":         ["City_86","City_87"],
    "swedish":        ["City_86","City_87"],
    "stockholm":      ["City_86"],
    "gothenburg":     ["City_87"],

    # ── Denmark ───────────────────────────────────────────────────────────────
    "denmark":        ["City_88"],
    "danish":         ["City_88"],
    "copenhagen":     ["City_88"],

    # ── Norway ────────────────────────────────────────────────────────────────
    "norway":         ["City_89"],
    "norwegian":      ["City_89"],
    "oslo":           ["City_89"],

    # ── Switzerland ───────────────────────────────────────────────────────────
    "switzerland":    ["City_90"],
    "swiss":          ["City_90"],
    "zurich":         ["City_90"],

    # ── Poland ────────────────────────────────────────────────────────────────
    "poland":         ["City_91","City_92","City_158"],
    "polish":         ["City_91","City_92","City_158"],
    "warsaw":         ["City_91"],
    "gdansk":         ["City_92"],
    "gdynia":         ["City_158"],

    # ── Czech Republic ────────────────────────────────────────────────────────
    "czech":          ["City_93"],
    "czech republic": ["City_93"],
    "prague":         ["City_93"],

    # ── Hungary ───────────────────────────────────────────────────────────────
    "hungary":        ["City_94"],
    "hungarian":      ["City_94"],
    "budapest":       ["City_94"],

    # ── Romania ───────────────────────────────────────────────────────────────
    "romania":        ["City_95"],
    "romanian":       ["City_95"],
    "bucharest":      ["City_95"],

    # ── Austria ───────────────────────────────────────────────────────────────
    "austria":        ["City_96"],
    "austrian":       ["City_96"],
    "vienna":         ["City_96"],

    # ── Portugal ──────────────────────────────────────────────────────────────
    "portugal":       ["City_97"],
    "portuguese":     ["City_97"],
    "lisbon":         ["City_97"],

    # ── Greece ────────────────────────────────────────────────────────────────
    "greece":         ["City_98"],
    "greek":          ["City_98"],
    "athens":         ["City_98"],

    # ── United Kingdom ────────────────────────────────────────────────────────
    "uk":             ["City_99","City_100","City_101"],
    "britain":        ["City_99","City_100","City_101"],
    "british":        ["City_99","City_100","City_101"],
    "england":        ["City_99","City_100","City_101"],
    "london":         ["City_99"],
    "manchester":     ["City_100"],
    "birmingham":     ["City_101"],

    # ── Finland ───────────────────────────────────────────────────────────────
    "finland":        ["City_153"],
    "finnish":        ["City_153"],
    "helsinki":       ["City_153"],

    # ── Slovakia ──────────────────────────────────────────────────────────────
    "slovakia":       ["City_154"],
    "bratislava":     ["City_154"],

    # ── Bulgaria ──────────────────────────────────────────────────────────────
    "bulgaria":       ["City_155"],
    "bulgarian":      ["City_155"],
    "sofia":          ["City_155"],

    # ── Serbia ────────────────────────────────────────────────────────────────
    "serbia":         ["City_156"],
    "serbian":        ["City_156"],
    "belgrade":       ["City_156"],

    # ── Latvia ────────────────────────────────────────────────────────────────
    "latvia":         ["City_157"],
    "latvian":        ["City_157"],
    "riga":           ["City_157"],

    # ── Russia ────────────────────────────────────────────────────────────────
    "russia":         ["City_143","City_144"],
    "russian":        ["City_143","City_144"],
    "moscow":         ["City_143"],
    "st petersburg":  ["City_144"],

    # ── Kazakhstan ────────────────────────────────────────────────────────────
    "kazakhstan":     ["City_145"],
    "almaty":         ["City_145"],

    # ── Azerbaijan ────────────────────────────────────────────────────────────
    "azerbaijan":     ["City_146"],
    "baku":           ["City_146"],

    # ── Middle East ───────────────────────────────────────────────────────────
    "middle east":    ["City_63","City_64","City_68","City_69","City_119","City_120",
                       "City_121","City_122","City_123","City_124","City_125"],
    "uae":            ["City_63","City_119"],
    "dubai":          ["City_63"],
    "abu dhabi":      ["City_119"],
    "saudi":          ["City_64","City_122"],
    "saudi arabia":   ["City_64","City_122"],
    "riyadh":         ["City_64"],
    "jeddah":         ["City_122"],
    "qatar":          ["City_120"],
    "doha":           ["City_120"],
    "kuwait":         ["City_121"],
    "kuwait city":    ["City_121"],
    "israel":         ["City_123"],
    "tel aviv":       ["City_123"],
    "oman":           ["City_124"],
    "muscat":         ["City_124"],
    "turkey":         ["City_68","City_125"],
    "turkish":        ["City_68","City_125"],
    "istanbul":       ["City_68"],
    "ankara":         ["City_125"],
    "iran":           ["City_69"],
    "tehran":         ["City_69"],
    "hormuz":         ["City_63","City_64","City_119","City_120","City_124"],
    "strait of hormuz":["City_63","City_64","City_119","City_120","City_124"],
    "red sea":        ["City_63","City_66","City_122","City_137"],
    "suez":           ["City_66","City_137"],

    # ── South Asia extra ──────────────────────────────────────────────────────
    "pakistan":       ["City_126"],
    "karachi":        ["City_126"],
    "bangladesh":     ["City_127"],
    "dhaka":          ["City_127"],

    # ── Africa ────────────────────────────────────────────────────────────────
    "africa":         ["City_65","City_66","City_67","City_70","City_136","City_137",
                       "City_138","City_139","City_140","City_141","City_142"],
    "nigeria":        ["City_65"],
    "lagos":          ["City_65"],
    "egypt":          ["City_66","City_137"],
    "cairo":          ["City_66"],
    "alexandria":     ["City_137"],
    "south africa":   ["City_67","City_138"],
    "johannesburg":   ["City_67"],
    "durban":         ["City_138"],
    "kenya":          ["City_70","City_142"],
    "nairobi":        ["City_70"],
    "mombasa":        ["City_142"],
    "ethiopia":       ["City_139"],
    "addis ababa":    ["City_139"],
    "ghana":          ["City_140"],
    "accra":          ["City_140"],
    "tanzania":       ["City_141"],
    "dar es salaam":  ["City_141"],
    "morocco":        ["City_136"],
    "casablanca":     ["City_136"],

    # ── Australia / Oceania ───────────────────────────────────────────────────
    "australia":      ["City_59","City_60","City_147","City_148"],
    "australian":     ["City_59","City_60","City_147","City_148"],
    "sydney":         ["City_59"],
    "melbourne":      ["City_60"],
    "brisbane":       ["City_147"],
    "perth":          ["City_148"],
    "new zealand":    ["City_149"],
    "auckland":       ["City_149"],
}

PRODUCT_CITY_MAP: Dict[str, List[str]] = {
    # ── Semiconductors ────────────────────────────────────────────────────────
    "semiconductor":  ["City_2","City_46","City_48","City_27","City_74","City_76",
                       "City_77","City_80","City_103","City_123","City_130"],
    "semiconductors": ["City_2","City_46","City_48","City_27","City_74","City_76",
                       "City_77","City_80","City_103","City_123","City_130"],
    "chip":           ["City_2","City_46","City_48","City_27","City_77","City_103"],
    "chips":          ["City_2","City_46","City_48","City_27","City_77","City_103"],
    "microchip":      ["City_2","City_46","City_48","City_77"],
    "wafer":          ["City_48","City_77","City_46","City_2"],

    # ── Electronics ───────────────────────────────────────────────────────────
    "electronics":    ["City_61","City_1","City_5","City_12","City_43","City_49",
                       "City_57","City_58","City_71","City_73","City_82","City_112",
                       "City_113","City_131","City_134","City_135"],
    "electronic":     ["City_61","City_1","City_5","City_12","City_43","City_49",
                       "City_57","City_58","City_71","City_73","City_82","City_112",
                       "City_113","City_131","City_134","City_135"],
    "factory":        ["City_61","City_1","City_4","City_12","City_71","City_73"],

    # ── Automotive ────────────────────────────────────────────────────────────
    "automotive":     ["City_3","City_9","City_15","City_33","City_37","City_42",
                       "City_45","City_54","City_81","City_83","City_87","City_91",
                       "City_93","City_94","City_95","City_101","City_107","City_110",
                       "City_114","City_125","City_144","City_151","City_154","City_155","City_156"],
    "automobile":     ["City_3","City_9","City_15","City_33","City_37","City_42",
                       "City_45","City_81","City_83","City_87","City_91","City_93",
                       "City_94","City_95","City_110","City_114","City_151"],
    "car":            ["City_30","City_33","City_37","City_45","City_81","City_87","City_107"],
    "vehicle":        ["City_30","City_33","City_37","City_45","City_81","City_87","City_107"],

    # ── Pharmaceuticals ───────────────────────────────────────────────────────
    "pharmaceutical": ["City_10","City_13","City_18","City_40","City_88","City_90",
                       "City_102","City_104"],
    "pharmaceuticals":["City_10","City_13","City_18","City_40","City_88","City_90",
                       "City_102","City_104"],
    "pharma":         ["City_10","City_13","City_18","City_40","City_88","City_90",
                       "City_102","City_104"],
    "medicine":       ["City_13","City_18","City_40","City_90","City_102"],
    "drug":           ["City_13","City_18","City_90"],
    "vaccine":        ["City_13","City_18","City_40","City_88","City_90","City_102"],

    # ── Textiles ──────────────────────────────────────────────────────────────
    "textile":        ["City_4","City_19","City_20","City_41","City_55","City_68",
                       "City_97","City_126","City_127","City_133","City_136","City_139","City_159"],
    "textiles":       ["City_4","City_19","City_20","City_41","City_55","City_68",
                       "City_97","City_126","City_127","City_133","City_136","City_139","City_159"],
    "clothing":       ["City_4","City_19","City_41","City_68","City_127","City_136"],
    "apparel":        ["City_4","City_19","City_41","City_127","City_136"],
    "garment":        ["City_4","City_19","City_55","City_127","City_133"],

    # ── Oil & Gas ─────────────────────────────────────────────────────────────
    "oil":            ["City_25","City_64","City_65","City_69","City_89","City_115",
                       "City_119","City_120","City_121","City_124","City_146"],
    "gas":            ["City_25","City_64","City_69","City_89","City_119","City_120",
                       "City_121","City_146"],
    "petroleum":      ["City_25","City_64","City_69","City_119","City_120"],
    "petrochemical":  ["City_69","City_146"],
    "crude":          ["City_25","City_64","City_115","City_119","City_120","City_121"],
    "lng":            ["City_64","City_119","City_120","City_121","City_89"],
    "energy":         ["City_25","City_64","City_89","City_119","City_120","City_121"],

    # ── Steel & Metals ────────────────────────────────────────────────────────
    "steel":          ["City_6","City_26","City_111","City_129"],
    "metal":          ["City_6","City_26","City_111","City_129"],
    "iron":           ["City_6","City_26","City_129"],

    # ── Chemicals ─────────────────────────────────────────────────────────────
    "chemical":       ["City_8","City_21","City_36","City_40"],
    "chemicals":      ["City_8","City_21","City_36","City_40"],

    # ── Aerospace ─────────────────────────────────────────────────────────────
    "aerospace":      ["City_7","City_28","City_109"],
    "aviation":       ["City_7","City_28","City_109"],
    "aircraft":       ["City_7","City_28","City_109"],

    # ── Minerals & Mining ─────────────────────────────────────────────────────
    "mineral":        ["City_59","City_67","City_117","City_140","City_147","City_148"],
    "minerals":       ["City_59","City_67","City_117","City_140","City_147","City_148"],
    "mining":         ["City_59","City_67","City_117","City_140","City_147","City_148"],
    "copper":         ["City_117","City_67","City_140"],
    "lithium":        ["City_117","City_116"],
    "rare earth":     ["City_59","City_67","City_148"],

    # ── Agriculture ───────────────────────────────────────────────────────────
    "agriculture":    ["City_60","City_70","City_116","City_149"],
    "food":           ["City_60","City_70","City_116","City_149"],
    "grain":          ["City_60","City_116","City_149"],

    # ── IT Hardware ───────────────────────────────────────────────────────────
    "it hardware":    ["City_14","City_27","City_86","City_153"],
    "server":         ["City_14","City_27","City_86","City_153"],
    "data center":    ["City_27","City_86"],

    # ── Luxury Goods ──────────────────────────────────────────────────────────
    "luxury":         ["City_39","City_41","City_96"],
    "luxury goods":   ["City_39","City_41","City_96"],

    # ── Ports & Logistics ─────────────────────────────────────────────────────
    "port":           ["City_35","City_38","City_47","City_51","City_53","City_63",
                       "City_72","City_75","City_79","City_84","City_92","City_98",
                       "City_108","City_122","City_128","City_137","City_138",
                       "City_141","City_142","City_150","City_152","City_158"],
    "logistics":      ["City_17","City_29","City_35","City_38","City_47","City_53",
                       "City_62","City_63","City_72","City_75","City_79","City_84",
                       "City_92","City_98","City_128","City_141","City_142"],
    "shipping":       ["City_35","City_38","City_47","City_51","City_53","City_63",
                       "City_72","City_84","City_92","City_108","City_141"],
    "container":      ["City_35","City_38","City_47","City_51","City_53","City_72",
                       "City_75","City_84","City_92","City_108"],

    # ── Distribution ─────────────────────────────────────────────────────────
    "distribution":   ["City_17","City_23","City_24","City_29","City_31","City_63",
                       "City_85","City_99","City_118","City_143"],

    # ── Manufacturing & Supply Chain ─────────────────────────────────────────
    "supply chain":   ["City_61","City_53","City_47","City_35","City_63","City_84"],
    "manufacturing":  ["City_61","City_16","City_26","City_33","City_44","City_100",
                       "City_132","City_157"],
    "production":     ["City_61","City_1","City_12","City_46","City_71","City_73"],
}

# ---------------------------------------------------------------------------
# LLM output → internal keyword mappings  (expanded for all new countries)
# ---------------------------------------------------------------------------
_COUNTRY_NAME_TO_KEYS: Dict[str, List[str]] = {
    # East Asia
    "china":               ["china"],
    "taiwan":              ["taiwan", "tsmc"],
    "south korea":         ["south korea", "korea", "samsung", "hyundai"],
    "korea":               ["korea"],
    "japan":               ["japan"],
    # Southeast Asia
    "vietnam":             ["vietnam"],
    "thailand":            ["thailand"],
    "singapore":           ["singapore"],
    "malaysia":            ["malaysia"],
    "indonesia":           ["indonesia"],
    "philippines":         ["philippines"],
    "myanmar":             ["myanmar"],
    "sri lanka":           ["sri lanka"],
    # South Asia
    "india":               ["india"],
    "pakistan":            ["pakistan"],
    "bangladesh":          ["bangladesh"],
    # Middle East
    "united arab emirates":["uae"],
    "uae":                 ["uae"],
    "saudi arabia":        ["saudi arabia", "saudi"],
    "qatar":               ["qatar"],
    "kuwait":              ["kuwait"],
    "iran":                ["iran"],
    "israel":              ["israel"],
    "oman":                ["oman"],
    "turkey":              ["turkey"],
    # USA & North America
    "united states":       ["usa", "america"],
    "usa":                 ["usa"],
    "canada":              ["canada"],
    "mexico":              ["mexico"],
    # Latin America
    "brazil":              ["brazil"],
    "argentina":           ["argentina"],
    "chile":               ["chile"],
    "colombia":            ["colombia"],
    "latin america":       ["latin america"],
    # Europe
    "germany":             ["germany"],
    "netherlands":         ["netherlands"],
    "france":              ["france"],
    "italy":               ["italy"],
    "spain":               ["spain"],
    "belgium":             ["belgium"],
    "sweden":              ["sweden"],
    "denmark":             ["denmark"],
    "norway":              ["norway"],
    "switzerland":         ["switzerland"],
    "poland":              ["poland"],
    "czech republic":      ["czech"],
    "hungary":             ["hungary"],
    "romania":             ["romania"],
    "austria":             ["austria"],
    "portugal":            ["portugal"],
    "greece":              ["greece"],
    "united kingdom":      ["uk", "britain"],
    "uk":                  ["uk"],
    "finland":             ["finland"],
    "slovakia":            ["slovakia"],
    "bulgaria":            ["bulgaria"],
    "serbia":              ["serbia"],
    "latvia":              ["latvia"],
    "europe":              ["europe"],
    # Russia & Central Asia
    "russia":              ["russia"],
    "kazakhstan":          ["kazakhstan"],
    "azerbaijan":          ["azerbaijan"],
    # Africa
    "nigeria":             ["nigeria"],
    "egypt":               ["egypt"],
    "south africa":        ["south africa"],
    "kenya":               ["kenya"],
    "ethiopia":            ["ethiopia"],
    "ghana":               ["ghana"],
    "tanzania":            ["tanzania"],
    "morocco":             ["morocco"],
    "africa":              ["africa"],
    # Oceania
    "australia":           ["australia"],
    "new zealand":         ["new zealand"],
    # Geopolitical zones
    "middle east":         ["middle east", "hormuz"],
    "southeast asia":      ["southeast asia"],
    "red sea":             ["red sea"],
    "suez canal":          ["suez"],
    "strait of hormuz":    ["hormuz", "strait of hormuz"],
}

_INDUSTRY_NAME_TO_KEYS: Dict[str, List[str]] = {
    "electronics":       ["electronics", "electronic", "factory"],
    "semiconductors":    ["semiconductor", "semiconductors", "chip", "chips",
                          "microchip", "wafer"],
    "semiconductor":     ["semiconductor", "semiconductors", "chip", "chips"],
    "automotive":        ["automotive", "automobile", "car", "vehicle"],
    "pharmaceuticals":   ["pharmaceutical", "pharmaceuticals", "pharma",
                          "medicine", "vaccine", "drug"],
    "pharmaceutical":    ["pharmaceutical", "pharmaceuticals", "pharma"],
    "textiles":          ["textile", "textiles", "clothing", "apparel", "garment"],
    "textile":           ["textile", "textiles"],
    "oil":               ["oil", "petroleum", "crude", "lng", "energy"],
    "gas":               ["gas", "lng", "energy"],
    "oil and gas":       ["oil", "gas", "petroleum", "crude", "lng", "energy"],
    "petrochemicals":    ["petrochemical"],
    "steel":             ["steel", "metal", "iron"],
    "chemicals":         ["chemical", "chemicals"],
    "chemical":          ["chemical", "chemicals"],
    "aerospace":         ["aerospace", "aviation", "aircraft"],
    "minerals":          ["mineral", "minerals", "mining", "copper",
                          "lithium", "rare earth"],
    "agriculture":       ["agriculture", "food", "grain"],
    "it hardware":       ["it hardware", "server", "data center"],
    "luxury goods":      ["luxury", "luxury goods"],
    "logistics":         ["logistics", "port", "shipping", "container"],
    "port":              ["port", "shipping"],
    "distribution":      ["distribution"],
    "manufacturing":     ["manufacturing", "production", "factory"],
}

# ---------------------------------------------------------------------------
# Severity / category keywords (used by keyword fallback)
# ---------------------------------------------------------------------------
HIGH_SEVERITY_KEYWORDS = {
    "earthquake", "tsunami", "hurricane", "typhoon", "cyclone",
    "wildfire", "war", "conflict", "sanctions", "explosion",
    "nuclear", "collapse", "catastrophic", "critical", "severe",
    "shutdown", "factory shutdown", "closure",
}
MEDIUM_SEVERITY_KEYWORDS = {
    "flood", "flooding", "strike", "protest", "shortage", "disruption",
    "damage", "fire", "storm", "drought",
}
LOW_SEVERITY_KEYWORDS = {
    "delay", "delayed", "slowdown", "congestion", "traffic",
    "minor", "partial", "limited",
}
CATEGORY_MAP = {
    "natural_disaster":   {"earthquake", "tsunami", "hurricane", "typhoon", "cyclone",
                           "flood", "flooding", "wildfire", "storm", "drought", "fire"},
    "labor":              {"strike", "protest", "walkout", "labor", "labour", "workers",
                           "union", "employee"},
    "geopolitical":       {"war", "conflict", "sanctions", "tariff", "ban", "embargo",
                           "political", "government", "trade war", "tensions", "naval"},
    "industrial_accident":{"explosion", "fire", "collapse", "shutdown", "closure",
                           "accident", "incident"},
    "logistics":          {"port", "congestion", "delay", "traffic", "route",
                           "shipping", "container"},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dedup(cities: List[str]) -> List[str]:
    seen, out = set(), []
    for c in cities:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _tokenize(text: str) -> str:
    return text.lower().strip()


def _get_severity_kw(text: str) -> str:
    t = _tokenize(text)
    for kw in HIGH_SEVERITY_KEYWORDS:
        if kw in t:
            return "high"
    for kw in MEDIUM_SEVERITY_KEYWORDS:
        if kw in t:
            return "medium"
    for kw in LOW_SEVERITY_KEYWORDS:
        if kw in t:
            return "low"
    return "medium"


def _get_category_kw(text: str) -> str:
    t = _tokenize(text)
    for cat, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in t:
                return cat
    return "other"


# ---------------------------------------------------------------------------
# LLM-based parser  (Gemini)
# ---------------------------------------------------------------------------

_LLM_PROMPT = """You are a senior supply chain risk analyst with deep knowledge of global trade networks, geopolitics, and industrial dependencies.

Analyse this disruption event and extract structured intelligence:

EVENT: "{event}"

Instructions:
- Consider DIRECT impact: which countries / cities are explicitly mentioned?
- Consider IMPLIED impact: e.g. "Strait of Hormuz tensions" implies oil shipping disruption affecting all of Asia & Europe; "TSMC fab issue" implies global semiconductor shortage
- Consider DOWNSTREAM effects: a factory shutdown in a key hub disrupts industries that depend on it
- Assess severity based on scale, criticality, and recovery difficulty
- Classify the disruption type accurately

Return ONLY valid JSON with exactly these fields:
{{
  "severity": "<high|medium|low>",
  "category": "<natural_disaster|labor|geopolitical|industrial_accident|logistics|other>",
  "affected_countries": ["<country1>", "<country2>"],
  "affected_industries": ["<industry1>", "<industry2>"],
  "reasoning": "<2-3 sentences explaining the supply chain impact, including any implied or downstream effects not obvious from the text>"
}}

For affected_countries use: China, Taiwan, South Korea, Japan, India, Vietnam, Thailand, Singapore, Malaysia, Indonesia, Philippines, United States, Germany, Netherlands, Europe, Middle East, Saudi Arabia, UAE, Nigeria, Egypt, Africa, Southeast Asia

For affected_industries use: electronics, semiconductors, automotive, pharmaceuticals, textiles, oil, gas, oil and gas, petrochemicals, steel, chemicals, aerospace, minerals, agriculture, logistics, port, manufacturing, it hardware, luxury goods

Be comprehensive — list all plausible affected countries and industries, not just the most obvious ones."""


def _llm_parse(text: str) -> dict | None:
    """
    Call Gemini to extract structured disruption data.
    Returns None on any failure (missing key, rate limit, parse error).
    """
    api_key = ""
    for env_name in GEMINI_KEY_ENV_VARS:
        candidate = os.getenv(env_name, "").strip()
        if candidate and not candidate.lower().startswith("your_"):
            api_key = candidate
            break

    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types as genai_types

        client   = genai.Client(api_key=api_key)
        prompt   = _LLM_PROMPT.format(event=text.replace('"', "'"))

        print("  [disruption_input] Calling Gemini for LLM classification …")
        response = client.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = prompt,
            config   = genai_types.GenerateContentConfig(
                temperature       = 0.1,
                max_output_tokens = 1024,
            ),
        )

        raw_text = response.text.strip()
        match    = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            print("  [disruption_input] LLM returned no JSON — falling back")
            return None

        data = json.loads(match.group(0))

        # Validate required fields
        if not all(k in data for k in ("severity", "category", "affected_countries", "affected_industries")):
            return None

        print(
            f"  [disruption_input] LLM classification: severity={data['severity']}, "
            f"category={data['category']}, "
            f"countries={data['affected_countries']}, "
            f"industries={data['affected_industries']}"
        )
        return data

    except Exception as exc:
        print(f"  [disruption_input] LLM parse failed: {exc} — falling back to keywords")
        return None


def _nodes_from_llm(data: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Map LLM output (full names) → City_XX node IDs using the internal lookup tables.
    Returns (city_ids, country_keywords_hit, product_keywords_hit)
    """
    cities: list[str] = []
    country_hit: list[str] = []
    product_hit: list[str] = []

    for country in data.get("affected_countries", []):
        c_lower = country.lower().strip()
        keys    = _COUNTRY_NAME_TO_KEYS.get(c_lower, [c_lower])
        for key in keys:
            if key in COUNTRY_CITY_MAP:
                cities.extend(COUNTRY_CITY_MAP[key])
                country_hit.append(key)

    for industry in data.get("affected_industries", []):
        i_lower = industry.lower().strip()
        keys    = _INDUSTRY_NAME_TO_KEYS.get(i_lower, [i_lower])
        for key in keys:
            if key in PRODUCT_CITY_MAP:
                cities.extend(PRODUCT_CITY_MAP[key])
                product_hit.append(key)

    return _dedup(cities), list(set(country_hit)), list(set(product_hit))


# ---------------------------------------------------------------------------
# Keyword-matching fallback
# ---------------------------------------------------------------------------

def _keyword_parse(text: str) -> Dict[str, Any]:
    t = _tokenize(text)
    cities: list[str] = []
    keywords_hit: list[str] = []
    country_hit: list[str] = []
    product_hit: list[str] = []

    for keyword, node_list in COUNTRY_CITY_MAP.items():
        if keyword in t:
            cities.extend(node_list)
            keywords_hit.append(keyword)
            country_hit.append(keyword)

    for keyword, node_list in PRODUCT_CITY_MAP.items():
        if keyword in t:
            cities.extend(node_list)
            keywords_hit.append(keyword)
            product_hit.append(keyword)

    unique = _dedup(cities)
    if not unique:
        unique       = ["City_1", "City_2", "City_13", "City_24"]
        keywords_hit = ["general"]

    return {
        "event_text":     text,
        "affected_nodes": unique,
        "severity":       _get_severity_kw(text),
        "category":       _get_category_kw(text),
        "keywords_hit":   list(set(keywords_hit)),
        "country_hit":    list(set(country_hit)),
        "product_hit":    list(set(product_hit)),
        "reasoning":      "",
        "llm_source":     "keyword-matching",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_disruption(text: str) -> Dict[str, Any]:
    """
    Parse a free-text disruption event.

    Tries Gemini LLM first for deep contextual understanding.
    Falls back to keyword matching when the API key is absent or the call fails.

    Returns
    -------
    dict with keys:
        event_text      : str
        affected_nodes  : list[str]  e.g. ["City_1", "City_5", …]
        severity        : "high" | "medium" | "low"
        category        : disruption type string
        keywords_hit    : list of matched keywords / node-mapping keys
        country_hit     : countries identified
        product_hit     : industries identified
        reasoning       : LLM explanation (empty string for keyword path)
        llm_source      : "gemini" | "keyword-matching"
    """
    # ── Try LLM ──────────────────────────────────────────────────────────────
    llm_data = _llm_parse(text)

    if llm_data:
        cities, country_hit, product_hit = _nodes_from_llm(llm_data)

        if not cities:
            # LLM returned countries/industries we can't map — fall through to keywords
            print("  [disruption_input] LLM output unmappable — using keyword fallback")
        else:
            return {
                "event_text":     text,
                "affected_nodes": cities,
                "severity":       llm_data.get("severity", "medium"),
                "category":       llm_data.get("category", "other"),
                "keywords_hit":   country_hit + product_hit,
                "country_hit":    country_hit,
                "product_hit":    product_hit,
                "reasoning":      llm_data.get("reasoning", ""),
                "llm_source":     "gemini",
            }

    # ── Keyword fallback ──────────────────────────────────────────────────────
    return _keyword_parse(text)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "Factory shutdown in China affecting electronics supply chain",
        "Tensions at the Strait of Hormuz escalating due to naval exercises",
        "Labor strike at South Korean semiconductor fabs",
        "Severe flooding in Vietnam affecting textile manufacturing",
    ]
    for t in tests:
        print(f"\nINPUT: {t}")
        result = parse_disruption(t)
        print(f"  source   : {result['llm_source']}")
        print(f"  severity : {result['severity']}")
        print(f"  category : {result['category']}")
        print(f"  countries: {result['country_hit']}")
        print(f"  products : {result['product_hit']}")
        print(f"  nodes    : {len(result['affected_nodes'])} nodes")
        if result.get("reasoning"):
            print(f"  reason   : {result['reasoning']}")
