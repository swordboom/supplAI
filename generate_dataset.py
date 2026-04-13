#!/usr/bin/env python3
"""
generate_dataset.py
====================
Generates a large, realistic global supply chain dataset for SupplAI.

Outputs:
  data/supply_chain.csv       — ~160 real city nodes with metadata
  datasets/distance.csv       — directed edges with real haversine distances
  datasets/order_large.csv    — simulated trade orders per edge

Run from project root:
    python generate_dataset.py
"""

import math
import random
import csv
import uuid
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# 1.  City definitions
#     (city_id, city_name, country, region, product_category, tier, risk_factor, lat, lon)
#
#     City_1 … City_70  →  kept IDENTICAL to existing data so all keyword maps still work
#     City_71 onwards   →  new realistic cities
# ─────────────────────────────────────────────────────────────────────────────
CITIES = [
    # ── EXISTING 70 (unchanged) ───────────────────────────────────────────────
    ("City_1",  "Shenzhen",       "China",        "East Asia",     "Electronics",       2, "high",    22.5431,  114.0579),
    ("City_2",  "Shanghai",       "China",        "East Asia",     "Semiconductors",    2, "high",    31.2304,  121.4737),
    ("City_3",  "Beijing",        "China",        "East Asia",     "Automotive",        3, "medium",  39.9042,  116.4074),
    ("City_4",  "Guangzhou",      "China",        "East Asia",     "Textiles",          2, "medium",  23.1291,  113.2644),
    ("City_5",  "Chengdu",        "China",        "East Asia",     "Electronics",       3, "medium",  30.5728,  104.0668),
    ("City_6",  "Wuhan",          "China",        "East Asia",     "Steel",             3, "high",    30.5928,  114.3055),
    ("City_7",  "Xian",           "China",        "East Asia",     "Aerospace",         3, "medium",  34.3416,  108.9398),
    ("City_8",  "Tianjin",        "China",        "East Asia",     "Chemicals",         3, "medium",  39.1235,  117.1980),
    ("City_9",  "Chongqing",      "China",        "East Asia",     "Automotive",        3, "medium",  29.4316,  106.9123),
    ("City_10", "Nanjing",        "China",        "East Asia",     "Pharmaceuticals",   2, "medium",  32.0603,  118.7969),
    ("City_11", "Hangzhou",       "China",        "East Asia",     "E-Commerce",        2, "low",     30.2741,  120.1551),
    ("City_12", "Dongguan",       "China",        "East Asia",     "Electronics",       2, "high",    23.0207,  113.7519),
    ("City_13", "Mumbai",         "India",        "South Asia",    "Pharmaceuticals",   2, "medium",  19.0760,   72.8777),
    ("City_14", "Bangalore",      "India",        "South Asia",    "IT Hardware",       2, "low",     12.9716,   77.5946),
    ("City_15", "Chennai",        "India",        "South Asia",    "Automotive",        2, "medium",  13.0827,   80.2707),
    ("City_16", "Pune",           "India",        "South Asia",    "Manufacturing",     3, "medium",  18.5204,   73.8567),
    ("City_17", "Delhi",          "India",        "South Asia",    "Distribution",      1, "low",     28.7041,   77.1025),
    ("City_18", "Hyderabad",      "India",        "South Asia",    "Pharmaceuticals",   2, "medium",  17.3850,   78.4867),
    ("City_19", "Kolkata",        "India",        "South Asia",    "Textiles",          3, "medium",  22.5726,   88.3639),
    ("City_20", "Surat",          "India",        "South Asia",    "Textiles",          3, "low",     21.1702,   72.8311),
    ("City_21", "Ahmedabad",      "India",        "South Asia",    "Chemicals",         3, "medium",  23.0225,   72.5714),
    ("City_22", "Coimbatore",     "India",        "South Asia",    "Machinery",         3, "low",     11.0168,   76.9558),
    ("City_23", "New York",       "USA",          "North America", "Distribution",      1, "low",     40.7128,  -74.0060),
    ("City_24", "Los Angeles",    "USA",          "North America", "Port Distribution", 1, "low",     34.0522, -118.2437),
    ("City_25", "Houston",        "USA",          "North America", "Oil Gas",           4, "high",    29.7604,  -95.3698),
    ("City_26", "Chicago",        "USA",          "North America", "Manufacturing",     2, "medium",  41.8781,  -87.6298),
    ("City_27", "San Jose",       "USA",          "North America", "Semiconductors",    2, "medium",  37.3382, -121.8863),
    ("City_28", "Seattle",        "USA",          "North America", "Aerospace",         2, "medium",  47.6062, -122.3321),
    ("City_29", "Atlanta",        "USA",          "North America", "Distribution",      1, "low",     33.7490,  -84.3880),
    ("City_30", "Detroit",        "USA",          "North America", "Automotive",        2, "medium",  42.3314,  -83.0458),
    ("City_31", "Dallas",         "USA",          "North America", "Distribution",      1, "low",     32.7767,  -96.7970),
    ("City_32", "Phoenix",        "USA",          "North America", "Electronics",       2, "medium",  33.4484, -112.0740),
    ("City_33", "Berlin",         "Germany",      "Europe",        "Automotive",        2, "medium",  52.5200,   13.4050),
    ("City_34", "Munich",         "Germany",      "Europe",        "Machinery",         2, "medium",  48.1351,   11.5820),
    ("City_35", "Hamburg",        "Germany",      "Europe",        "Port Distribution", 1, "low",     53.5753,   10.0153),
    ("City_36", "Frankfurt",      "Germany",      "Europe",        "Chemicals",         3, "low",     50.1109,    8.6821),
    ("City_37", "Stuttgart",      "Germany",      "Europe",        "Automotive",        2, "medium",  48.7758,    9.1829),
    ("City_38", "Rotterdam",      "Netherlands",  "Europe",        "Port Distribution", 1, "low",     51.9244,    4.4777),
    ("City_39", "Paris",          "France",       "Europe",        "Luxury Goods",      2, "low",     48.8566,    2.3522),
    ("City_40", "Lyon",           "France",       "Europe",        "Pharmaceuticals",   2, "medium",  45.7640,    4.8357),
    ("City_41", "Milan",          "Italy",        "Europe",        "Textiles",          2, "low",     45.4654,    9.1859),
    ("City_42", "Madrid",         "Spain",        "Europe",        "Automotive",        2, "medium",  40.4168,   -3.7038),
    ("City_43", "Tokyo",          "Japan",        "East Asia",     "Electronics",       2, "medium",  35.6762,  139.6503),
    ("City_44", "Osaka",          "Japan",        "East Asia",     "Manufacturing",     2, "medium",  34.6937,  135.5023),
    ("City_45", "Nagoya",         "Japan",        "East Asia",     "Automotive",        2, "medium",  35.1815,  136.9066),
    ("City_46", "Seoul",          "South Korea",  "East Asia",     "Semiconductors",    2, "high",    37.5665,  126.9780),
    ("City_47", "Busan",          "South Korea",  "East Asia",     "Port Logistics",    1, "low",     35.1796,  129.0756),
    ("City_48", "Taipei",         "Taiwan",       "East Asia",     "Semiconductors",    2, "high",    25.0330,  121.5654),
    ("City_49", "Incheon",        "South Korea",  "East Asia",     "Electronics",       2, "medium",  37.4563,  126.7052),
    ("City_50", "Yokohama",       "Japan",        "East Asia",     "Automotive",        2, "medium",  35.4437,  139.6380),
    ("City_51", "Kaohsiung",      "Taiwan",       "East Asia",     "Port Distribution", 1, "low",     22.6273,  120.3014),
    ("City_52", "Hiroshima",      "Japan",        "East Asia",     "Automotive",        3, "medium",  34.3963,  132.4596),
    ("City_53", "Singapore",      "Singapore",    "Southeast Asia","Port Distribution", 1, "low",      1.3521,  103.8198),
    ("City_54", "Bangkok",        "Thailand",     "Southeast Asia","Automotive",        2, "medium",  13.7563,  100.5018),
    ("City_55", "Ho Chi Minh",    "Vietnam",      "Southeast Asia","Textiles",          3, "medium",  10.8231,  106.6297),
    ("City_56", "Jakarta",        "Indonesia",    "Southeast Asia","Raw Materials",     4, "medium",  -6.2088,  106.8456),
    ("City_57", "Kuala Lumpur",   "Malaysia",     "Southeast Asia","Electronics",       2, "medium",   3.1390,  101.6869),
    ("City_58", "Manila",         "Philippines",  "Southeast Asia","Electronics",       3, "medium",  14.5995,  120.9842),
    ("City_59", "Sydney",         "Australia",    "Oceania",       "Minerals",          4, "low",    -33.8688,  151.2093),
    ("City_60", "Melbourne",      "Australia",    "Oceania",       "Agriculture",       4, "low",    -37.8136,  144.9631),
    ("City_61", "Hanoi",          "Vietnam",      "Southeast Asia","Electronics",       2, "high",    21.0285,  105.8542),
    ("City_62", "Colombo",        "Sri Lanka",    "South Asia",    "Port Logistics",    1, "low",      6.9271,   79.8612),
    ("City_63", "Dubai",          "UAE",          "Middle East",   "Port Distribution", 1, "low",     25.2048,   55.2708),
    ("City_64", "Riyadh",         "Saudi Arabia", "Middle East",   "Oil Gas",           4, "high",    24.7136,   46.6753),
    ("City_65", "Lagos",          "Nigeria",      "Africa",        "Oil Minerals",      4, "high",     6.5244,    3.3792),
    ("City_66", "Cairo",          "Egypt",        "Africa",        "Port Distribution", 1, "low",     30.0444,   31.2357),
    ("City_67", "Johannesburg",   "South Africa", "Africa",        "Minerals",          4, "medium", -26.2041,   28.0473),
    ("City_68", "Istanbul",       "Turkey",       "Middle East",   "Textiles",          2, "medium",  41.0082,   28.9784),
    ("City_69", "Tehran",         "Iran",         "Middle East",   "Petrochemicals",    4, "high",    35.6892,   51.3890),
    ("City_70", "Nairobi",        "Kenya",        "Africa",        "Agriculture",       4, "low",     -1.2921,   36.8219),

    # ── NEW CITIES (City_71 onwards) ─────────────────────────────────────────
    # China expansion — major manufacturing hubs
    ("City_71", "Suzhou",         "China",        "East Asia",     "Electronics",       2, "high",    31.2990,  120.5853),
    ("City_72", "Qingdao",        "China",        "East Asia",     "Port Distribution", 1, "medium",  36.0671,  120.3826),
    ("City_73", "Zhengzhou",      "China",        "East Asia",     "Electronics",       2, "high",    34.7466,  113.6254),
    ("City_74", "Hefei",          "China",        "East Asia",     "Semiconductors",    3, "medium",  31.8206,  117.2272),
    ("City_75", "Ningbo",         "China",        "East Asia",     "Port Logistics",    1, "medium",  29.8683,  121.5440),
    ("City_76", "Wuxi",           "China",        "East Asia",     "Semiconductors",    3, "high",    31.4912,  120.3119),

    # Taiwan expansion
    ("City_77", "Hsinchu",        "Taiwan",       "East Asia",     "Semiconductors",    2, "high",    24.8138,  120.9675),
    ("City_78", "Taichung",       "Taiwan",       "East Asia",     "Machinery",         3, "medium",  24.1477,  120.6736),

    # Japan expansion
    ("City_79", "Kobe",           "Japan",        "East Asia",     "Port Logistics",    1, "low",     34.6901,  135.1956),
    ("City_80", "Fukuoka",        "Japan",        "East Asia",     "Semiconductors",    3, "medium",  33.5904,  130.4017),

    # South Korea expansion
    ("City_81", "Ulsan",          "South Korea",  "East Asia",     "Automotive",        2, "medium",  35.5384,  129.3114),
    ("City_82", "Suwon",          "South Korea",  "East Asia",     "Electronics",       2, "high",    37.2636,  127.0286),

    # Europe expansion
    ("City_83", "Barcelona",      "Spain",        "Europe",        "Automotive",        2, "medium",  41.3851,    2.1734),
    ("City_84", "Antwerp",        "Belgium",      "Europe",        "Port Distribution", 1, "low",     51.2194,    4.4025),
    ("City_85", "Amsterdam",      "Netherlands",  "Europe",        "Distribution",      1, "low",     52.3676,    4.9041),
    ("City_86", "Stockholm",      "Sweden",       "Europe",        "IT Hardware",       2, "low",     59.3293,   18.0686),
    ("City_87", "Gothenburg",     "Sweden",       "Europe",        "Automotive",        2, "low",     57.7089,   11.9746),
    ("City_88", "Copenhagen",     "Denmark",      "Europe",        "Pharmaceuticals",   2, "low",     55.6761,   12.5683),
    ("City_89", "Oslo",           "Norway",       "Europe",        "Oil Gas",           4, "low",     59.9139,   10.7522),
    ("City_90", "Zurich",         "Switzerland",  "Europe",        "Pharmaceuticals",   2, "low",     47.3769,    8.5417),
    ("City_91", "Warsaw",         "Poland",       "Europe",        "Automotive",        3, "medium",  52.2297,   21.0122),
    ("City_92", "Gdansk",         "Poland",       "Europe",        "Port Logistics",    1, "low",     54.3520,   18.6466),
    ("City_93", "Prague",         "Czech Republic","Europe",       "Automotive",        3, "medium",  50.0755,   14.4378),
    ("City_94", "Budapest",       "Hungary",      "Europe",        "Automotive",        3, "medium",  47.4979,   19.0402),
    ("City_95", "Bucharest",      "Romania",      "Europe",        "Automotive",        3, "medium",  44.4268,   26.1025),
    ("City_96", "Vienna",         "Austria",      "Europe",        "Machinery",         2, "low",     48.2082,   16.3738),
    ("City_97", "Lisbon",         "Portugal",     "Europe",        "Textiles",          3, "low",     38.7223,   -9.1393),
    ("City_98", "Athens",         "Greece",       "Europe",        "Port Logistics",    1, "medium",  37.9838,   23.7275),

    # United Kingdom
    ("City_99",  "London",        "UK",           "Europe",        "Distribution",      1, "low",     51.5074,   -0.1278),
    ("City_100", "Manchester",    "UK",           "Europe",        "Manufacturing",     2, "medium",  53.4808,   -2.2426),
    ("City_101", "Birmingham",    "UK",           "Europe",        "Automotive",        2, "medium",  52.4862,   -1.8904),

    # USA expansion
    ("City_102", "Boston",        "USA",          "North America", "Pharmaceuticals",   2, "low",     42.3601,  -71.0589),
    ("City_103", "Austin",        "USA",          "North America", "Semiconductors",    2, "medium",  30.2672,  -97.7431),
    ("City_104", "San Diego",     "USA",          "North America", "Pharmaceuticals",   2, "low",     32.7157, -117.1611),
    ("City_105", "Minneapolis",   "USA",          "North America", "Manufacturing",     3, "low",     44.9778,  -93.2650),
    ("City_106", "Portland",      "USA",          "North America", "Port Distribution", 1, "low",     45.5051, -122.6750),

    # Canada
    ("City_107", "Toronto",       "Canada",       "North America", "Automotive",        2, "low",     43.6532,  -79.3832),
    ("City_108", "Vancouver",     "Canada",       "North America", "Port Distribution", 1, "low",     49.2827, -123.1207),
    ("City_109", "Montreal",      "Canada",       "North America", "Aerospace",         2, "low",     45.5017,  -73.5673),

    # Mexico
    ("City_110", "Mexico City",   "Mexico",       "Latin America", "Automotive",        2, "medium",  19.4326,  -99.1332),
    ("City_111", "Monterrey",     "Mexico",       "Latin America", "Steel",             3, "medium",  25.6866, -100.3161),
    ("City_112", "Guadalajara",   "Mexico",       "Latin America", "Electronics",       2, "medium",  20.6597, -103.3496),
    ("City_113", "Tijuana",       "Mexico",       "Latin America", "Electronics",       3, "medium",  32.5149, -117.0382),

    # Latin America
    ("City_114", "Sao Paulo",     "Brazil",       "Latin America", "Automotive",        2, "medium", -23.5505,  -46.6333),
    ("City_115", "Rio de Janeiro","Brazil",       "Latin America", "Oil Gas",           4, "high",   -22.9068,  -43.1729),
    ("City_116", "Buenos Aires",  "Argentina",    "Latin America", "Agriculture",       4, "low",    -34.6037,  -58.3816),
    ("City_117", "Santiago",      "Chile",        "Latin America", "Minerals",          4, "low",    -33.4489,  -70.6693),
    ("City_118", "Bogota",        "Colombia",     "Latin America", "Distribution",      2, "medium",   4.7110,  -74.0721),

    # Middle East expansion
    ("City_119", "Abu Dhabi",     "UAE",          "Middle East",   "Oil Gas",           4, "low",     24.4539,   54.3773),
    ("City_120", "Doha",          "Qatar",        "Middle East",   "Oil Gas",           4, "medium",  25.2854,   51.5310),
    ("City_121", "Kuwait City",   "Kuwait",       "Middle East",   "Oil Gas",           4, "medium",  29.3759,   47.9774),
    ("City_122", "Jeddah",        "Saudi Arabia", "Middle East",   "Port Logistics",    1, "medium",  21.4858,   39.1925),
    ("City_123", "Tel Aviv",      "Israel",       "Middle East",   "Semiconductors",    2, "medium",  32.0853,   34.7818),
    ("City_124", "Muscat",        "Oman",         "Middle East",   "Oil Gas",           4, "medium",  23.5880,   58.3829),
    ("City_125", "Ankara",        "Turkey",       "Middle East",   "Automotive",        3, "medium",  39.9334,   32.8597),

    # South Asia expansion
    ("City_126", "Karachi",       "Pakistan",     "South Asia",    "Textiles",          3, "medium",  24.8607,   67.0011),
    ("City_127", "Dhaka",         "Bangladesh",   "South Asia",    "Textiles",          3, "medium",  23.8103,   90.4125),
    ("City_128", "Kochi",         "India",        "South Asia",    "Port Logistics",    1, "low",      9.9312,   76.2673),
    ("City_129", "Visakhapatnam", "India",        "South Asia",    "Steel",             3, "medium",  17.6868,   83.2185),

    # Southeast Asia expansion
    ("City_130", "Penang",        "Malaysia",     "Southeast Asia","Semiconductors",    2, "medium",   5.4164,  100.3327),
    ("City_131", "Cebu",          "Philippines",  "Southeast Asia","Electronics",       3, "medium",  10.3157,  123.8854),
    ("City_132", "Surabaya",      "Indonesia",    "Southeast Asia","Manufacturing",     3, "medium",  -7.2575,  112.7521),
    ("City_133", "Yangon",        "Myanmar",      "Southeast Asia","Textiles",          4, "medium",  16.8661,   96.1951),
    ("City_134", "Da Nang",       "Vietnam",      "Southeast Asia","Electronics",       3, "medium",  16.0544,  108.2022),
    ("City_135", "Batam",         "Indonesia",    "Southeast Asia","Electronics",       3, "medium",   1.0456,  104.0305),

    # Africa expansion
    ("City_136", "Casablanca",    "Morocco",      "Africa",        "Textiles",          3, "medium",  33.5731,   -7.5898),
    ("City_137", "Alexandria",    "Egypt",        "Africa",        "Port Logistics",    1, "medium",  31.2001,   29.9187),
    ("City_138", "Durban",        "South Africa", "Africa",        "Port Distribution", 1, "medium", -29.8587,   31.0218),
    ("City_139", "Addis Ababa",   "Ethiopia",     "Africa",        "Textiles",          4, "low",      9.0320,   38.7469),
    ("City_140", "Accra",         "Ghana",        "Africa",        "Oil Minerals",      4, "medium",   5.6037,   -0.1870),
    ("City_141", "Dar es Salaam", "Tanzania",     "Africa",        "Port Logistics",    1, "medium",  -6.7924,   39.2083),
    ("City_142", "Mombasa",       "Kenya",        "Africa",        "Port Logistics",    1, "low",     -4.0435,   39.6682),

    # Russia & Central Asia
    ("City_143", "Moscow",        "Russia",       "Europe",        "Distribution",      2, "high",    55.7558,   37.6173),
    ("City_144", "St Petersburg", "Russia",       "Europe",        "Automotive",        3, "high",    59.9311,   30.3609),
    ("City_145", "Almaty",        "Kazakhstan",   "Central Asia",  "Raw Materials",     4, "medium",  43.2220,   76.8512),
    ("City_146", "Baku",          "Azerbaijan",   "Central Asia",  "Oil Gas",           4, "medium",  40.4093,   49.8671),

    # Oceania expansion
    ("City_147", "Brisbane",      "Australia",    "Oceania",       "Minerals",          4, "low",    -27.4698,  153.0251),
    ("City_148", "Perth",         "Australia",    "Oceania",       "Minerals",          4, "low",    -31.9505,  115.8605),
    ("City_149", "Auckland",      "New Zealand",  "Oceania",       "Agriculture",       4, "low",    -36.8509,  174.7645),

    # More Europe
    ("City_150", "Marseille",     "France",       "Europe",        "Port Logistics",    1, "low",     43.2965,    5.3698),
    ("City_151", "Turin",         "Italy",        "Europe",        "Automotive",        2, "medium",  45.0703,    7.6869),
    ("City_152", "Genoa",         "Italy",        "Europe",        "Port Logistics",    1, "low",     44.4056,    8.9463),
    ("City_153", "Helsinki",      "Finland",      "Europe",        "IT Hardware",       2, "low",     60.1699,   24.9384),
    ("City_154", "Bratislava",    "Slovakia",     "Europe",        "Automotive",        3, "medium",  48.1486,   17.1077),
    ("City_155", "Sofia",         "Bulgaria",     "Europe",        "Automotive",        3, "medium",  42.6977,   23.3219),
    ("City_156", "Belgrade",      "Serbia",       "Europe",        "Automotive",        3, "medium",  44.8176,   20.4633),
    ("City_157", "Riga",          "Latvia",       "Europe",        "Manufacturing",     3, "medium",  56.9460,   24.1059),
    ("City_158", "Gdynia",        "Poland",       "Europe",        "Port Logistics",    1, "low",     54.5189,   18.5305),
    ("City_159", "Valencia",      "Spain",        "Europe",        "Textiles",          3, "low",     39.4699,   -0.3763),
]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Industry supply-flow rules  (source_industry → set of valid destination industries)
# ─────────────────────────────────────────────────────────────────────────────
SUPPLY_TO = {
    # Deep upstream → intermediate
    "Oil Gas":           {"Petrochemicals", "Chemicals", "Port Distribution", "Distribution"},
    "Oil Minerals":      {"Petrochemicals", "Steel", "Chemicals", "Port Distribution"},
    "Minerals":          {"Steel", "Semiconductors", "Chemicals", "Manufacturing", "Electronics"},
    "Raw Materials":     {"Manufacturing", "Steel", "Chemicals", "Textiles"},
    "Agriculture":       {"Textiles", "Chemicals", "Distribution", "E-Commerce"},

    # Intermediate → manufacturing
    "Petrochemicals":    {"Chemicals", "Textiles", "Manufacturing"},
    "Steel":             {"Automotive", "Aerospace", "Machinery", "Manufacturing"},
    "Chemicals":         {"Pharmaceuticals", "Textiles", "Electronics", "Manufacturing"},
    "Semiconductors":    {"Electronics", "IT Hardware", "Automotive", "Aerospace", "Manufacturing"},
    "Textiles":          {"Distribution", "E-Commerce", "Manufacturing"},
    "Machinery":         {"Automotive", "Aerospace", "Manufacturing", "Electronics"},

    # Manufacturing → distribution
    "Electronics":       {"Distribution", "E-Commerce", "Port Distribution"},
    "IT Hardware":       {"Distribution", "E-Commerce"},
    "Automotive":        {"Distribution", "Port Distribution"},
    "Pharmaceuticals":   {"Distribution", "E-Commerce"},
    "Aerospace":         {"Distribution"},
    "Manufacturing":     {"Distribution", "E-Commerce", "Port Distribution"},
    "Luxury Goods":      {"Distribution", "E-Commerce"},

    # Hubs → anywhere downstream
    "Distribution":      {"E-Commerce"},
    "Port Distribution": {"Distribution", "E-Commerce", "Automotive", "Electronics",
                          "Pharmaceuticals", "IT Hardware", "Machinery"},
    "Port Logistics":    {"Distribution", "Port Distribution", "E-Commerce"},
    "Logistics":         {"Distribution", "E-Commerce"},
    "E-Commerce":        set(),
}

# Which industries trade globally (no distance cap)
GLOBAL_INDUSTRIES = {
    "Oil Gas", "Semiconductors", "Electronics", "Automotive",
    "Port Distribution", "Port Logistics", "Distribution",
    "Pharmaceuticals", "Minerals", "Oil Minerals", "Luxury Goods",
    "Aerospace", "IT Hardware",
}

# Adjacent region pairs (symmetric)
REGION_NEIGHBORS = {
    frozenset({"East Asia", "Southeast Asia"}),
    frozenset({"East Asia", "South Asia"}),
    frozenset({"East Asia", "Oceania"}),
    frozenset({"East Asia", "North America"}),   # trans-Pacific
    frozenset({"Southeast Asia", "South Asia"}),
    frozenset({"Southeast Asia", "Oceania"}),
    frozenset({"Southeast Asia", "Middle East"}),
    frozenset({"South Asia", "Middle East"}),
    frozenset({"Middle East", "Europe"}),
    frozenset({"Middle East", "Africa"}),
    frozenset({"Europe", "Africa"}),
    frozenset({"Europe", "North America"}),      # trans-Atlantic
    frozenset({"Europe", "Central Asia"}),
    frozenset({"Africa", "Latin America"}),      # Atlantic crossing
    frozenset({"North America", "Latin America"}),
    frozenset({"Latin America", "Europe"}),
    frozenset({"Central Asia", "Middle East"}),
    frozenset({"Central Asia", "South Asia"}),
    frozenset({"Oceania", "North America"}),
}


def _are_connected_regions(r1: str, r2: str) -> bool:
    return r1 == r2 or frozenset({r1, r2}) in REGION_NEIGHBORS


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Haversine distance (returns km)
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Edge generation
# ─────────────────────────────────────────────────────────────────────────────
def should_add_edge(src: dict, dst: dict, dist_km: float) -> bool:
    si, di = src["industry"], dst["industry"]

    # Must satisfy industry flow
    if di not in SUPPLY_TO.get(si, set()):
        return False

    sr, dr = src["region"], dst["region"]

    # Port-to-port: always connect worldwide
    if si in {"Port Distribution", "Port Logistics"} and di in {"Port Distribution", "Port Logistics"}:
        return True

    # Same region: always connect if flow allows
    if sr == dr:
        return True

    # Neighboring regions: connect if distance < 8,000 km
    if _are_connected_regions(sr, dr):
        if dist_km < 8000:
            return True
        # Global commodities bridge any distance in neighboring regions
        if si in GLOBAL_INDUSTRIES or di in GLOBAL_INDUSTRIES:
            return True

    # Non-neighboring regions: only global industries, no distance cap
    if si in GLOBAL_INDUSTRIES and di in GLOBAL_INDUSTRIES:
        return True

    return False


def add_hub_edges(city_lookup: dict, edges: set):
    """Ports connect to all distribution/manufacturing hubs globally."""
    port_ids  = [cid for cid, c in city_lookup.items()
                 if c["industry"] in {"Port Distribution", "Port Logistics"}]
    other_ids = [cid for cid, c in city_lookup.items()
                 if c["industry"] in {"Distribution", "Electronics", "Automotive",
                                      "Pharmaceuticals", "IT Hardware", "Manufacturing",
                                      "Luxury Goods", "E-Commerce", "Aerospace"}]

    for pid in port_ids:
        pc = city_lookup[pid]
        for oid in other_ids:
            if pid == oid:
                continue
            oc = city_lookup[oid]
            dist = haversine_km(pc["lat"], pc["lon"], oc["lat"], oc["lon"])
            # Port → customer city (if reachable within 15,000 km)
            if dist < 15000:
                edges.add((pid, oid, round(dist * 1000)))
            # Customer city → port (for export)
            if dist < 12000:
                edges.add((oid, pid, round(dist * 1000)))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Order material maps
# ─────────────────────────────────────────────────────────────────────────────
INDUSTRY_MATERIALS = {
    "Semiconductors":    ["SEMI-WAFER", "SEMI-CHIP", "SEMI-MEM", "SEMI-PROC"],
    "Electronics":       ["ELEC-PCB",   "ELEC-DISP", "ELEC-COMP","ELEC-MODUL"],
    "Automotive":        ["AUTO-ENG",   "AUTO-TRNS", "AUTO-BODY","AUTO-ELEC"],
    "Steel":             ["STL-COIL",   "STL-BEAM",  "STL-PLATE","STL-PIPE"],
    "Chemicals":         ["CHEM-POLY",  "CHEM-ACID", "CHEM-SLVT","CHEM-FERT"],
    "Pharmaceuticals":   ["PHARM-API",  "PHARM-TABL","PHARM-INFT","PHARM-VCNE"],
    "Oil Gas":           ["OIL-CRUDE",  "OIL-LNG",   "OIL-RFND", "OIL-PETRO"],
    "Oil Minerals":      ["OIL-CRUDE",  "MIN-IRON",  "MIN-COPP", "MIN-BAUX"],
    "Minerals":          ["MIN-IRON",   "MIN-COPP",  "MIN-RARE", "MIN-COBL"],
    "Raw Materials":     ["RAW-WOOD",   "RAW-RUBB",  "RAW-CTTN", "RAW-SAND"],
    "Agriculture":       ["AGR-GRAIN",  "AGR-CTTN",  "AGR-SUGR", "AGR-COFE"],
    "Textiles":          ["TEX-FABR",   "TEX-YARN",  "TEX-APPRL","TEX-DENIM"],
    "Machinery":         ["MACH-TOOL",  "MACH-PUMP", "MACH-TURB","MACH-ROBT"],
    "IT Hardware":       ["IT-SERVER",  "IT-STOR",   "IT-NETW",  "IT-PCMP"],
    "Aerospace":         ["AERO-AIRFR", "AERO-ENGIN","AERO-AVNC","AERO-COMP"],
    "Luxury Goods":      ["LUX-WATCH",  "LUX-LEATH", "LUX-JWLRY","LUX-PRFM"],
    "Petrochemicals":    ["PETRO-PLST", "PETRO-RESIN","PETRO-FBRE","PETRO-RUBB"],
    "Manufacturing":     ["MFG-PARTS",  "MFG-ASSEM", "MFG-COMP", "MFG-SUBAS"],
    "Distribution":      ["DIST-PALT",  "DIST-CTNR", "DIST-CARG","DIST-PARCEL"],
    "Port Distribution": ["PORT-CTNR",  "PORT-BULK", "PORT-TANKER","PORT-BRKBULK"],
    "Port Logistics":    ["PORT-CTNR",  "PORT-CARGO","PORT-LIFT", "PORT-RORO"],
    "Logistics":         ["LOG-TRUCK",  "LOG-RAIL",  "LOG-AIR",  "LOG-SEA"],
    "E-Commerce":        ["ECOM-PARCEL","ECOM-EXPR", "ECOM-LTPK","ECOM-PALLET"],
}

DANGER_BY_INDUSTRY = {
    "Oil Gas": "type_4", "Oil Minerals": "type_4", "Petrochemicals": "type_4",
    "Chemicals": "type_3", "Minerals": "type_2", "Raw Materials": "type_2",
    "Pharmaceuticals": "type_2", "Aerospace": "type_2",
}

WEIGHT_RANGE = {
    "Oil Gas":    (5_000_000, 50_000_000),  "Steel":        (2_000_000, 20_000_000),
    "Minerals":   (2_000_000, 15_000_000),  "Raw Materials":(1_000_000, 10_000_000),
    "Chemicals":  (500_000,   5_000_000),   "Automotive":   (1_000_000, 8_000_000),
    "Electronics":(50_000,    1_000_000),   "Semiconductors":(10_000,   200_000),
    "Pharmaceuticals":(5_000, 100_000),     "Aerospace":    (100_000,  2_000_000),
    "IT Hardware":(50_000,    500_000),     "Textiles":     (100_000,  2_000_000),
    "Machinery":  (500_000,   5_000_000),   "Luxury Goods": (1_000,    50_000),
}

BASE_DATE = datetime(2022, 1, 1)

# ─────────────────────────────────────────────────────────────────────────────
# City archetypes — determines order volume multiplier per city
# ─────────────────────────────────────────────────────────────────────────────
# port_hub: massive throughput (5x)
# manufacturing_core: high volume, heavy weight (3x)
# regional_hub: moderate (2x)
# emerging: low, irregular (1x, sparse dates)
# normal: baseline (1x)
CITY_ARCHETYPE = {}

# Port mega-hubs → very high volume
for _cid in ["City_53","City_24","City_38","City_35","City_63","City_51",
             "City_72","City_66","City_84","City_47","City_75","City_79",
             "City_92","City_108","City_106","City_138","City_137","City_141",
             "City_150","City_152","City_128","City_142","City_62","City_122"]:
    CITY_ARCHETYPE[_cid] = "port_hub"

# Manufacturing cores → heavy weight, regular
for _cid in ["City_1","City_2","City_46","City_48","City_43","City_27",
             "City_33","City_26","City_30","City_37","City_44","City_45",
             "City_34","City_71","City_76","City_77","City_82","City_130"]:
    CITY_ARCHETYPE[_cid] = "manufacturing_core"

# Emerging/irregular cities → low volume, sporadic
for _cid in ["City_70","City_65","City_69","City_139","City_140","City_133",
             "City_145","City_146","City_143","City_144","City_149"]:
    CITY_ARCHETYPE[_cid] = "emerging"

# ── Anomaly cities (injected unusual behaviour) ───────────────────────────────
# Format: city_id → (anomaly_type, description)
# anomaly_type controls what gets distorted in orders
ANOMALY_CITIES = {
    # Sudden volume collapse (city goes nearly silent)
    "City_3":   "volume_collapse",   # Beijing
    "City_13":  "volume_collapse",   # Mumbai
    "City_56":  "volume_collapse",   # Jakarta

    # Weight spike (3-5x heavier than normal — unusual cargo)
    "City_6":   "weight_spike",      # Wuhan
    "City_64":  "weight_spike",      # Riyadh
    "City_25":  "weight_spike",      # Houston

    # Dangerous goods surge (normally safe goods suddenly flagged as hazmat)
    "City_48":  "danger_surge",      # Taipei
    "City_74":  "danger_surge",      # Hefei

    # Deadline compression (orders arrive with almost no lead time — panic buying)
    "City_57":  "deadline_crunch",   # Kuala Lumpur
    "City_46":  "deadline_crunch",   # Seoul

    # Going silent then massive spike (bullwhip effect)
    "City_55":  "bullwhip",          # Ho Chi Minh
    "City_115": "bullwhip",          # Rio de Janeiro
}


def _orders_for_edge(src_id: str, dst_id: str, src_industry: str, order_seq: int):
    """
    Generate orders for one edge, applying city archetypes and anomaly patterns.
    Returns list of order rows.
    """
    archetype    = CITY_ARCHETYPE.get(src_id, "normal")
    anomaly_type = ANOMALY_CITIES.get(src_id)

    # ── Base order count by archetype ────────────────────────────────────────
    if archetype == "port_hub":
        n_orders = random.randint(10, 18)
    elif archetype == "manufacturing_core":
        n_orders = random.randint(6, 12)
    elif archetype == "emerging":
        n_orders = random.randint(1, 3)
    else:
        n_orders = random.randint(3, 7)

    # Anomaly: volume collapse → only 0-1 orders
    if anomaly_type == "volume_collapse":
        n_orders = random.randint(0, 1)

    # Anomaly: bullwhip — half the edges get 0 orders, other half get 20+
    if anomaly_type == "bullwhip":
        n_orders = 0 if random.random() < 0.6 else random.randint(15, 25)

    rows = []
    lo, hi = WEIGHT_RANGE.get(src_industry, (10_000, 500_000))
    danger_base = DANGER_BY_INDUSTRY.get(src_industry, "type_1")

    for _ in range(n_orders):
        mat    = random.choice(INDUSTRY_MATERIALS.get(src_industry, ["GEN-CARGO"]))
        mat_id = f"{mat}-{random.randint(1000, 9999)}"
        item_id = f"P{order_seq:02d}-{uuid.uuid4().hex[:8]}"
        order_id = f"A{order_seq + 140000:06d}"

        # ── Date range ───────────────────────────────────────────────────────
        avail    = BASE_DATE + timedelta(days=random.randint(0, 900))

        if anomaly_type == "deadline_crunch":
            # Only 0-2 days lead time instead of 3-30
            deadline = avail + timedelta(days=random.randint(0, 2))
        else:
            deadline = avail + timedelta(days=random.randint(3, 30))

        # ── Weight ───────────────────────────────────────────────────────────
        weight = random.randint(lo, hi)
        if anomaly_type == "weight_spike":
            weight = int(weight * random.uniform(3.0, 5.5))  # 3-5x heavier

        # ── Danger type ──────────────────────────────────────────────────────
        danger = danger_base
        if anomaly_type == "danger_surge":
            # Force high-danger regardless of industry
            danger = random.choice(["type_3", "type_4"])

        area = random.randint(500, 100_000)
        rows.append([order_id, mat_id, item_id, src_id, dst_id,
                     avail.strftime("%Y-%m-%d %H:%M:%S"),
                     deadline.strftime("%Y-%m-%d %H:%M:%S"),
                     danger, area, weight])
        order_seq += 1

    return rows, order_seq


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main generation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating dataset with {len(CITIES)} cities …")

    # Build city lookup
    city_lookup = {}
    for row in CITIES:
        cid = row[0]
        city_lookup[cid] = {
            "city_name": row[1], "country": row[2], "region": row[3],
            "industry":  row[4], "tier":    row[5], "risk":   row[6],
            "lat": row[7], "lon": row[8],
        }

    # ── Write supply_chain.csv ────────────────────────────────────────────────
    supply_path = PROJECT_ROOT / "data" / "supply_chain.csv"
    supply_path.parent.mkdir(exist_ok=True)
    with open(supply_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["city_id","city_name","country","region",
                    "product_category","tier","risk_factor","lat","lon"])
        for row in CITIES:
            w.writerow(row)
    print(f"  OK supply_chain.csv  ({len(CITIES)} nodes)")

    # ── Build directed edges ──────────────────────────────────────────────────
    ids = list(city_lookup.keys())
    edges = set()   # (src, dst, distance_m)

    for i, src_id in enumerate(ids):
        src = city_lookup[src_id]
        for dst_id in ids:
            if src_id == dst_id:
                continue
            dst  = city_lookup[dst_id]
            dist = haversine_km(src["lat"], src["lon"], dst["lat"], dst["lon"])
            if should_add_edge(src, dst, dist):
                edges.add((src_id, dst_id, round(dist * 1000)))  # store in meters

    # Add global hub mesh
    add_hub_edges(city_lookup, edges)

    # ── Write distance.csv ───────────────────────────────────────────────────
    dist_path = PROJECT_ROOT / "datasets" / "distance.csv"
    dist_path.parent.mkdir(exist_ok=True)
    edge_list = sorted(edges)
    with open(dist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source","Destination","Distance(M)"])
        for src, dst, dm in edge_list:
            w.writerow([src, dst, dm])
    print(f"  OK distance.csv      ({len(edge_list)} directed edges)")

    # ── Write order_large.csv ────────────────────────────────────────────────
    order_path = PROJECT_ROOT / "datasets" / "order_large.csv"
    order_seq  = 0
    with open(order_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Order_ID","Material_ID","Item_ID","Source","Destination",
                    "Available_Time","Deadline","Danger_Type","Area","Weight"])
        for src_id, dst_id, _ in edge_list:
            src_industry = city_lookup[src_id]["industry"]
            rows, order_seq = _orders_for_edge(src_id, dst_id, src_industry, order_seq)
            for row in rows:
                w.writerow(row)
    print(f"  OK order_large.csv   ({order_seq} orders)")
    print(f"  Anomaly cities injected: {len(ANOMALY_CITIES)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    countries = len({c["country"] for c in city_lookup.values()})
    regions   = len({c["region"]  for c in city_lookup.values()})
    print(f"\n  Nodes    : {len(CITIES)}")
    print(f"  Edges    : {len(edge_list)}")
    print(f"  Countries: {countries}")
    print(f"  Regions  : {regions}")
    print(f"  Orders   : {order_seq}")
    print("\nDone. Run 'streamlit run app.py' to launch with the new dataset.")


if __name__ == "__main__":
    main()
