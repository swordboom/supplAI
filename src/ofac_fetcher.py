import json
import os

def generate_ofac_sanctions(output_path="data/ofac_sanctions.json"):
    """
    Simulates fetching and parsing the US Treasury OFAC Specially Designated Nationals (SDN) 
    and Comprehensive Sanctions lists. 
    
    For prototype stability, this returns a deterministic list of countries under 
    heavy comprehensive US sanctions which would strictly prohibit logistics routing.
    """
    
    # Countries under broad comprehensive US embargoes / high OFAC risk
    sanctioned_nations = [
        "Russia",
        "Iran",
        "North Korea",
        "Syria",
        "Cuba",
        "Belarus",
        "Myanmar"
    ]
    
    # Structuring similar to an API response
    payload = {
        "source": "US_Treasury_OFAC_Simulation",
        "timestamp": "Real-time",
        "sanctioned_countries": sanctioned_nations,
        "sanctioned_cities": ["Sevastopol", "Donetsk", "Luhansk"]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    print(f"✅ OFAC Sanctions List successfully generated at {output_path}")
    return payload

if __name__ == "__main__":
    generate_ofac_sanctions()
