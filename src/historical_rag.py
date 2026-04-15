"""
historical_rag.py
------------------
Lightweight Retrieval-Augmented Generation (RAG) module using scikit-learn.
Simulates a corporate memory database of historical supply chain crises.
"""

from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# A mock database of historical supply chain crises.
# In a full enterprise application, this would be a Vector Database (like Chroma or FAISS)
# containing hundreds of post-mortem PDFs.
HISTORICAL_DB = [
    {
        "event_id": "EVT-2021-SUEZ",
        "title": "2021 Suez Canal Blockage",
        "description": "A massive shipping container ship got stuck in the Suez Canal, completely halting global trade traffic between Europe and Asia for six days. Massive port congestion followed.",
        "lessons_learned": "Relying on a single chokepoint without alternative continent-spanning rail or air freight options leads to catastrophic delays. Future mitigations require pre-booking air cargo space during maritime anomalies.",
        "mitigation_success": "High when diverted via Cape of Good Hope, though added 14 days to transit."
    },
    {
        "event_id": "EVT-2020-CHIP",
        "title": "2020-2022 Global Semiconductor Shortage",
        "description": "COVID-19 lockdowns in Asia caused factory shutdowns in the electronics and semiconductor industry. The disruption cascaded globally, particularly hitting automotive manufacturing.",
        "lessons_learned": "Just-in-time (JIT) manufacturing is highly vulnerable to Tier 2/Tier 3 supplier blackouts. Companies that warehoused strategic 6-month buffer stocks survived the disruption.",
        "mitigation_success": "Medium when using multi-sourcing and redesigned components, low when waiting on legacy nodes."
    },
    {
        "event_id": "EVT-2011-THAI",
        "title": "2011 Thailand Floods",
        "description": "Severe flooding in Thailand inundated major industrial estates. Heavy impact was felt in the hard disk drive (HDD) and automotive parts supply chains globally.",
        "lessons_learned": "Geographic concentration of critical component manufacturing creates massive systemic risk. Supplier mapping to Tier 3 is essential to identify physical location clusters.",
        "mitigation_success": "Low initially; took over 6 months to shift manufacturing to Malaysia and China."
    },
    {
        "event_id": "EVT-2021-TEXAS",
        "title": "2021 Texas Power Freeze",
        "description": "A severe winter storm caused massive power grid failures in Texas, USA, forcing petrochemical and plastics refinement plants to shut down for weeks.",
        "lessons_learned": "Climate anomalies can take down critical domestic infrastructure. Winterization of facilities and maintaining geographic diversity in chemical supply bases is crucial.",
        "mitigation_success": "Medium. Disrupted supply was slowly supplemented through European imports at high premium costs."
    },
    {
        "event_id": "EVT-2024-REDSEA",
        "title": "2023-2024 Red Sea Shipping Crisis",
        "description": "Geopolitical attacks on shipping vessels in the Red Sea forced major operators to suspend passage through the Bab-el-Mandeb strait, driving up freight rates and insurance.",
        "lessons_learned": "Geopolitical security flashpoints require immediate activation of near-shoring options or land-bridge multimodal transport from the Middle East to Europe.",
        "mitigation_success": "High but costly; rerouting entirely around Africa added $1M+ fuel cost per voyage."
    }
]

def retrieve_historical_context(query: str) -> Optional[Dict[str, Any]]:
    """
    Given a disruption description query, returns the most relevant 
    historical event from the database using TF-IDF and Cosine Similarity.
    """
    if not query or not str(query).strip():
        return None
        
    # We match against the Title + Description for better context handling
    match_texts = [item["title"] + " " + item["description"] for item in HISTORICAL_DB]
    
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Fit and transform the descriptions
        tfidf_matrix = vectorizer.fit_transform(match_texts)
        # Transform the query
        query_vec = vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Find the index of the highest similarity
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        # threshold for relevance
        if best_score > 0.05:
             match = HISTORICAL_DB[best_match_idx].copy()
             match["similarity_score"] = float(best_score)
             return match
             
    except Exception as e:
        print(f"[RAG Error] {e}")
        
    return None

if __name__ == "__main__":
    # Small test
    query = "Factory shutdown in China affecting electronics supply chain"
    res = retrieve_historical_context(query)
    print(f"Query: {query}")
    if res:
        print(f"Match: {res['title']} (Score: {res['similarity_score']:.2f})")
    else:
        print("No match found.")
