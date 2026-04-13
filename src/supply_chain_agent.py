"""
supply_chain_agent.py
----------------------
Agentic AI loop for autonomous supply chain decision-making.

The agent uses Groq's tool-calling API to:
  1. Assess disruption scope (get_top_risks, query_node_risk)
  2. Identify what materials / goods are choked (get_material_risks)
  3. Check for pre-existing anomalies (get_anomaly_alerts)
  4. Find and validate alternate routes (find_alternate_route)
  5. Autonomously approve reroutes (approve_reroute)
  6. Flag critical suppliers for backup sourcing (flag_critical_supplier)
  7. Produce a structured CSCO action plan (finalize_action_plan)

Falls back to a deterministic simulation when no API key is available,
so the UI always works for demo purposes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

GEMINI_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")
GROQ_KEY_ENV_VARS = ("GROQ_API_KEY",)


def _resolve_env_key(env_names: tuple[str, ...]) -> str:
    for env_name in env_names:
        candidate = os.getenv(env_name, "").strip()
        if candidate and not candidate.lower().startswith("your_"):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class SupplyChainAgent:
    def __init__(
        self,
        G: nx.DiGraph,
        cascade_result: dict,
        risk_df: pd.DataFrame,
        reroute_suggestions: list,
        material_summary: pd.DataFrame,
        anomaly_df: pd.DataFrame,
        disruption_info: dict,
        api_key: str | None = None,
    ):
        self.G = G
        self.cascade_result = cascade_result
        self.risk_df = risk_df
        self.reroute_suggestions = reroute_suggestions
        self.material_summary = material_summary
        self.anomaly_df = anomaly_df
        self.disruption_info = disruption_info
        self.api_key = api_key or _resolve_env_key(GROQ_KEY_ENV_VARS)

        # Mutable state updated by tool calls
        self.action_log: list[dict] = []
        self.approved_reroutes: list[dict] = []
        self.flagged_nodes: list[dict] = []
        self.final_plan: dict | None = None

    # ------------------------------------------------------------------
    # Tool implementations  (pure Python — no LLM needed)
    # ------------------------------------------------------------------

    def _get_top_risks(self) -> dict:
        if self.risk_df.empty:
            return {"total_affected": 0, "critical_count": 0, "nodes": []}
        top5 = self.risk_df.head(5)
        return {
            "total_affected": len(self.risk_df),
            "critical_count": int((self.risk_df["risk_score"] >= 0.65).sum()),
            "countries_affected": int(self.risk_df["country"].nunique()),
            "nodes": [
                {
                    "node_id":          r.get("node", "?"),
                    "city":             r.get("city_name", "?"),
                    "country":          r.get("country", "?"),
                    "product_category": r.get("product_category", "General"),
                    "tier":             int(r.get("tier", 3)),
                    "risk_score":       round(float(r.get("risk_score", 0)), 4),
                    "risk_level":       r.get("risk_level", "Unknown"),
                    "cascade_depth":    int(r.get("cascade_depth", 0)),
                }
                for _, r in top5.iterrows()
            ],
        }

    def _query_node_risk(self, node_id: str) -> dict:
        if self.risk_df.empty:
            return {"error": "No risk data available"}
        row = self.risk_df[self.risk_df["node"] == node_id]
        if row.empty:
            return {"error": f"Node {node_id} not in risk table"}
        r = row.iloc[0]
        return {
            "node_id":           node_id,
            "city_name":         r.get("city_name", "?"),
            "country":           r.get("country", "?"),
            "product_category":  r.get("product_category", "General"),
            "tier":              int(r.get("tier", 3)),
            "risk_score":        round(float(r.get("risk_score", 0)), 4),
            "risk_level":        r.get("risk_level", "Unknown"),
            "cascade_depth":     int(r.get("cascade_depth", 0)),
            "delay_probability": round(float(r.get("delay_prob", r.get("delay_probability", 0.5))), 4),
            "out_degree":        self.G.out_degree(node_id),
            "in_degree":         self.G.in_degree(node_id),
        }

    def _get_material_risks(self) -> dict:
        if self.material_summary.empty:
            return {"total_material_types": 0, "materials": []}
        top5 = self.material_summary.head(5)
        return {
            "total_material_types": len(self.material_summary),
            "total_disrupted_routes": int(self.material_summary["Disrupted Routes"].sum()),
            "materials": [
                {
                    "material":           r["Material Flow"],
                    "disrupted_routes":   int(r["Disrupted Routes"]),
                    "countries_affected": int(r["Countries Affected"]),
                    "example_route":      r["Example Route"],
                    "specific_items":     r["Specific Items"][:120],
                }
                for _, r in top5.iterrows()
            ],
        }

    def _get_anomaly_alerts(self) -> dict:
        if self.anomaly_df.empty:
            return {"total_anomalous": 0, "overlap_with_disruption": 0, "alerts": []}
        anomalous     = self.anomaly_df[self.anomaly_df["is_anomalous"]]
        cascade_nodes = set(self.cascade_result.keys())
        overlap       = anomalous[anomalous["node"].isin(cascade_nodes)]
        return {
            "total_anomalous":           len(anomalous),
            "overlap_with_disruption":   len(overlap),
            "alerts": [
                {
                    "node_id": r["node"],
                    "city":    r["city_name"],
                    "level":   r["anomaly_level"],
                    "score":   round(float(r["anomaly_score"]), 4),
                }
                for _, r in overlap.head(4).iterrows()
            ],
        }

    def _find_alternate_route(self, source: str, destination: str) -> dict:
        # Try exact match first
        for r in self.reroute_suggestions:
            if (r["source"] == source and r["destination"] == destination) or \
               (r["source_name"] == source and r["destination_name"] == destination):
                return {
                    "status":             r["status"],
                    "source":             r["source_name"],
                    "destination":        r["destination_name"],
                    "original_dist_km":   r.get("original_dist_km"),
                    "alternate_dist_km":  r.get("alternate_dist_km"),
                    "distance_delta_km":  r.get("distance_delta_km"),
                    "detour_pct":         r.get("detour_pct"),
                    "hops_alternate":     r.get("hops_alternate"),
                }
        # Best available fallback
        found = [r for r in self.reroute_suggestions if r["status"] == "✅ Alternate Found"]
        if found:
            best = found[0]
            return {
                "status":            "✅ Alternate Found (closest match)",
                "source":            best["source_name"],
                "destination":       best["destination_name"],
                "distance_delta_km": best.get("distance_delta_km"),
                "detour_pct":        best.get("detour_pct"),
                "note":              "Exact pair not found; returning best available alternate",
            }
        return {"status": "⚠️ No alternate route found", "source": source, "destination": destination}

    def _approve_reroute(self, source: str, destination: str, reason: str) -> dict:
        self.approved_reroutes.append({
            "source":            source,
            "destination":       destination,
            "reason":            reason,
            "approved_at_step":  len(self.action_log) + 1,
        })
        return {"status": "approved", "message": f"Reroute {source} → {destination} approved"}

    def _flag_critical_supplier(self, node_id: str, reason: str, priority: str = "high") -> dict:
        city = node_id
        if not self.risk_df.empty:
            row = self.risk_df[self.risk_df["node"] == node_id]
            if not row.empty:
                city = row.iloc[0]["city_name"]
        self.flagged_nodes.append({
            "node_id":  node_id,
            "city":     city,
            "reason":   reason,
            "priority": priority,
        })
        return {"status": "flagged", "node_id": node_id, "city": city}

    def _finalize_action_plan(
        self,
        summary: str,
        priority_actions: list[str],
        estimated_recovery_days: int,
        risk_level: str,
    ) -> dict:
        self.final_plan = {
            "summary":                  summary,
            "priority_actions":         priority_actions,
            "estimated_recovery_days":  estimated_recovery_days,
            "risk_level":               risk_level,
            "approved_reroutes":        self.approved_reroutes,
            "flagged_nodes":            self.flagged_nodes,
        }
        return {"status": "plan_finalized"}

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, name: str, args: dict) -> dict:
        dispatch_map = {
            "get_top_risks":          lambda: self._get_top_risks(),
            "query_node_risk":        lambda: self._query_node_risk(**args),
            "get_material_risks":     lambda: self._get_material_risks(),
            "get_anomaly_alerts":     lambda: self._get_anomaly_alerts(),
            "find_alternate_route":   lambda: self._find_alternate_route(**args),
            "approve_reroute":        lambda: self._approve_reroute(**args),
            "flag_critical_supplier": lambda: self._flag_critical_supplier(**args),
            "finalize_action_plan":   lambda: self._finalize_action_plan(**args),
        }
        fn = dispatch_map.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        return fn()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, max_turns: int = 12) -> dict:
        gemini_key = _resolve_env_key(GEMINI_KEY_ENV_VARS)
        groq_key = self.api_key

        # Try Groq tool-calling agent first
        if groq_key and groq_key.strip() not in ("", "your_groq_api_key_here"):
            try:
                return self._run_groq(max_turns)
            except Exception as exc:
                print(f"  [agent] Groq agent error: {exc} — trying deterministic fallback")

        # Run deterministic tools + optionally enrich with a single Gemini reasoning call.
        source_tag = "gemini-enhanced" if (gemini_key and gemini_key.strip()) else "groq-unavailable"
        result = self._run_fallback(source_tag=source_tag)
        if gemini_key and gemini_key.strip():
            self._enrich_with_gemini(result, gemini_key)
        return result

    # ------------------------------------------------------------------
    # Gemini single-call reasoning enrichment
    # ------------------------------------------------------------------

    def _enrich_with_gemini(self, result: dict, gemini_key: str) -> None:
        """
        Make ONE Gemini call to generate human-readable reasoning for each
        tool step already executed by the deterministic agent.
        Replaces generic thought strings with Gemini-generated analysis.
        """
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)

            event    = self.disruption_info.get("event_text", "Unknown disruption")
            severity = self.disruption_info.get("severity", "medium")

            # Summarise what the agent found
            tool_summary = "\n".join([
                f"Step {e['step']}: {e['tool']}() → {str(e['result'])[:200]}"
                for e in result["action_log"] if e.get("tool")
            ])

            prompt = (
                f"You are an autonomous supply chain AI agent that just completed an analysis.\n"
                f"Disruption: {event} (Severity: {severity.upper()})\n\n"
                f"Tools executed and results:\n{tool_summary}\n\n"
                f"For each tool call step, write ONE sentence of reasoning that a CSCO "
                f"(Chief Supply Chain Officer) would say BEFORE calling that tool — "
                f"explaining WHY they are calling it and what they expect to learn.\n"
                f"Return a JSON array of strings, one per step (in order):\n"
                f'["<reasoning for step 1>", "<reasoning for step 2>", ...]'
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            import json, re
            raw = response.text.strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                thoughts = json.loads(match.group(0))
                tool_steps = [e for e in result["action_log"] if e.get("tool")]
                for i, step_entry in enumerate(tool_steps):
                    if i < len(thoughts):
                        step_entry["thought"] = thoughts[i]
                result["source"] = "gemini-enhanced"
                print(f"  [agent] Gemini reasoning enrichment applied ({len(thoughts)} thoughts)")
        except Exception as e:
            print(f"  [agent] Gemini enrichment failed: {e} — keeping deterministic thoughts")

    # ------------------------------------------------------------------
    # Groq tool-calling loop
    # ------------------------------------------------------------------

    def _run_groq(self, max_turns: int) -> dict:
        import json
        from groq import Groq

        client = Groq(api_key=self.api_key)

        event      = self.disruption_info.get("event_text", "Unknown disruption")
        severity   = self.disruption_info.get("severity", "medium")
        n_affected = len(self.cascade_result)

        system_prompt = (
            f"You are an autonomous supply chain CSCO (Chief Supply Chain Officer) AI agent.\n\n"
            f"A disruption has just been detected:\n"
            f"  Event    : {event}\n"
            f"  Severity : {severity.upper()}\n"
            f"  Nodes at risk: {n_affected}\n\n"
            f"Your task is to independently assess the situation and take action:\n"
            f"  1. Call get_top_risks to understand scope\n"
            f"  2. Call get_material_risks to identify what goods are disrupted\n"
            f"  3. Call get_anomaly_alerts to check pre-existing vulnerabilities\n"
            f"  4. Use query_node_risk to inspect the most critical nodes\n"
            f"  5. Use find_alternate_route and approve_reroute for viable paths\n"
            f"  6. Use flag_critical_supplier for nodes needing immediate attention\n"
            f"  7. Call finalize_action_plan with your conclusions\n\n"
            f"Be decisive. Use data from tools before making decisions. "
            f"Cite specific node names, risk scores, and materials in your plan."
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_top_risks",
                    "description": "Get the top 5 highest-risk supply chain nodes with risk scores and product categories.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_node_risk",
                    "description": "Query detailed risk information for a specific node by its ID (e.g. 'City_1').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string", "description": "Node ID, e.g. City_1"},
                        },
                        "required": ["node_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_material_risks",
                    "description": "Get the top disrupted material flows — which goods and commodities are choked.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_anomaly_alerts",
                    "description": "Check for supply chain nodes showing anomalous shipment patterns.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_alternate_route",
                    "description": "Find an alternate supply route that avoids disrupted nodes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source":      {"type": "string", "description": "Origin city name or node ID"},
                            "destination": {"type": "string", "description": "Destination city name or node ID"},
                        },
                        "required": ["source", "destination"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "approve_reroute",
                    "description": "Approve an alternate supply route for immediate activation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source":      {"type": "string"},
                            "destination": {"type": "string"},
                            "reason":      {"type": "string", "description": "Business justification"},
                        },
                        "required": ["source", "destination", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "flag_critical_supplier",
                    "description": "Flag a supply chain node as critical — requiring immediate backup sourcing.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id":  {"type": "string"},
                            "reason":   {"type": "string"},
                            "priority": {"type": "string", "description": "high | medium | low"},
                        },
                        "required": ["node_id", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_action_plan",
                    "description": "Create the final structured CSCO action plan.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "2-3 sentence executive summary citing specific nodes and materials",
                            },
                            "priority_actions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "4-6 specific, actionable decisions the agent has made",
                            },
                            "estimated_recovery_days": {
                                "type": "integer",
                                "description": "Estimated days to full supply chain recovery",
                            },
                            "risk_level": {
                                "type": "string",
                                "description": "Critical | High | Medium | Low",
                            },
                        },
                        "required": ["summary", "priority_actions", "estimated_recovery_days", "risk_level"],
                    },
                },
            },
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "Begin your autonomous assessment now."},
        ]

        start = time.time()
        step  = 0

        while step < max_turns:
            step += 1

            req = {
                "model": "openai/gpt-oss-120b",
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.2,
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

            msg = response.choices[0].message
            messages.append(msg)

            thought_text   = msg.content or ""
            tool_calls_raw = msg.tool_calls or []

            if not tool_calls_raw:
                # Model stopped calling tools
                if thought_text:
                    self.action_log.append({
                        "step":    step,
                        "type":    "conclusion",
                        "thought": thought_text,
                        "tool":    None,
                        "args":    None,
                        "result":  None,
                    })
                break

            # Execute all tool calls in this turn
            for tc in tool_calls_raw:
                fn_obj = getattr(tc, "function", None)
                tool_name = getattr(fn_obj, "name", "") if fn_obj is not None else ""
                raw_args = getattr(fn_obj, "arguments", "{}") if fn_obj is not None else "{}"

                if not tool_name:
                    continue

                try:
                    tool_args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    tool_args = {}
                if not isinstance(tool_args, dict):
                    tool_args = {}

                print(f"  [agent] Step {step}: {tool_name}({list(tool_args.keys())})")
                tool_result = self._dispatch(tool_name, tool_args)

                self.action_log.append({
                    "step":    step,
                    "type":    "tool_call",
                    "thought": thought_text,
                    "tool":    tool_name,
                    "args":    tool_args,
                    "result":  tool_result,
                })
                thought_text = ""  # only attach to first call in a turn

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(tool_result),
                })

                if tool_name == "finalize_action_plan" and self.final_plan:
                    break

            if self.final_plan:
                break

        elapsed = round(time.time() - start, 1)
        return self._build_result(elapsed, source="groq-agent")

    # ------------------------------------------------------------------
    # Deterministic fallback agent (no API key needed)
    # ------------------------------------------------------------------

    def _run_fallback(self, source_tag: str = "template-agent") -> dict:
        self._fallback_source = source_tag
        start = time.time()

        # Step 1 — assess top risks
        top_risks = self._get_top_risks()
        self.action_log.append({
            "step": 1, "type": "tool_call",
            "thought": (
                "I'll start by assessing the full scope of this disruption — "
                "how many nodes are affected and which ones carry the highest risk?"
            ),
            "tool": "get_top_risks", "args": {}, "result": top_risks,
        })

        # Step 2 — identify disrupted materials
        mat_risks = self._get_material_risks()
        self.action_log.append({
            "step": 2, "type": "tool_call",
            "thought": (
                "Now I need to understand what physical goods are choked. "
                "Which materials and commodities flow through the disrupted routes?"
            ),
            "tool": "get_material_risks", "args": {}, "result": mat_risks,
        })

        # Step 3 — check anomaly pre-signals
        anomalies = self._get_anomaly_alerts()
        self.action_log.append({
            "step": 3, "type": "tool_call",
            "thought": (
                "Before acting I want to check if any of these nodes were already "
                "showing anomalous patterns. Compound risk is the real danger here."
            ),
            "tool": "get_anomaly_alerts", "args": {}, "result": anomalies,
        })

        # Step 4 — deep-inspect top node, then flag it
        if top_risks.get("nodes"):
            top_node = top_risks["nodes"][0]
            node_detail = self._query_node_risk(top_node["node_id"])
            self.action_log.append({
                "step": 4, "type": "tool_call",
                "thought": (
                    f"The highest-risk node is {top_node['city']} "
                    f"(score {top_node['risk_score']:.3f}, Tier {top_node['tier']}). "
                    f"Let me inspect it in detail before deciding on action."
                ),
                "tool": "query_node_risk",
                "args": {"node_id": top_node["node_id"]},
                "result": node_detail,
            })

            flag_result = self._flag_critical_supplier(
                top_node["node_id"],
                f"Highest-risk node in cascade (score {top_node['risk_score']:.3f}, Tier {top_node['tier']}) — "
                f"requires immediate backup sourcing for {top_node['product_category']}",
                "high",
            )
            self.action_log.append({
                "step": 4, "type": "tool_call",
                "thought": (
                    f"{top_node['city']} is a critical chokepoint. "
                    f"I'm flagging it for immediate escalation to backup suppliers."
                ),
                "tool": "flag_critical_supplier",
                "args": {
                    "node_id":  top_node["node_id"],
                    "reason":   flag_result.get("city", top_node["city"]),
                    "priority": "high",
                },
                "result": flag_result,
            })

        # Step 5 — approve top reroutes
        found_routes = [r for r in self.reroute_suggestions if r["status"] == "✅ Alternate Found"]
        for route in found_routes[:3]:
            approve_result = self._approve_reroute(
                route["source_name"],
                route["destination_name"],
                f"Alternate path is operationally viable — adds only "
                f"+{route.get('detour_pct', 0):.1f}% distance ({route.get('distance_delta_km', 0):+.0f} km)",
            )
            self.action_log.append({
                "step": 5, "type": "tool_call",
                "thought": (
                    f"Route {route['source_name']} → {route['destination_name']} has a viable alternate "
                    f"(+{route.get('detour_pct', 0):.1f}% distance). I approve it for immediate activation."
                ),
                "tool": "approve_reroute",
                "args": {
                    "source":      route["source_name"],
                    "destination": route["destination_name"],
                    "reason":      approve_result.get("message", ""),
                },
                "result": approve_result,
            })

        # Step 6 — finalise plan
        sev     = self.disruption_info.get("severity", "medium").upper()
        n_nodes = len(self.cascade_result)
        event   = self.disruption_info.get("event_text", "disruption event")
        n_countries = self.risk_df["country"].nunique() if not self.risk_df.empty else 0

        recovery_days = {"HIGH": 14, "MEDIUM": 7, "LOW": 3}.get(sev, 7)
        risk_level    = {"HIGH": "Critical", "MEDIUM": "High", "LOW": "Medium"}.get(sev, "High")

        top_mats = [m["material"] for m in mat_risks.get("materials", [])[:2]]
        mat_str  = " & ".join(top_mats) if top_mats else "key materials"

        priority_actions = [
            f"Activate emergency procurement for {mat_str} — "
            f"{mat_risks.get('total_disrupted_routes', 0)} supply routes currently disrupted",
            f"Deploy {len(self.approved_reroutes)} approved alternate route(s) immediately to maintain throughput",
            f"Flag {len(self.flagged_nodes)} critical supplier(s) for backup sourcing within 24 hours",
            f"Issue risk advisory to procurement teams in {n_countries} affected countries",
            f"Escalate to Tier-1 suppliers in unaffected regions to pre-emptively increase buffer stock",
        ]
        if anomalies.get("overlap_with_disruption", 0) > 0:
            priority_actions.append(
                f"URGENT: {anomalies['overlap_with_disruption']} disrupted node(s) also show anomalous patterns — "
                f"investigate for compounding risk before re-routing through them"
            )

        top_city = (
            top_risks["nodes"][0]["city"]
            if top_risks.get("nodes") else "affected nodes"
        )
        summary = (
            f"A {sev.lower()}-severity event ({event}) has disrupted {n_nodes} supply chain nodes across "
            f"{n_countries} countries, with {top_city} as the highest-risk chokepoint. "
            f"The agent has assessed {mat_risks.get('total_material_types', 0)} disrupted material flows, "
            f"approved {len(self.approved_reroutes)} alternate route(s), and flagged "
            f"{len(self.flagged_nodes)} critical supplier(s) for immediate action. "
            f"Estimated recovery: {recovery_days} days."
        )

        finalize_result = self._finalize_action_plan(
            summary                  = summary,
            priority_actions         = priority_actions,
            estimated_recovery_days  = recovery_days,
            risk_level               = risk_level,
        )
        self.action_log.append({
            "step": 6, "type": "tool_call",
            "thought": (
                "I have enough data to write the final action plan. "
                "Reroutes are approved, critical suppliers are flagged, and the situation is fully assessed. "
                "Committing the plan now."
            ),
            "tool": "finalize_action_plan",
            "args": {"summary": summary, "priority_actions": priority_actions,
                     "estimated_recovery_days": recovery_days, "risk_level": risk_level},
            "result": finalize_result,
        })

        elapsed = round(time.time() - start, 1)
        return self._build_result(elapsed, source=getattr(self, "_fallback_source", "template-agent"))

    # ------------------------------------------------------------------
    # Build final result dict
    # ------------------------------------------------------------------

    def _build_result(self, elapsed: float, source: str) -> dict:
        return {
            "action_log":       self.action_log,
            "final_plan":       self.final_plan or {
                "summary":                 "Analysis completed — see action log for details.",
                "priority_actions":        ["Review the action log above for full findings."],
                "estimated_recovery_days": 7,
                "risk_level":              "Medium",
                "approved_reroutes":       self.approved_reroutes,
                "flagged_nodes":           self.flagged_nodes,
            },
            "approved_reroutes": self.approved_reroutes,
            "flagged_nodes":     self.flagged_nodes,
            "elapsed_seconds":   elapsed,
            "source":            source,
            "steps_taken":       sum(1 for e in self.action_log if e["type"] == "tool_call"),
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_agent(
    G:                   nx.DiGraph,
    cascade_result:      dict,
    risk_df:             pd.DataFrame,
    reroute_suggestions: list,
    material_summary:    pd.DataFrame,
    anomaly_df:          pd.DataFrame,
    disruption_info:     dict,
    api_key:             str | None = None,
    max_turns:           int = 12,
) -> dict:
    agent = SupplyChainAgent(
        G                   = G,
        cascade_result      = cascade_result,
        risk_df             = risk_df,
        reroute_suggestions = reroute_suggestions,
        material_summary    = material_summary,
        anomaly_df          = anomaly_df,
        disruption_info     = disruption_info,
        api_key             = api_key,
    )
    return agent.run(max_turns=max_turns)
