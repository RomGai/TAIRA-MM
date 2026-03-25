"""Execution orchestration agent for multi-stage recommendation pipelines.

This module adds a lightweight meta-agent that can schedule existing agents
with different strategies, instead of hardcoding one global fixed order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

PipelineProfile = Literal["standard", "user_stream", "minimal_refresh"]


@dataclass
class OrchestrationPlan:
    profile: PipelineProfile
    stages: List[str]
    notes: List[str]


class PipelineOrchestratorAgent:
    """A small planner agent that decides execution order from runtime goals.

    The planner intentionally outputs a constrained stage list so the existing
    pipeline implementation can safely execute the selected schedule.
    """

    def plan(
        self,
        profile: PipelineProfile = "standard",
        hints: Optional[Dict[str, bool]] = None,
    ) -> OrchestrationPlan:
        hints = hints or {}
        needs_fresh_item_profiles = bool(hints.get("needs_fresh_item_profiles", True))

        if profile == "user_stream":
            return OrchestrationPlan(
                profile=profile,
                stages=["agent1", "agent2", "agent3_agent45_stream", "bundle"],
                notes=[
                    "Run Agent3 + Agent4/5 immediately per user to avoid extra artifact scans.",
                    "Useful for goal-dependent user-level scheduling.",
                ],
            )

        if profile == "minimal_refresh":
            stages = ["agent2", "agent3_batch", "agent45_batch", "bundle"]
            notes = [
                "Skip Agent1 when global item DB is already warm to reduce redundant work.",
            ]
            if needs_fresh_item_profiles:
                stages.insert(0, "agent1")
                notes.append("Detected fresh-item requirement, Agent1 was re-enabled.")
            return OrchestrationPlan(profile=profile, stages=stages, notes=notes)

        return OrchestrationPlan(
            profile="standard",
            stages=["agent1", "agent2", "agent3_batch", "agent45_batch", "bundle"],
            notes=["Default fixed-order execution for maximum backward compatibility."],
        )
