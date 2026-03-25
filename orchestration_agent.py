"""Execution orchestration agent for multi-stage recommendation pipelines.

This module adds a lightweight meta-agent that can schedule existing agents
with different strategies, instead of hardcoding one global fixed order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - keep lightweight fallback
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

PipelineProfile = Literal["standard", "user_stream", "minimal_refresh"]
VALID_STAGES = {"agent1", "agent2", "agent3_batch", "agent45_batch", "agent3_agent45_stream", "bundle"}


@dataclass
class OrchestrationPlan:
    profile: PipelineProfile
    stages: List[str]
    notes: List[str]


class QwenOrchestrationLLM:
    """Optional LLM planner for dynamic stage planning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers/torch are not available for orchestration LLM planning.")
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto" if torch is not None else None,
            device_map="auto",
        )

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
        for i, ch in enumerate(stripped):
            if ch != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(stripped, i)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def plan(self, profile: PipelineProfile, hints: Dict[str, bool]) -> Optional[OrchestrationPlan]:
        self._load()
        prompt = (
            "You are a pipeline orchestrator. Return exactly one JSON object with keys: "
            "profile (standard|user_stream|minimal_refresh), stages (string array), notes (string array). "
            f"Requested profile: {profile}. Runtime hints: {json.dumps(hints, ensure_ascii=False)}.\n"
            "Constraints: stages must end with bundle; include agent2 before any agent3* stage; "
            "for batch use agent3_batch + agent45_batch; for streaming use agent3_agent45_stream.\n"
            "No markdown."
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        raw = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        payload = self._extract_json_object(raw)
        if not payload:
            return None
        stages = [str(s) for s in payload.get("stages", []) if str(s) in VALID_STAGES]
        if not stages or "bundle" not in stages:
            return None
        return OrchestrationPlan(
            profile=str(payload.get("profile", profile)),
            stages=stages,
            notes=[str(x) for x in payload.get("notes", [])],
        )


class PipelineOrchestratorAgent:
    """A small planner agent that decides execution order from runtime goals.

    The planner intentionally outputs a constrained stage list so the existing
    pipeline implementation can safely execute the selected schedule.
    """

    def __init__(self, llm_planner: Optional[QwenOrchestrationLLM] = None) -> None:
        self.llm_planner = llm_planner

    def plan(
        self,
        profile: PipelineProfile = "standard",
        hints: Optional[Dict[str, bool]] = None,
    ) -> OrchestrationPlan:
        hints = hints or {}
        needs_fresh_item_profiles = bool(hints.get("needs_fresh_item_profiles", True))

        if self.llm_planner is not None:
            try:
                llm_plan = self.llm_planner.plan(profile=profile, hints=hints)
                if llm_plan is not None:
                    return llm_plan
            except Exception as exc:
                # Fallback to deterministic heuristic plan to keep pipeline robust.
                print(f"[PipelineOrchestratorAgent] LLM planning failed, fallback to heuristic plan: {exc}")

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
