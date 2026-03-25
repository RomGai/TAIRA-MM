"""Automate the 5-agent Amazon pipeline and bundle generated artifacts.

Execution order:
1) item_profiler_agents.py (Agent 1 + Agent 2)
2) intent_dual_recall_agent.py (Agent 3)
3) dynamic_reasoning_ranking_agent.py (Agent 4 + Agent 5)

Key behavior requested by product usage:
- Agent 1 processes all items (no sample_k / user sampling).
- Agent 4+5 automatically consume all Agent-3 output files.
- Final artifacts are bundled into a caller-specified path.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import zipfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from item_profiler_agents import (
    HistoryItemProfileInput,
    ItemProfileInput,
    _build_user_item_timestamp_map,
    _export_sqlite_table_as_jsonl,
    _write_jsonl,
    bootstrap_agents_from_processed,
    expand_pos_neg_rows,
    load_item_desc_tsv,
)
from orchestration_agent import PipelineOrchestratorAgent, QwenOrchestrationLLM


def _collect_all_labeled_history_rows(
    user_pairs_tsv_path: str | Path,
    user_items_negs_path: str | Path,
    include_negative: bool = True,
) -> List[Dict[str, Any]]:
    """Collect all users' labeled rows (positive + negative) with ordering.

    Ordering keeps each user's sequence deterministic:
    1) positive rows by timestamp asc
    2) negative rows after positives (timestamp may be missing)
    """
    ts_map = _build_user_item_timestamp_map(user_pairs_tsv_path)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in expand_pos_neg_rows(user_items_negs_path):
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        behavior = str(row["behavior"])
        if not include_negative and behavior == "negative":
            continue
        ts = ts_map.get((user_id, item_id))

        if behavior == "positive" and ts is None:
            # Skip impossible positive labels without real timestamp.
            continue

        grouped.setdefault(user_id, []).append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "behavior": behavior,
                "timestamp": ts,
            }
        )

    merged: List[Dict[str, Any]] = []
    for user_id in grouped.keys():
        seq = grouped[user_id]
        seq.sort(
            key=lambda r: (
                0 if r["behavior"] == "positive" else 1,
                int(r["timestamp"]) if r["timestamp"] is not None else 10**30,
                str(r["item_id"]),
            )
        )
        merged.extend(seq)

    return merged


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _bundle_paths(bundle_path: str | Path, include_paths: List[Path]) -> Path:
    bundle_file = _ensure_parent(bundle_path)
    with zipfile.ZipFile(bundle_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in include_paths:
            p = p.resolve()
            if not p.exists():
                continue
            if p.is_file():
                zf.write(p, arcname=p.name)
                continue
            for child in p.rglob("*"):
                if child.is_file():
                    zf.write(child, arcname=str(p.name / child.relative_to(p)))
    return bundle_file


def _list_saved_agent3_outputs(output_dir: str | Path) -> List[Path]:
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("*_intent_dual_recall_output.json"))




def _progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    filled = int(width * current / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _build_user_sample_progress(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    per_user_total: Dict[str, int] = defaultdict(int)
    for r in rows:
        per_user_total[str(r.get("user_id", ""))] += 1

    return {u: {"done": 0, "total": t} for u, t in per_user_total.items()}




def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    from intent_dual_recall_agent import GlobalHistoryAccessor, Qwen3RouterLLM, RoutingRecallAgent
    from dynamic_reasoning_ranking_agent import run_module3

    item_map = load_item_desc_tsv(args.item_desc_tsv)
    fallback_item_map: Dict[str, Dict[str, str]] = {}
    fallback_item_desc_tsv = str(getattr(args, "agent2_item_desc_tsv", "") or "").strip()
    if fallback_item_desc_tsv:
        fallback_item_map = load_item_desc_tsv(fallback_item_desc_tsv)

    candidate_profiler, history_profiler = bootstrap_agents_from_processed(
        item_desc_tsv=args.item_desc_tsv,
        global_db_path=args.global_db,
        history_db_path=args.history_db,
        model_name=args.vl_model,
    )

    run_out_dir = Path(args.profiler_run_out_dir)
    run_out_dir.mkdir(parents=True, exist_ok=True)

    candidate_meta_records: List[Dict[str, Any]] = []
    candidate_profile_records: List[Dict[str, Any]] = []
    history_meta_records: List[Dict[str, Any]] = []
    history_profile_records: List[Dict[str, Any]] = []
    all_item_ids = list(item_map.keys())
    all_history_rows: List[Dict[str, Any]] = []
    all_user_ids: List[str] = []
    intent_outputs: List[Path] = []

    def _run_agent1() -> None:
        print(f"[Agent 1] Processing all items: {len(all_item_ids)}")
        for item_idx, item_id in enumerate(all_item_ids, start=1):
            meta = item_map[item_id]
            item = ItemProfileInput(
                item_id=item_id,
                title=f"item_{item_id}",
                detail_text=meta.get("summary", "") or "",
                main_image=meta.get("image", ""),
                detail_images=[],
                category_hint=args.category_hint,
            )
            candidate_meta_records.append(asdict(item))

            profile = candidate_profiler.global_db.get_profile(item_id)
            profile_source = "global_db_reused"
            if profile is None:
                profile = candidate_profiler.profile_and_store(item)
                profile_source = "newly_profiled"

            candidate_profile_records.append(
                {
                    "item_id": item_id,
                    "source": "candidate",
                    "profile_source": profile_source,
                    "profile": profile,
                }
            )
            print(
                f"[Agent 1][Item Progress] {item_idx}/{len(all_item_ids)} "
                f"{_progress_bar(item_idx, len(all_item_ids))} item_id={item_id} source={profile_source}"
            )

    def _run_agent2() -> None:
        nonlocal all_history_rows
        all_history_rows = _collect_all_labeled_history_rows(
            user_pairs_tsv_path=args.user_pairs_tsv,
            user_items_negs_path=args.user_items_negs_tsv,
            include_negative=not bool(getattr(args, "positive_history_only", False)),
        )
        print(f"[Agent 2] Processing all labeled history rows: {len(all_history_rows)}")
        user_sample_progress = _build_user_sample_progress(all_history_rows)
        total_users_with_history = len(user_sample_progress)
        processed_user_ids: set[str] = set()

        for row_idx, row in enumerate(all_history_rows, start=1):
            user_id = str(row["user_id"])
            processed_user_ids.add(user_id)
            item_id = str(row["item_id"])
            timestamp_raw = row.get("timestamp")
            timestamp: Optional[int] = None if timestamp_raw is None else int(timestamp_raw)
            meta = item_map.get(item_id) or fallback_item_map.get(item_id, {"image": "", "summary": ""})

            hist_item = HistoryItemProfileInput(
                item_id=item_id,
                title=f"item_{item_id}",
                detail_text=meta.get("summary", "") or "",
                main_image=meta.get("image", ""),
                detail_images=[],
                category_hint=args.category_hint,
                user_id=user_id,
                behavior=str(row["behavior"]),
                timestamp=timestamp,
            )
            history_meta_records.append(asdict(hist_item))

            profile = candidate_profiler.global_db.get_profile(item_id)
            profile_source = "global_db_reused"
            if profile is None:
                profile = candidate_profiler.profile_and_store(
                    ItemProfileInput(
                        item_id=item_id,
                        title=hist_item.title,
                        detail_text=hist_item.detail_text,
                        main_image=hist_item.main_image,
                        detail_images=hist_item.detail_images,
                        category_hint=hist_item.category_hint,
                    )
                )
                profile_source = "newly_profiled"

            profile["behavior"] = hist_item.behavior
            profile["timestamp"] = hist_item.timestamp
            profile["user_id"] = hist_item.user_id

            if not history_profiler.history_db.exists(
                user_id=hist_item.user_id,
                item_id=hist_item.item_id,
                behavior=hist_item.behavior,
                timestamp=hist_item.timestamp,
            ):
                history_profiler.history_db.insert(
                    user_id=hist_item.user_id,
                    item_id=hist_item.item_id,
                    behavior=hist_item.behavior,
                    timestamp=hist_item.timestamp,
                    profile=profile,
                )
                history_write_status = "inserted"
            else:
                history_write_status = "already_exists"

            history_profile_records.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "source": "history",
                    "profile_source": profile_source,
                    "history_write_status": history_write_status,
                    "profile": profile,
                }
            )
            user_sample_progress[user_id]["done"] += 1
            user_done = user_sample_progress[user_id]["done"]
            user_total = user_sample_progress[user_id]["total"]
            print(
                f"[Agent 2][User Progress] user {len(processed_user_ids)}/{total_users_with_history} "
                f"(user_id={user_id}) sample {user_done}/{user_total} {_progress_bar(user_done, user_total)} | "
                f"overall {row_idx}/{len(all_history_rows)}"
            )

    def _load_all_user_ids() -> None:
        nonlocal all_user_ids
        conn = sqlite3.connect(str(args.history_db))
        try:
            user_rows = conn.execute("SELECT DISTINCT user_id FROM user_history_profiles").fetchall()
        finally:
            conn.close()
        all_user_ids = [str(r[0]) for r in user_rows]

    router = Qwen3RouterLLM(model_name=args.text_model)
    accessor = GlobalHistoryAccessor(args.global_db, args.history_db)
    recall_agent = RoutingRecallAgent(llm=router, accessor=accessor)

    def _run_agent3_batch() -> None:
        nonlocal intent_outputs
        if not all_user_ids:
            _load_all_user_ids()
        print(f"[Agent 3] Running intent dual recall for all users: {len(all_user_ids)}")
        for user_idx, user_id in enumerate(all_user_ids, start=1):
            out = recall_agent.run(
                user_id=user_id,
                query=args.query,
                min_candidate_items=args.min_candidate_items,
                max_candidate_items=args.max_candidate_items,
                max_history_rows=args.max_history_rows,
                filter_candidates_by_item_type=bool(getattr(args, "filter_candidates_by_item_type", True)),
                candidate_item_ids_scope=getattr(args, "candidate_item_ids_scope", None),
                save_output=True,
                output_dir=args.intent_output_dir,
            )
            print(
                f"[Agent 3] user={user_id}, candidate_items={len(out.candidate_items)}, "
                f"history_rows={len(out.query_relevant_history)} | "
                f"progress {user_idx}/{len(all_user_ids)} {_progress_bar(user_idx, len(all_user_ids))}"
            )
        intent_outputs = _list_saved_agent3_outputs(args.intent_output_dir)

    def _run_agent45_batch() -> None:
        nonlocal intent_outputs
        if not intent_outputs:
            intent_outputs = _list_saved_agent3_outputs(args.intent_output_dir)
        print(f"[Agent 4/5] Running module-3 for all agent3 outputs: {len(intent_outputs)}")
        for p in intent_outputs:
            payload = json.loads(p.read_text(encoding="utf-8"))
            run_module3(
                intent_dual_recall_output=payload,
                model_name=args.text_model,
                top_n=args.top_n,
                disable_must_avoid=bool(getattr(args, "positive_history_only", False)),
                disable_must_have=bool(getattr(args, "disable_must_have", False)),
                disable_prediction_bonus=bool(getattr(args, "disable_prediction_bonus", False)),
                save_output=True,
                output_dir=args.dynamic_output_dir,
            )

    def _run_agent3_agent45_stream() -> None:
        nonlocal intent_outputs
        if not all_user_ids:
            _load_all_user_ids()
        print(f"[Orchestrator] Streaming Agent3 -> Agent4/5 per user: {len(all_user_ids)}")
        for user_idx, user_id in enumerate(all_user_ids, start=1):
            out = recall_agent.run(
                user_id=user_id,
                query=args.query,
                min_candidate_items=args.min_candidate_items,
                max_candidate_items=args.max_candidate_items,
                max_history_rows=args.max_history_rows,
                filter_candidates_by_item_type=bool(getattr(args, "filter_candidates_by_item_type", True)),
                candidate_item_ids_scope=getattr(args, "candidate_item_ids_scope", None),
                save_output=True,
                output_dir=args.intent_output_dir,
            )
            out_file = Path(args.intent_output_dir) / f"{user_id}_intent_dual_recall_output.json"
            if out_file.exists():
                intent_outputs.append(out_file)
                payload = json.loads(out_file.read_text(encoding="utf-8"))
                run_module3(
                    intent_dual_recall_output=payload,
                    model_name=args.text_model,
                    top_n=args.top_n,
                    disable_must_avoid=bool(getattr(args, "positive_history_only", False)),
                    disable_must_have=bool(getattr(args, "disable_must_have", False)),
                    disable_prediction_bonus=bool(getattr(args, "disable_prediction_bonus", False)),
                    save_output=True,
                    output_dir=args.dynamic_output_dir,
                )
            print(
                f"[Orchestrator] user={user_id} done | candidate_items={len(out.candidate_items)} "
                f"| progress {user_idx}/{len(all_user_ids)} {_progress_bar(user_idx, len(all_user_ids))}"
            )

    def _persist_profile_artifacts() -> None:
        _write_jsonl(run_out_dir / "candidate_meta.jsonl", candidate_meta_records)
        _write_jsonl(run_out_dir / "history_meta.jsonl", history_meta_records)
        _write_jsonl(run_out_dir / "candidate_profiles.jsonl", candidate_profile_records)
        _write_jsonl(run_out_dir / "history_profiles.jsonl", history_profile_records)
        _export_sqlite_table_as_jsonl(
            args.global_db,
            "global_item_features",
            run_out_dir / "global_item_features_snapshot.jsonl",
        )
        _export_sqlite_table_as_jsonl(
            args.history_db,
            "user_history_profiles",
            run_out_dir / "user_history_profiles_snapshot.jsonl",
        )

    llm_planner = None
    if bool(getattr(args, "orchestration_use_llm", False)):
        llm_planner = QwenOrchestrationLLM(model_name=str(getattr(args, "orchestration_llm_model", "Qwen/Qwen3-8B")))
    orchestrator = PipelineOrchestratorAgent(llm_planner=llm_planner)
    plan = orchestrator.plan(
        profile=str(getattr(args, "orchestration_profile", "standard")),
        hints={"needs_fresh_item_profiles": not bool(getattr(args, "skip_agent1_if_db_exists", False))},
    )
    print(f"[Orchestrator] profile={plan.profile}, stages={plan.stages}, notes={plan.notes}")

    for stage in plan.stages:
        if stage == "agent1":
            if bool(getattr(args, "skip_agent1_if_db_exists", False)) and Path(args.global_db).exists():
                print("[Orchestrator] Skip Agent1 due to --skip-agent1-if-db-exists and existing global DB.")
                continue
            _run_agent1()
        elif stage == "agent2":
            _run_agent2()
            _persist_profile_artifacts()
        elif stage == "agent3_batch":
            _run_agent3_batch()
        elif stage == "agent45_batch":
            _run_agent45_batch()
        elif stage == "agent3_agent45_stream":
            _run_agent3_agent45_stream()
        elif stage == "bundle":
            pass

    bundle_file = _bundle_paths(
        args.bundle_output,
        [
            Path(args.profiler_run_out_dir),
            Path(args.intent_output_dir),
            Path(args.dynamic_output_dir),
            Path(args.global_db),
            Path(args.history_db),
        ],
    )

    return {
        "orchestration_profile": plan.profile,
        "orchestration_stages": plan.stages,
        "items_processed": len(all_item_ids),
        "history_rows_processed": len(all_history_rows),
        "users_processed": len(all_user_ids),
        "intent_outputs": len(intent_outputs),
        "bundle_output": str(bundle_file),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full 5-agent Amazon pipeline")
    parser.add_argument("--item-desc-tsv", default="./processed/Video_Games_item_desc.tsv")
    parser.add_argument("--user-pairs-tsv", default="./processed/Video_Games_u_i_pairs.tsv")
    parser.add_argument("--user-items-negs-tsv", default="./processed/Video_Games_user_items_negs.tsv")
    parser.add_argument(
        "--agent2-item-desc-tsv",
        default="",
        help="Optional full item_desc tsv for Agent2 metadata fallback when Agent1 uses filtered catalog",
    )

    parser.add_argument("--global-db", default="./processed/global_item_features.db")
    parser.add_argument("--history-db", default="./processed/user_history_log.db")
    parser.add_argument("--profiler-run-out-dir", default="./processed/profiler_runs/full_pipeline")
    parser.add_argument("--intent-output-dir", default="./processed/intent_dual_recall_outputs")
    parser.add_argument("--dynamic-output-dir", default="./processed/dynamic_reasoning_ranking_outputs")

    parser.add_argument("--bundle-output", required=True, help="Zip file path for bundled outputs")

    parser.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--category-hint", default="Video_Games")
    parser.add_argument("--query", default="")

    parser.add_argument("--min-candidate-items", type=int, default=20)
    parser.add_argument("--max-candidate-items", type=int, default=200)
    parser.add_argument("--max-history-rows", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=21)
    parser.add_argument(
        "--orchestration-profile",
        choices=["standard", "user_stream", "minimal_refresh"],
        default="standard",
        help="Execution planning profile decided by the orchestration agent.",
    )
    parser.add_argument(
        "--skip-agent1-if-db-exists",
        action="store_true",
        help="Allow orchestration agent to skip Agent1 when global DB already exists.",
    )
    parser.add_argument(
        "--orchestration-use-llm",
        action="store_true",
        help="Use LLM-based orchestration planning (fallback to heuristic plan on failure).",
    )
    parser.add_argument(
        "--orchestration-llm-model",
        default="Qwen/Qwen3-8B",
        help="LLM model name for orchestration planning.",
    )
    parser.add_argument(
        "--disable-agent3-item-type-filter",
        action="store_true",
        help="Disable Agent3 item-type/category filtering and pass full scoped candidate catalog to Agent5.",
    )
    parser.add_argument(
        "--positive-history-only",
        action="store_true",
        help="Use only positive history rows in Agent2/Agent3 and ignore Must_Avoid constraints in Agent5.",
    )
    parser.add_argument(
        "--disable-must-have",
        action="store_true",
        help="Disable Agent4 Must_Have constraints before Agent5 scoring.",
    )
    parser.add_argument(
        "--disable-prediction-bonus",
        action="store_true",
        help="Disable Agent5 prediction bonus and use logits-weighted score only.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_argparser().parse_args()
    cli_args.filter_candidates_by_item_type = not bool(getattr(cli_args, "disable_agent3_item_type_filter", False))
    cli_args.candidate_item_ids_scope = None
    summary = run_pipeline(cli_args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
