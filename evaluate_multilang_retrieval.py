#!/usr/bin/env python3
"""Evaluate recommendation run JSON against multilang ground truth CSV.

Computes per-student retrieval metrics for one or more K values:
- MRR@K
- Precision@K
- Hit@K
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

print("Position : evaluate_multilang_retrieval.py")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recommendations JSON using ground-truth CSV."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("tests/test_multilang/recommendations_10_stu_20260507T115715Z.json"),
        help="Path to run-result JSON.",
    )
    parser.add_argument(
        "--groundtruth-csv",
        type=Path,
        default=Path("tests/test_multilang/ground_truth_multilang.csv"),
        help="Path to ground-truth CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output per-user metrics CSV.",
    )
    parser.add_argument(
        "--top-ks",
        type=str,
        default="3,5,10",
        help="Comma-separated K values, e.g. '5,10,15'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, evaluate only first N run rows from input JSON.",
    )
    return parser.parse_args()


def parse_top_ks(raw: str) -> list[int]:
    vals: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        k = int(p)
        if k <= 0:
            raise ValueError("All K values must be > 0")
        vals.append(k)
    uniq = sorted(set(vals))
    if not uniq:
        raise ValueError("--top-ks produced no valid K values")
    return uniq


def normalize_student_id(student_id: str) -> str:
    sid = (student_id or "").strip()
    if sid.startswith("user_"):
        return "stu_" + sid[len("user_") :]
    return sid


def parse_gold_ids(row: dict[str, str]) -> list[str]:
    gold_titles = (row.get("gold_titles") or "").strip()
    if gold_titles:
        try:
            parsed = ast.literal_eval(gold_titles)
            if isinstance(parsed, list):
                vals = [str(x).strip() for x in parsed if str(x).strip()]
                if vals:
                    return vals
        except (SyntaxError, ValueError):
            pass

    out: list[str] = []
    for i in range(1, 101):
        key = f"Gold#{i} title"
        if key not in row:
            break
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        feed_id = raw.split("|", 1)[0].strip()
        if feed_id:
            out.append(feed_id)
    return out


def load_groundtruth(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header row in {path}")

        student_key = None
        for k in reader.fieldnames:
            if isinstance(k, str) and k.strip().lower() == "student_id":
                student_key = k
                break
        if not student_key:
            raise ValueError(f'Missing "student_id" column in {path}')

        out: dict[str, list[str]] = {}
        for row in reader:
            sid = (row.get(student_key) or "").strip()
            if not sid:
                continue
            out[sid] = parse_gold_ids(row)
    return out


def extract_predicted_ids(run: dict[str, Any]) -> list[str]:
    response = run.get("response")
    if not isinstance(response, dict):
        return []
    recs = response.get("recommendations")
    if not isinstance(recs, list):
        return []

    ids: list[str] = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        cid = rec.get("feedId")
        if cid is None:
            continue
        val = str(cid).strip()
        if val:
            ids.append(val)
    return ids


def compute_metrics_at_k(predicted_ids: list[str], gold_ids: list[str], k: int) -> dict[str, float | int]:
    top_k = predicted_ids[:k]
    gold_set = {x for x in gold_ids if x}
    if not gold_set:
        return {"mrr": 0.0, "precision": 0.0, "hit": 0.0, "hit_count": 0}

    first_rank = None
    hit_count = 0
    for idx, pred_id in enumerate(top_k, start=1):
        if pred_id in gold_set:
            hit_count += 1
            if first_rank is None:
                first_rank = idx

    mrr = 0.0 if first_rank is None else 1.0 / float(first_rank)
    precision = float(hit_count) / float(k)
    hit = 1.0 if hit_count > 0 else 0.0
    return {"mrr": mrr, "precision": precision, "hit": hit, "hit_count": hit_count}


def main() -> int:
    args = parse_args()
    top_ks = parse_top_ks(args.top_ks)
    print(f"top_ks : {top_ks}")

    with args.input_json.open("r", encoding="utf-8") as f:
        runs = json.load(f)
    if not isinstance(runs, list):
        raise ValueError("Input JSON must be an array of run objects")
    print(f"runs : {(json.dumps(runs[0], ensure_ascii=False, indent=2))}")
    if args.limit > 0:
        runs = runs[: args.limit]

    student_to_gold = load_groundtruth(args.groundtruth_csv)
    print(F"student_to_gold : {json.dumps(student_to_gold,indent=4)}")
    max_k = max(top_ks)
    output_csv = args.output_csv or args.input_json.with_name(f"{args.input_json.stem}_metrics.csv")

    fieldnames = [
        "studentId",
        "student_id_normalized",
        "statusCode",
        "source",
        "error",
        "gold_ids",
        f"predicted_ids_top_{max_k}",
    ]
    for k in top_ks:
        fieldnames.extend([f"MRR@{k}", f"Precision@{k}", f"Hit@{k}", f"hit_count@{k}"])
    for i in range(1, max_k + 1):
        fieldnames.append(f"top_{i}_feed_id")
    print(f"fieldnames : {fieldnames}")

    rows: list[dict[str, Any]] = []
    for run in runs:
        print("="*75)
        if not isinstance(run, dict):
            continue

        student_id_raw = str(run.get("studentId", "")).strip()
        student_id_norm = normalize_student_id(student_id_raw)
        status_code = run.get("statusCode")
        print(F"student_id_raw : {student_id_raw}, status_code : {status_code}")

        response = run.get("response") if isinstance(run.get("response"), dict) else {}
        # print(f"response : {json.dumps(response, indent=4)}")
        source = str(response.get("source", ""))
        predicted_ids = extract_predicted_ids(run)
        print(f"predicted_ids : {predicted_ids}")
        gold_ids = student_to_gold.get(student_id_norm, [])
        print(f"gold_ids : {gold_ids}")

        row: dict[str, Any] = {k: "" for k in fieldnames}
        row["studentId"] = student_id_raw
        row["student_id_normalized"] = student_id_norm
        row["statusCode"] = status_code if status_code is not None else ""
        row["source"] = source
        row["gold_ids"] = " | ".join(gold_ids)
        row[f"predicted_ids_top_{max_k}"] = " | ".join(predicted_ids[:max_k])

        if status_code != 200:
            row["error"] = str(run.get("error") or f"status={status_code}")

        for k in top_ks:
            metrics = compute_metrics_at_k(predicted_ids, gold_ids, k)
            row[f"MRR@{k}"] = round(float(metrics["mrr"]), 6)
            row[f"Precision@{k}"] = round(float(metrics["precision"]), 6)
            row[f"Hit@{k}"] = round(float(metrics["hit"]), 6)
            row[f"hit_count@{k}"] = int(metrics["hit_count"])
            print(f"k : {k:02d} -> metrics : {metrics}")

        for i in range(1, max_k + 1):
            idx = i - 1
            row[f"top_{i}_feed_id"] = predicted_ids[idx] if idx < len(predicted_ids) else ""

        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if not r.get("error")]
    summary: dict[str, Any] = {
        "evaluated_students": len(rows),
        "successful_students": len(valid_rows),
        # "output_csv": str(output_csv),
    }
    if valid_rows:
        for k in top_ks:
            mrr_key = f"MRR@{k}"
            p_key = f"Precision@{k}"
            h_key = f"Hit@{k}"
            summary[f"mean_{mrr_key}"] = round(
                sum(float(r[mrr_key]) for r in valid_rows) / len(valid_rows),
                5,
            )
            summary[f"mean_{p_key}"] = round(
                sum(float(r[p_key]) for r in valid_rows) / len(valid_rows),
                5,
            )
            summary[f"mean_{h_key}"] = round(
                sum(float(r[h_key]) for r in valid_rows) / len(valid_rows),
                5,
            )

    print(json.dumps(summary, indent=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
