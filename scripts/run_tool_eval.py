"""Main script to run tool calling benchmarks and generate evaluation reports."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from app.eval.tool_runner import ToolRunner
from app.eval.tool_metrics import (
    calculate_intent_accuracy,
    calculate_tool_success_rate,
    calculate_end_to_end_accuracy,
    calculate_confusion_matrix,
    calculate_latency_stats,
)

def run_evaluation() -> Dict[str, Any]:
    """Execute the evaluation runner and return calculated metrics."""
    runner = ToolRunner()
    raw_results = runner.run_eval()
    cases = raw_results["cases"]

    # Gather lists for metric calculators
    expected_intents = [c["expected_intent"] for c in cases]
    actual_intents = [c["actual_intent"] for c in cases]
    
    # Tool success rates (only count actual tool executions)
    tool_successes = [
        c["tool_success"] for c in cases if c["expected_intent"] != "rag"
    ]
    
    answers = [c["answer"] for c in cases]
    expected_contains = [c["expected_contains"] for c in cases]
    
    router_latencies = [c["latency_router_ms"] for c in cases]
    exec_latencies = [c["latency_exec_ms"] for c in cases if c["latency_exec_ms"] > 0]
    e2e_latencies = [c["latency_e2e_ms"] for c in cases]

    # Calculate overall metrics
    intent_acc = calculate_intent_accuracy(expected_intents, actual_intents)
    exec_rate = calculate_tool_success_rate(tool_successes)
    e2e_acc = calculate_end_to_end_accuracy(answers, expected_contains)
    
    conf_matrix = calculate_confusion_matrix(expected_intents, actual_intents)
    
    router_lat_stats = calculate_latency_stats(router_latencies)
    exec_lat_stats = calculate_latency_stats(exec_latencies)
    e2e_lat_stats = calculate_latency_stats(e2e_latencies)

    # Compute stats per category/tool
    categories = sorted(list(set(expected_intents)))
    category_metrics = {}
    for cat in categories:
        cat_cases = [c for c in cases if c["expected_intent"] == cat]
        cat_expected = [c["expected_intent"] for c in cat_cases]
        cat_actual = [c["actual_intent"] for c in cat_cases]
        cat_answers = [c["answer"] for c in cat_cases]
        cat_contains = [c["expected_contains"] for c in cat_cases]
        
        cat_intent_acc = calculate_intent_accuracy(cat_expected, cat_actual)
        cat_e2e_acc = calculate_end_to_end_accuracy(cat_answers, cat_contains)
        
        category_metrics[cat] = {
            "total_cases": len(cat_cases),
            "intent_accuracy": cat_intent_acc,
            "end_to_end_accuracy": cat_e2e_acc,
        }

    # Identify failed cases
    failed_cases = []
    for c in cases:
        if not c["intent_match"] or not c["answer_correct"] or not c["tool_success"]:
            failed_cases.append({
                "id": c["id"],
                "query": c["query"],
                "expected_intent": c["expected_intent"],
                "actual_intent": c["actual_intent"],
                "tool_success": c["tool_success"],
                "tool_error": c["tool_error"],
                "expected_contains": c["expected_contains"],
                "answer": c["answer"],
                "intent_match": c["intent_match"],
                "answer_correct": c["answer_correct"]
            })

    return {
        "overall": {
            "total_cases": len(cases),
            "intent_accuracy": intent_acc,
            "tool_success_rate": exec_rate,
            "end_to_end_accuracy": e2e_acc,
        },
        "per_category": category_metrics,
        "confusion_matrix": conf_matrix,
        "latencies": {
            "router_ms": router_lat_stats,
            "executor_ms": exec_lat_stats,
            "e2e_ms": e2e_lat_stats
        },
        "failed_cases": failed_cases,
        "raw_cases": cases
    }


def generate_markdown_report(metrics_data: Dict[str, Any], filepath: Path) -> None:
    """Generate and write a detailed human-readable Markdown evaluation report."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    overall = metrics_data["overall"]
    latencies = metrics_data["latencies"]
    per_cat = metrics_data["per_category"]
    matrix = metrics_data["confusion_matrix"]
    failed = metrics_data["failed_cases"]
    
    lines = [
        "# Sprint 10 — Tool Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "This report scores the deterministic intent routing accuracy, execution health, and end-to-end correctness of the RAG tool calling pipeline.",
        "",
        "### Key Metrics",
        "",
        "| Metric | Result |",
        "| :--- | :--- |",
        f"| **Total Evaluation Cases** | {overall['total_cases']} |",
        f"| **Intent Classification Accuracy** | {overall['intent_accuracy'] * 100.0:.2f}% |",
        f"| **Tool Execution Success Rate** | {overall['tool_success_rate'] * 100.0:.2f}% |",
        f"| **End-to-End Answer Accuracy** | {overall['end_to_end_accuracy'] * 100.0:.2f}% |",
        "",
        "### Per-Category Performance",
        "",
        "| Category | Cases | Intent Accuracy | E2E Accuracy |",
        "| :--- | :---: | :---: | :---: |"
    ]
    
    for cat, data in per_cat.items():
        lines.append(f"| `{cat}` | {data['total_cases']} | {data['intent_accuracy']*100.0:.1f}% | {data['end_to_end_accuracy']*100.0:.1f}% |")

    lines.extend([
        "",
        "## Intent Classification Matrix (Confusion Matrix)",
        "",
        "Shows actual routed intent mappings against expected ground truth:"
    ])
    
    categories = sorted(matrix.keys())
    header_row = "| Expected \\ Actual | " + " | ".join(f"`{c}`" for c in categories) + " |"
    sep_row = "| :--- | " + " | ".join(":---:" for _ in categories) + " |"
    lines.append(header_row)
    lines.append(sep_row)
    
    for exp in categories:
        row = f"| **{exp}** | " + " | ".join(str(matrix[exp][act]) for act in categories) + " |"
        lines.append(row)

    lines.extend([
        "",
        "## Latency Statistics",
        "",
        "| Stage | Average Latency | Min Latency | Max Latency |",
        "| :--- | :---: | :---: | :---: |",
        f"| Intent Routing | {latencies['router_ms']['avg']:.2f} ms | {latencies['router_ms']['min']:.2f} ms | {latencies['router_ms']['max']:.2f} ms |",
        f"| Tool Execution | {latencies['executor_ms']['avg']:.2f} ms | {latencies['executor_ms']['min']:.2f} ms | {latencies['executor_ms']['max']:.2f} ms |",
        f"| End-to-End RAG | {latencies['e2e_ms']['avg']:.2f} ms | {latencies['e2e_ms']['min']:.2f} ms | {latencies['e2e_ms']['max']:.2f} ms |",
        "",
        "## Failed Cases Summary",
        ""
    ])
    
    if not failed:
        lines.append("🎉 All cases passed successfully! Zero classification or execution errors detected.")
    else:
        lines.append(f"Found {len(failed)} case failure(s):")
        lines.append("")
        for fcase in failed:
            lines.extend([
                f"### Case ID: `{fcase['id']}`",
                f"- **Query**: \"{fcase['query']}\"",
                f"- **Expected Intent**: `{fcase['expected_intent']}` | **Actual Intent**: `{fcase['actual_intent']}`",
                f"- **Tool Success**: `{fcase['tool_success']}` (Error: `{fcase['tool_error']}`)",
                f"- **Expected Substring in Answer**: \"{fcase['expected_contains']}\"",
                f"- **Actual LLM Output**:",
                f"  ```text\n  {fcase['answer']}\n  ```",
                "---",
                ""
            ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Entrypoint to run benchmarks and render CLI outputs."""
    metrics_data = run_evaluation()
    
    # 1. Output json if requested
    if "--json" in sys.argv:
        print(json.dumps(metrics_data, indent=2))
        return

    # 2. Output formatted CLI tables
    overall = metrics_data["overall"]
    latencies = metrics_data["latencies"]
    per_cat = metrics_data["per_category"]
    
    print("================================================================================")
    print("SPRINT 10 — TOOL CALLING EVALUATION REPORT")
    print("================================================================================")
    print(f"Total Cases Run        : {overall['total_cases']}")
    print(f"Intent Accuracy        : {overall['intent_accuracy'] * 100.0:.2f}%")
    print(f"Tool Execution Success : {overall['tool_success_rate'] * 100.0:.2f}%")
    print(f"End-to-End Accuracy   : {overall['end_to_end_accuracy'] * 100.0:.2f}%")
    print("--------------------------------------------------------------------------------")
    print("PER-CATEGORY STATS:")
    print("Category        | Cases | Intent Accuracy | E2E Accuracy")
    print("--------------------------------------------------------------------------------")
    for cat, data in per_cat.items():
        print(f"{cat:<15} | {data['total_cases']:<5} | {data['intent_accuracy']*100.0:<14.1f}% | {data['end_to_end_accuracy']*100.0:.1f}%")
    print("--------------------------------------------------------------------------------")
    print("LATENCY METRICS:")
    print(f" - Router Latency : Avg {latencies['router_ms']['avg']:.2f}ms | Min {latencies['router_ms']['min']:.2f}ms | Max {latencies['router_ms']['max']:.2f}ms")
    print(f" - Exec Latency   : Avg {latencies['executor_ms']['avg']:.2f}ms | Min {latencies['executor_ms']['min']:.2f}ms | Max {latencies['executor_ms']['max']:.2f}ms")
    print(f" - End-to-End     : Avg {latencies['e2e_ms']['avg']:.2f}ms | Min {latencies['e2e_ms']['min']:.2f}ms | Max {latencies['e2e_ms']['max']:.2f}ms")
    print("================================================================================")

    # 3. Write Markdown Report
    report_path = Path("docs/eval/tool_eval_report.md")
    generate_markdown_report(metrics_data, report_path)
    print(f"Report written to: {report_path.absolute()}")

if __name__ == "__main__":
    main()
