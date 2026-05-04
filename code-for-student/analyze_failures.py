#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_failures.py

从 GUI_Agent 的 test_run.log 中提取失败 step，自动归类，并输出：
1. failures.csv                每条失败 step 明细
2. failures.json               JSON 明细
3. summary_by_error_type.csv   按错误类型统计
4. summary_by_case.csv         按用例统计
5. summary_by_action_pair.csv  按 expect/got 动作组合统计
6. failure_report.md           Markdown 汇总报告

用法：
    python analyze_failures.py --log ./output/test_run.log --out ./output/failure_analysis

也支持多个日志：
    python analyze_failures.py --log "./output/*.log" --out ./output/failure_analysis
    python analyze_failures.py --log ./log1.txt ./log2.txt --out ./analysis

说明：
- 只依赖 Python 标准库。
- 支持 Windows PowerShell。
- CSV 使用 utf-8-sig 编码，Excel 打开中文不乱码。
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RE_CASE = re.compile(r"Start testing case:\s*(?P<case>\S+)")
RE_STEP = re.compile(r"--- Step\s+(?P<step>\d+):\s+Current Status\s+(?P<status>-?\d+)\s+---")
RE_AGENT = re.compile(r"Agent Output:\s*action=(?P<action>[A-Z_]+),\s*params=(?P<params>.*)")
RE_HTTP = re.compile(r'HTTP Request:.*"HTTP/1\.1\s+(?P<code>\d{3})[^"]*"')
RE_CHECKER = re.compile(r"\[Checker\]\s*(?P<msg>.*)")
RE_SCORE = re.compile(r"\[No\.\s*(?P<no>\d+):\s*(?P<case>\S+)\]\s+Score:\s+(?P<ok>\d+)/(?P<total>\d+)\s*=\s*(?P<score>[0-9.]+)")
RE_RESULT = re.compile(r"\[No\.\s*(?P<no>\d+):\s*(?P<case>\S+)\]\s+Result:\s+(?P<result>PASS|FAIL)")

RE_ACTION_MISMATCH = re.compile(r"Action mismatch:\s*expect\s*\[(?P<expect>[^\]]+)\],\s*got\s*\[(?P<got>[^\]]+)\]")
RE_CLICK_FAILED_RANGE = re.compile(
    r"CLICK failed:\s*\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)\s*not in\s*"
    r"\(\[(?P<x1>-?\d+),\s*(?P<x2>-?\d+)\],\s*\[(?P<y1>-?\d+),\s*(?P<y2>-?\d+)\]\)"
)
RE_CLICK_FAILED_SCOPE = re.compile(r"CLICK failed:\s*\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)\s*not in scope")
RE_OPEN_MISMATCH = re.compile(r"OPEN:\s*app mismatch,\s*expect\s*'(?P<expect>.*?)',\s*got\s*'(?P<got>.*?)'")


@dataclass
class FailureRecord:
    log_file: str
    case_name: str
    step: Optional[int]
    status: Optional[int]
    agent_action: str
    agent_params: str
    api_status: str
    checker_message: str
    error_type: str
    error_subtype: str
    expected_action: str = ""
    got_action: str = ""
    click_x: str = ""
    click_y: str = ""
    expected_x1: str = ""
    expected_x2: str = ""
    expected_y1: str = ""
    expected_y2: str = ""
    suggested_point: str = ""
    coordinate_bias: str = ""
    expected_app: str = ""
    got_app: str = ""
    history_tail: str = ""


def _safe_literal_eval(text: str) -> Any:
    text = (text or "").strip()
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _action_summary(action: str, params_text: str) -> str:
    params = _safe_literal_eval(params_text)
    if isinstance(params, dict):
        if "point" in params:
            return f"{action}{params.get('point')}"
        if "text" in params:
            text = str(params.get("text", ""))
            if len(text) > 10:
                text = text[:10] + "..."
            return f"{action}({text})"
        if "app_name" in params:
            return f"{action}({params.get('app_name')})"
        if "start_point" in params and "end_point" in params:
            return f"{action}{params.get('start_point')}->{params.get('end_point')}"
    return action


def _normalize_action_name(x: str) -> str:
    return str(x or "").strip().strip("'\"").upper()


def _classify_action_mismatch(expect: str, got: str) -> Tuple[str, str]:
    expect = _normalize_action_name(expect)
    got = _normalize_action_name(got)
    pair = f"expect_{expect}_got_{got}"

    if expect == "CLICK" and got == "SCROLL":
        return "action_mismatch", "premature_scroll"
    if expect == "TYPE" and got == "CLICK":
        return "action_mismatch", "missing_type_after_focus"
    if expect == "TYPE" and got == "SCROLL":
        return "action_mismatch", "missing_type_got_scroll"
    if expect == "COMPLETE" and got in {"CLICK", "SCROLL", "TYPE", "OPEN"}:
        return "action_mismatch", "missed_complete"
    if expect == "CLICK" and got == "COMPLETE":
        return "action_mismatch", "premature_complete"
    if expect == "CLICK" and got == "OPEN":
        return "action_mismatch", "reopen_app_instead_of_click"
    if expect == "OPEN":
        return "action_mismatch", "open_sequence_error"
    return "action_mismatch", pair


def _classify_click_bias(
    x: int,
    y: int,
    x1: Optional[int],
    x2: Optional[int],
    y1: Optional[int],
    y2: Optional[int],
) -> str:
    if x1 is None or x2 is None or y1 is None or y2 is None:
        return "unknown_scope"

    parts: List[str] = []
    if x <= x1:
        parts.append("too_left")
    elif x >= x2:
        parts.append("too_right")

    # 屏幕坐标：y 越大越靠下
    if y <= y1:
        parts.append("too_high")
    elif y >= y2:
        parts.append("too_low")

    return "+".join(parts) if parts else "inside_but_failed"


def classify_checker_message(msg: str) -> Dict[str, str]:
    out: Dict[str, str] = {
        "error_type": "unknown_checker_error",
        "error_subtype": "unknown",
        "expected_action": "",
        "got_action": "",
        "click_x": "",
        "click_y": "",
        "expected_x1": "",
        "expected_x2": "",
        "expected_y1": "",
        "expected_y2": "",
        "suggested_point": "",
        "coordinate_bias": "",
        "expected_app": "",
        "got_app": "",
    }

    m = RE_ACTION_MISMATCH.search(msg)
    if m:
        expect = _normalize_action_name(m.group("expect"))
        got = _normalize_action_name(m.group("got"))
        error_type, subtype = _classify_action_mismatch(expect, got)
        out.update(
            {
                "error_type": error_type,
                "error_subtype": subtype,
                "expected_action": expect,
                "got_action": got,
            }
        )
        return out

    m = RE_CLICK_FAILED_RANGE.search(msg)
    if m:
        x, y = int(m.group("x")), int(m.group("y"))
        x1, x2 = int(m.group("x1")), int(m.group("x2"))
        y1, y2 = int(m.group("y1")), int(m.group("y2"))
        sx = int((x1 + x2) / 2)
        sy = int((y1 + y2) / 2)
        bias = _classify_click_bias(x, y, x1, x2, y1, y2)
        out.update(
            {
                "error_type": "click_failed",
                "error_subtype": "click_out_of_expected_region",
                "expected_action": "CLICK",
                "got_action": "CLICK",
                "click_x": str(x),
                "click_y": str(y),
                "expected_x1": str(x1),
                "expected_x2": str(x2),
                "expected_y1": str(y1),
                "expected_y2": str(y2),
                "suggested_point": f"[{sx}, {sy}]",
                "coordinate_bias": bias,
            }
        )
        return out

    m = RE_CLICK_FAILED_SCOPE.search(msg)
    if m:
        x, y = int(m.group("x")), int(m.group("y"))
        out.update(
            {
                "error_type": "click_failed",
                "error_subtype": "click_not_in_scope",
                "expected_action": "CLICK",
                "got_action": "CLICK",
                "click_x": str(x),
                "click_y": str(y),
                "coordinate_bias": "unknown_scope",
            }
        )
        return out

    m = RE_OPEN_MISMATCH.search(msg)
    if m:
        out.update(
            {
                "error_type": "open_app_mismatch",
                "error_subtype": "app_name_alias_or_wrong_app",
                "expected_action": "OPEN",
                "got_action": "OPEN",
                "expected_app": m.group("expect"),
                "got_app": m.group("got"),
            }
        )
        return out

    if "parse" in msg.lower() or "format" in msg.lower():
        out.update({"error_type": "parse_or_format_error", "error_subtype": "parser_or_output_format"})
        return out

    return out


def expand_log_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        matched = glob.glob(item)
        if matched:
            paths.extend(Path(p) for p in matched)
        else:
            paths.append(Path(item))

    seen = set()
    unique: List[Path] = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def parse_log_file(path: Path) -> Tuple[List[FailureRecord], Dict[str, Dict[str, str]]]:
    failures: List[FailureRecord] = []
    case_scores: Dict[str, Dict[str, str]] = {}

    current_case = ""
    current_step: Optional[int] = None
    current_status: Optional[int] = None
    last_action = ""
    last_params_text = ""
    last_api_status = ""
    history: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = RE_CASE.search(line)
            if m:
                current_case = m.group("case")
                current_step = None
                current_status = None
                last_action = ""
                last_params_text = ""
                last_api_status = ""
                history = []
                continue

            m = RE_STEP.search(line)
            if m:
                current_step = int(m.group("step"))
                current_status = int(m.group("status"))
                last_action = ""
                last_params_text = ""
                last_api_status = ""
                continue

            m = RE_HTTP.search(line)
            if m:
                last_api_status = m.group("code")
                continue

            m = RE_AGENT.search(line)
            if m:
                last_action = _normalize_action_name(m.group("action"))
                last_params_text = m.group("params").strip()
                history.append(_action_summary(last_action, last_params_text))
                if len(history) > 8:
                    history = history[-8:]
                continue

            m = RE_CHECKER.search(line)
            if m:
                msg = m.group("msg").strip()
                info = classify_checker_message(msg)

                if last_api_status and last_api_status != "200":
                    if info["error_type"] == "unknown_checker_error":
                        info["error_type"] = "api_error_or_fallback"
                        info["error_subtype"] = f"api_{last_api_status}"
                    else:
                        info["error_subtype"] = f"{info['error_subtype']}+api_{last_api_status}"

                failures.append(
                    FailureRecord(
                        log_file=str(path),
                        case_name=current_case,
                        step=current_step,
                        status=current_status,
                        agent_action=last_action,
                        agent_params=last_params_text,
                        api_status=last_api_status,
                        checker_message=msg,
                        error_type=info["error_type"],
                        error_subtype=info["error_subtype"],
                        expected_action=info["expected_action"],
                        got_action=info["got_action"],
                        click_x=info["click_x"],
                        click_y=info["click_y"],
                        expected_x1=info["expected_x1"],
                        expected_x2=info["expected_x2"],
                        expected_y1=info["expected_y1"],
                        expected_y2=info["expected_y2"],
                        suggested_point=info["suggested_point"],
                        coordinate_bias=info["coordinate_bias"],
                        expected_app=info["expected_app"],
                        got_app=info["got_app"],
                        history_tail=" -> ".join(history[-5:]),
                    )
                )
                continue

            m = RE_SCORE.search(line)
            if m:
                case = m.group("case")
                case_scores[case] = {
                    "score_ok": m.group("ok"),
                    "score_total": m.group("total"),
                    "score": m.group("score"),
                    "result": "PASS" if m.group("ok") == m.group("total") else "FAIL",
                }
                continue

            m = RE_RESULT.search(line)
            if m:
                case = m.group("case")
                case_scores.setdefault(case, {})
                case_scores[case]["result"] = m.group("result")
                continue

    return failures, case_scores


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_failures(failures: List[FailureRecord]) -> Dict[str, Any]:
    by_type = Counter(r.error_type for r in failures)
    by_subtype = Counter(r.error_subtype for r in failures)
    by_case = Counter(r.case_name for r in failures)
    by_pair = Counter(
        f"{r.expected_action}->{r.got_action}"
        for r in failures
        if r.expected_action or r.got_action
    )
    by_bias = Counter(r.coordinate_bias for r in failures if r.coordinate_bias)
    by_api = Counter(r.api_status or "unknown" for r in failures)

    return {
        "total_failures": len(failures),
        "by_type": by_type,
        "by_subtype": by_subtype,
        "by_case": by_case,
        "by_pair": by_pair,
        "by_bias": by_bias,
        "by_api": by_api,
    }


def rows_from_counter(counter: Counter, name_key: str = "name") -> List[Dict[str, Any]]:
    total = sum(counter.values())
    rows = []
    for k, v in counter.most_common():
        rows.append(
            {
                name_key: k,
                "count": v,
                "ratio": f"{(v / total * 100):.2f}%" if total else "0.00%",
            }
        )
    return rows


def md_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    if not rows:
        return "_无数据_\n"
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    path: Path,
    failures: List[FailureRecord],
    summary: Dict[str, Any],
    case_scores: Dict[str, Dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    top_cases = rows_from_counter(summary["by_case"], "case")[:20]
    top_types = rows_from_counter(summary["by_type"], "error_type")
    top_subtypes = rows_from_counter(summary["by_subtype"], "error_subtype")[:20]
    top_pairs = rows_from_counter(summary["by_pair"], "expected_to_got")[:20]
    top_bias = rows_from_counter(summary["by_bias"], "coordinate_bias")[:20]
    top_api = rows_from_counter(summary["by_api"], "api_status")[:20]
    latest_failures = [asdict(r) for r in failures[:30]]

    content = []
    content.append("# GUI Agent 失败 Step 分析报告\n")
    content.append(f"- 失败记录总数：**{len(failures)}**\n")
    if case_scores:
        passed = [k for k, v in case_scores.items() if v.get("result") == "PASS"]
        failed = [k for k, v in case_scores.items() if v.get("result") == "FAIL"]
        content.append(f"- 解析到 case 数：**{len(case_scores)}**\n")
        content.append(f"- PASS case 数：**{len(passed)}**\n")
        content.append(f"- FAIL case 数：**{len(failed)}**\n")

    content.append("\n## 1. 错误类型分布\n")
    content.append(md_table(top_types, ["error_type", "count", "ratio"]))

    content.append("\n## 2. 错误子类型分布\n")
    content.append(md_table(top_subtypes, ["error_subtype", "count", "ratio"]))

    content.append("\n## 3. 按用例统计\n")
    content.append(md_table(top_cases, ["case", "count", "ratio"]))

    content.append("\n## 4. 动作期望/实际组合\n")
    content.append(md_table(top_pairs, ["expected_to_got", "count", "ratio"]))

    content.append("\n## 5. 点击偏差方向\n")
    content.append(md_table(top_bias, ["coordinate_bias", "count", "ratio"]))

    content.append("\n## 6. API 状态分布\n")
    content.append(md_table(top_api, ["api_status", "count", "ratio"]))

    content.append("\n## 7. 前 30 条失败明细\n")
    if latest_failures:
        headers = [
            "case_name",
            "step",
            "status",
            "agent_action",
            "error_type",
            "error_subtype",
            "checker_message",
            "suggested_point",
            "history_tail",
        ]
        content.append(md_table(latest_failures, headers))
    else:
        content.append("_没有解析到失败记录。_\n")

    content.append("\n## 8. 提分建议读取方式\n")
    content.append(
        "- `missing_type_after_focus` 多：优先做历史驱动 TYPE 保护。\n"
        "- `premature_scroll` 多：降低 SCROLL 优先级，优先 CLICK 可见目标。\n"
        "- `missed_complete` 多：加强 TYPE 后完成判断，尤其是评价/评论/备注任务。\n"
        "- `click_failed` 多：优先引入 bbox 中心化和 UI anchor 坐标修正。\n"
        "- `open_app_mismatch` 多：扩充 App 名归一化表。\n"
        "- `api_400 / api_429` 多：先修 API 配置或限流，否则分析结果会混入 fallback 假动作。\n"
    )

    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GUI_Agent test_run.log failure steps.")
    parser.add_argument("--log", nargs="+", required=True, help="Path(s) or glob(s) of test_run.log")
    parser.add_argument("--out", default="./failure_analysis", help="Output directory")
    args = parser.parse_args()

    log_paths = expand_log_paths(args.log)
    existing_paths = [p for p in log_paths if p.exists()]
    if not existing_paths:
        raise SystemExit(f"没有找到日志文件：{args.log}")

    all_failures: List[FailureRecord] = []
    all_case_scores: Dict[str, Dict[str, str]] = {}

    for path in existing_paths:
        failures, case_scores = parse_log_file(path)
        all_failures.extend(failures)
        all_case_scores.update(case_scores)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_rows = [asdict(r) for r in all_failures]
    fieldnames = list(FailureRecord.__dataclass_fields__.keys())

    write_csv(out_dir / "failures.csv", failure_rows, fieldnames)
    (out_dir / "failures.json").write_text(
        json.dumps(failure_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = summarize_failures(all_failures)

    write_csv(out_dir / "summary_by_error_type.csv", rows_from_counter(summary["by_type"], "error_type"), ["error_type", "count", "ratio"])
    write_csv(out_dir / "summary_by_error_subtype.csv", rows_from_counter(summary["by_subtype"], "error_subtype"), ["error_subtype", "count", "ratio"])
    write_csv(out_dir / "summary_by_case.csv", rows_from_counter(summary["by_case"], "case"), ["case", "count", "ratio"])
    write_csv(out_dir / "summary_by_action_pair.csv", rows_from_counter(summary["by_pair"], "expected_to_got"), ["expected_to_got", "count", "ratio"])
    write_csv(out_dir / "summary_by_coordinate_bias.csv", rows_from_counter(summary["by_bias"], "coordinate_bias"), ["coordinate_bias", "count", "ratio"])
    write_csv(out_dir / "summary_by_api_status.csv", rows_from_counter(summary["by_api"], "api_status"), ["api_status", "count", "ratio"])

    case_score_rows = []
    for case, info in sorted(all_case_scores.items()):
        row = {"case": case}
        row.update(info)
        case_score_rows.append(row)
    write_csv(
        out_dir / "case_scores.csv",
        case_score_rows,
        ["case", "result", "score_ok", "score_total", "score"],
    )

    write_markdown_report(out_dir / "failure_report.md", all_failures, summary, all_case_scores)

    print("分析完成。")
    print(f"日志文件数: {len(existing_paths)}")
    print(f"失败记录数: {len(all_failures)}")
    print(f"输出目录: {out_dir.resolve()}")
    print("\nTop 错误类型:")
    for k, v in summary["by_type"].most_common(10):
        print(f"  {k}: {v}")

    print("\n建议先看:")
    print(f"  {out_dir / 'failure_report.md'}")
    print(f"  {out_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()
