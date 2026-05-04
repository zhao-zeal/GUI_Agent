"""Robust parser for GUI Agent model outputs.

The evaluation code expects an AgentOutput(action, parameters) object.  VLM
responses are often not perfectly formatted, so this module accepts JSON,
markdown code blocks, function-call style strings and the compact examples in
the problem statement, then normalizes them to the official action schema.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, Optional, Tuple

VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"}

ACTION_ALIASES = {
    "CLICK": "CLICK",
    "TAP": "CLICK",
    "点击": "CLICK",
    "单击": "CLICK",
    "点按": "CLICK",
    "SCROLL": "SCROLL",
    "SWIPE": "SCROLL",
    "SLIDE": "SCROLL",
    "滚动": "SCROLL",
    "滑动": "SCROLL",
    "上滑": "SCROLL",
    "下滑": "SCROLL",
    "TYPE": "TYPE",
    "INPUT": "TYPE",
    "ENTER": "TYPE",
    "输入": "TYPE",
    "填写": "TYPE",
    "OPEN": "OPEN",
    "LAUNCH": "OPEN",
    "打开": "OPEN",
    "启动": "OPEN",
    "COMPLETE": "COMPLETE",
    "DONE": "COMPLETE",
    "FINISH": "COMPLETE",
    "完成": "COMPLETE",
    "结束": "COMPLETE",
}


def clamp_int(value: Any, lo: int = 0, hi: int = 1000) -> int:
    """Convert value to an integer normalized coordinate and clamp it."""
    try:
        num = int(round(float(value)))
    except Exception:
        num = 500
    return max(lo, min(hi, num))


def _strip_fences(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json|python|text)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    return text


def _find_json_like(text: str) -> Optional[str]:
    """Return the first balanced {...} segment, if any."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str: Optional[str] = None
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _loads_relaxed(text: str) -> Optional[Any]:
    """Parse JSON or Python-literal-like content."""
    candidates = []
    stripped = _strip_fences(text)
    candidates.append(stripped)
    json_like = _find_json_like(stripped)
    if json_like:
        candidates.append(json_like)

    for item in candidates:
        item = item.strip()
        if not item:
            continue
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(item)
            except Exception:
                pass
        # Common model mistake: unquoted keys in a JSON-like object.
        fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', item)
        fixed = fixed.replace("，", ",").replace("：", ":")
        try:
            return json.loads(fixed)
        except Exception:
            try:
                return ast.literal_eval(fixed)
            except Exception:
                pass
    return None


def _normalize_action(action: Any) -> Optional[str]:
    if action is None:
        return None
    if not isinstance(action, str):
        action = str(action)
    action = action.strip().strip("'\"")
    if not action:
        return None
    upper = action.upper()
    if upper in ACTION_ALIASES:
        return ACTION_ALIASES[upper]
    if action in ACTION_ALIASES:
        return ACTION_ALIASES[action]
    # Some models return "click(point=[...])" in the action field.
    m = re.match(r"([A-Za-z_\u4e00-\u9fa5]+)\s*\(", action)
    if m:
        return _normalize_action(m.group(1))
    return upper if upper in VALID_ACTIONS else None


def _numbers_from_text(text: str) -> list:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _flatten_one_level(value: Any) -> Any:
    # Convert [[100, 200]] to [100, 200].
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def _as_point(value: Any) -> Optional[list]:
    value = _flatten_one_level(value)
    if isinstance(value, str):
        nums = _numbers_from_text(value)
        if len(nums) >= 2:
            return [clamp_int(nums[0]), clamp_int(nums[1])]
        return None
    if isinstance(value, dict):
        # Accept {'x': 100, 'y': 200} or {'point': [100, 200]}.
        if "point" in value:
            return _as_point(value.get("point"))
        if "x" in value and "y" in value:
            return [clamp_int(value.get("x")), clamp_int(value.get("y"))]
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first, second = value[0], value[1]
        if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
            return _as_point(first)
        return [clamp_int(first), clamp_int(second)]
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)) and value:
        return _as_text(value[0])
    return str(value).strip()


def _pick(mapping: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        if key in mapping:
            return mapping[key]
        low = key.lower()
        if low in lower_map:
            return lower_map[low]
    return default


def _normalize_parameters(action: str, params: Any, source_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source_obj = source_obj or {}
    if params is None:
        params = {}

    if action == "COMPLETE":
        return {}

    if action == "CLICK":
        value = None
        if isinstance(params, dict):
            value = _pick(params, ["point", "position", "coord", "coordinate", "坐标", "位置", "target"])
            if value is None:
                value = params
        else:
            value = params
        point = _as_point(value) or [500, 500]
        return {"point": point}

    if action == "SCROLL":
        start = end = None
        if isinstance(params, dict):
            start = _pick(params, ["start_point", "start", "from", "起点", "开始"])
            end = _pick(params, ["end_point", "end", "to", "终点", "结束"])
            if start is None and end is None:
                nums = _numbers_from_text(str(params))
                if len(nums) >= 4:
                    start, end = nums[:2], nums[2:4]
        elif isinstance(params, str):
            nums = _numbers_from_text(params)
            if len(nums) >= 4:
                start, end = nums[:2], nums[2:4]
        elif isinstance(params, (list, tuple)):
            if len(params) >= 2 and isinstance(params[0], (list, tuple)) and isinstance(params[1], (list, tuple)):
                start, end = params[0], params[1]
            else:
                nums = []
                for item in params:
                    if isinstance(item, (int, float, str)):
                        try:
                            nums.append(float(item))
                        except Exception:
                            nums.extend(_numbers_from_text(str(item)))
                if len(nums) >= 4:
                    start, end = nums[:2], nums[2:4]
        start_point = _as_point(start) or [500, 800]
        end_point = _as_point(end) or [500, 250]
        return {"start_point": start_point, "end_point": end_point}

    if action == "TYPE":
        if isinstance(params, dict):
            text = _pick(params, ["text", "content", "value", "input", "文字", "内容"], "")
        else:
            text = params
        text = _as_text(text)
        # The official GitHub checker currently reads the key "text".  Keep a
        # content alias to tolerate statement/checker variations without
        # affecting the current checker.
        return {"text": text, "content": text}

    if action == "OPEN":
        if isinstance(params, dict):
            app_name = _pick(params, ["app_name", "app", "name", "application", "应用", "应用名"], "")
        else:
            app_name = params
        return {"app_name": _as_text(app_name)}

    return {}


def _from_dict(obj: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    action = _normalize_action(_pick(obj, ["action", "Action", "act", "操作", "动作", "type", "name"]))

    # Some models output {"CLICK": [[100, 200]]}.
    if not action:
        for key, value in obj.items():
            maybe = _normalize_action(key)
            if maybe:
                return maybe, _normalize_parameters(maybe, value, obj)
        return None

    params = _pick(obj, ["parameters", "params", "parameter", "args", "arguments", "参数"], None)
    if params is None:
        # Treat remaining keys as params, e.g. {"action":"CLICK","point":[...]}
        params = {k: v for k, v in obj.items() if str(k).lower() not in {"action", "act", "操作", "动作", "type", "name", "thought", "reason"}}
    return action, _normalize_parameters(action, params, obj)


def _from_text(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    raw = _strip_fences(text)

    # Prefer an explicit Action line if present.
    action_line = None
    for line in raw.splitlines():
        if re.search(r"^\s*(Action|动作|操作)\s*[:：]", line, flags=re.I):
            action_line = re.sub(r"^\s*(Action|动作|操作)\s*[:：]\s*", "", line, flags=re.I).strip()
    candidates = [action_line, raw] if action_line else [raw]

    for cand in candidates:
        if not cand:
            continue
        cand = cand.strip()

        # Function-call style: click(point='<point>100 200</point>')
        m = re.search(r"([A-Za-z_\u4e00-\u9fa5]+)\s*\((.*)\)\s*$", cand, flags=re.S)
        if m:
            action = _normalize_action(m.group(1))
            if action:
                body = m.group(2)
                params: Dict[str, Any] = {}
                # Extract named arguments. This intentionally stays lenient.
                for key, val in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('[^']*'|\"[^\"]*\"|\[[^\]]*\]|<point>.*?</point>|[^,]+)", body, flags=re.S):
                    params[key] = val.strip().strip("'\"")
                if not params:
                    params = {"raw": body}
                return action, _normalize_parameters(action, params)

        # Compact official examples: CLICK:[[100, 200]], TYPE:['内容'].
        m = re.search(r"\b(CLICK|TYPE|SCROLL|OPEN|COMPLETE)\b\s*[:：]\s*(.*)", cand, flags=re.I | re.S)
        if m:
            action = _normalize_action(m.group(1))
            payload = m.group(2).strip()
            parsed_payload = _loads_relaxed(payload)
            if parsed_payload is None:
                parsed_payload = payload
            if action:
                return action, _normalize_parameters(action, parsed_payload)

        # Bare action plus coordinates/text.
        for alias in sorted(ACTION_ALIASES, key=len, reverse=True):
            if alias in cand or alias.lower() in cand.lower():
                action = _normalize_action(alias)
                if action:
                    return action, _normalize_parameters(action, cand)

    return None


def parse_action(raw_output: Any) -> Tuple[str, Dict[str, Any]]:
    """Return a normalized (action, parameters) pair.

    Raises:
        ValueError: if no action can be extracted.
    """
    if raw_output is None:
        raise ValueError("empty model output")
    if isinstance(raw_output, dict):
        parsed = _from_dict(raw_output)
        if parsed:
            return parsed
        raise ValueError("no valid action in dict")

    text = str(raw_output).strip()
    obj = _loads_relaxed(text)
    if isinstance(obj, dict):
        parsed = _from_dict(obj)
        if parsed:
            return parsed
    elif isinstance(obj, (list, tuple)) and obj:
        # e.g. ["CLICK", [[100, 200]]]
        action = _normalize_action(obj[0])
        if action:
            params = obj[1] if len(obj) > 1 else {}
            return action, _normalize_parameters(action, params)

    parsed_text = _from_text(text)
    if parsed_text:
        return parsed_text
    raise ValueError(f"unable to parse action from output: {text[:200]}")
