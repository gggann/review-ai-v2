"""
five_point.py — 3단계: 10점제 품질 평가
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10점 만점 품질 평가에 관한 모든 로직을 담당합니다.
· 채점 프롬프트 생성  ← five_point_rules.json 에서 동적으로 조립
· AI API 호출
· 프레임 그리드 생성
· 평가 결과 처리 + 점수 범위 방어 검증
· 카테고리 예외 규칙 최소 점수 보장 (v1.3)

[점수 구성]
  품질(Quality)    : 0 ~ 2점
  가시성(Visibility): 0 ~ 4점
  행동성(Action)   : 0 ~ 4점
  총점             : 0 ~ 10점

채점 기준을 바꿀 때는 five_point_rules.json 만 수정하면 됩니다.
"""

import os
import re
import json
import asyncio
import base64
import numpy as np
import cv2

from config import openai_client, GPT_MODEL, log, BASE_DIR


# ==========================================
# rules JSON 로드 (모듈 임포트 시 1회만 실행)
# ==========================================
_RULES_FILE = os.path.join(BASE_DIR, "five_point_rules.json")
_rules_cache: dict = {}


def _load_rules() -> dict:
    """five_point_rules.json 을 로드하고 캐싱한다."""
    global _rules_cache
    if _rules_cache:
        return _rules_cache

    path = _RULES_FILE
    if not os.path.exists(path):
        # config.py 와 같은 폴더에서 한 번 더 탐색
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "five_point_rules.json")
        if os.path.exists(alt):
            path = alt

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _rules_cache = json.load(f)
            log(f"[five_point] rules 로드 완료: {path}")
            return _rules_cache
        except Exception as e:
            log(f"[경고] five_point_rules.json 파싱 오류: {e}")

    log("[경고] five_point_rules.json 을 찾지 못했습니다. 빈 규칙으로 진행합니다.")
    _rules_cache = {}
    return _rules_cache


# ==========================================
# 점수 범위 상수 — JSON validation 섹션에서 동적으로 읽되,
# JSON 로드 실패 시 하드코딩 값으로 폴백
# ==========================================
def _get_score_ranges() -> dict[str, tuple[int, int]]:
    rules = _load_rules()
    clamp = rules.get("validation", {}).get("clamp_ranges", {})
    if clamp:
        return {k: tuple(v) for k, v in clamp.items()}
    # 폴백
    return {
        "quality":    (0, 2),
        "visibility": (0, 4),
        "action":     (0, 4),
    }


# ==========================================
# 프롬프트 조립 헬퍼
# ==========================================
def _build_score_item_section(key: str, item: dict) -> str:
    """
    score_items 의 항목 1개를 프롬프트 텍스트 블록으로 변환한다.
    """
    label     = item.get("label", key)
    max_score = item.get("max", "?")
    desc      = item.get("desc", "")
    levels    = item.get("levels", {})

    # 행동성 전용 부가 안내
    judgment_basis   = item.get("judgment_basis", "")
    excluded_actions = item.get("excluded_as_action", [])

    sep = "━" * 29
    lines = [sep, f"### {label} [0 ~ {max_score}점]", sep]

    if desc:
        lines.append(f"※ {desc}")
    if judgment_basis:
        lines.append(f'※ "{judgment_basis}"')
    for ex in excluded_actions:
        lines.append(f"※ {ex}은 행동으로 인정하지 않는다")

    # 점수 내림차순 출력
    for score_key in sorted(levels.keys(), key=lambda x: int(x), reverse=True):
        level      = levels[score_key]
        label_str  = level.get("label", "")
        conditions = level.get("conditions", [])
        guard      = level.get("guard", "")

        lines.append(f"\n{score_key}점 ({label_str}):")
        for cond in conditions:
            lines.append(f"- {cond}")
        if guard:
            lines.append(f"※ {guard}")

    return "\n".join(lines)


def _build_category_action_guide(category_info: str) -> tuple[str, list[str]]:
    """
    five_point_rules.json > category_rules 에서 카테고리에 맞는
    특수 규칙 블록을 생성한다. 해당 없으면 빈 문자열 반환.
    
    [v1.1] 모든 매칭된 카테고리 규칙을 결합하여 반환
    - 카테고리 중복 시 모든 규칙을 포함 (가장 유리한 규칙 우선 적용)
    
    [v1.2] 매칭된 카테고리 정보도 함께 반환
    - 반환값: (프롬프트_텍스트, 매칭된_카테고리_목록)
    """
    if not category_info:
        return "", []

    rules         = _load_rules()
    category_data = rules.get("category_rules", {})
    ci            = category_info.lower()

    matched_blocks = []
    matched_categories = []

    for cat_key, cat in category_data.items():
        if cat_key.startswith("_"):
            continue
        if not isinstance(cat, dict):
            continue
        if cat_key == "식품":
            desc = cat.get("desc", "")
            if desc:
                matched_blocks.append(f"\n[카테고리 특수 규칙 — 식품]")
                matched_blocks.append(f"- {desc}")
            continue
        keywords = cat.get("keywords", [])
        if any(kw in ci for kw in keywords):
            cat_label  = cat_key.replace("_", "/")
            rule_lines = cat.get("rules", [])
            block = [f"\n[카테고리 특수 규칙 — {cat_label}]"]
            block += [f"- {r}" for r in rule_lines]
            matched_blocks.append("\n".join(block))
            matched_categories.append(cat_label)

    if matched_blocks:
        return "\n".join(matched_blocks) + "\n", matched_categories

    return "", []


def _build_enforcement_section(enforcement: dict) -> str:
    """enforcement_rules 섹션을 프롬프트 텍스트로 변환한다."""
    sep   = "━" * 29
    lines = [sep, "[강제 룰 — 반드시 준수]", sep]

    independence = enforcement.get("independence", "")
    if independence:
        lines += ["\n1. 항목 독립성:", f"   - {independence}"]

    action_limits = enforcement.get("action_limits", [])
    if action_limits:
        lines.append("\n2. 행동성 판단 제한:")
        for al in action_limits:
            lines.append(f"   - {al}")

    zero_limits = enforcement.get("zero_score_limits", {})
    if zero_limits:
        lines.append("\n3. 0점 기준 제한:")
        label_map = {
            "quality_0":    "품질 0점",
            "visibility_0": "가시성 0점",
            "action_0":     "행동성 0점",
        }
        for field, desc in zero_limits.items():
            prefix = label_map.get(field, field)
            lines.append(f"   - {prefix}: {desc}")

    lines += [
        "\n4. 판단 방식:",
        '   - 모든 판단은 제공된 "프레임 이미지 기준"으로만 수행한다',
        "   - 영상 길이나 전체 맥락은 점수에 영향을 주지 않는다",
    ]

    score_calc = enforcement.get("score_calculation", "")
    if score_calc:
        lines += ["\n[최종 점수 계산]", score_calc]

    return "\n".join(lines)


def _build_output_section(output_format: dict) -> str:
    """output_format 섹션을 프롬프트 텍스트로 변환한다."""
    fields        = output_format.get("fields", {})
    example       = output_format.get("example", {})
    reason_fields = fields.get("reason", {})

    reason_lines = "\n".join(
        f'    "{k}": "{v}"' for k, v in reason_fields.items()
    )

    applied_cat_desc = fields.get("applied_category_rules", "적용된 카테고리 규칙 목록 (배열)")

    lines = [
        "[출력 형식 — JSON만 출력, 마크다운 금지]",
        "{",
        f'  "quality": {fields.get("quality", "0~2 사이 정수")},',
        f'  "visibility": {fields.get("visibility", "0~4 사이 정수")},',
        f'  "action": {fields.get("action", "0~4 사이 정수")},',
        f'  "total_score": {fields.get("total_score", "합계 정수")},',
        f'  "applied_category_rules": "{applied_cat_desc}",',
        '  "reason": {',
        reason_lines,
        "  }",
        "}",
    ]

    if example:
        lines += [
            "\n[출력 예시]",
            json.dumps(example, ensure_ascii=False, indent=2),
        ]

    return "\n".join(lines)


# ==========================================
# 10점제 채점 프롬프트 — JSON 기반 동적 조립
# ==========================================
def build_5point_prompt(category_info: str = "") -> tuple[str, str, list[str]]:
    """
    five_point_rules.json 을 읽어 10점제 채점 프롬프트를 조립한다.
    채점 기준 변경은 JSON 파일만 수정하면 됩니다.
    
    [v1.2] 매칭된 카테고리 정보도 함께 반환
    - 반환값: (system_role, user_prompt, matched_categories)
    """
    rules = _load_rules()

    system_role = rules.get("system_role", "")

    core_rules  = rules.get("core_rules", [])
    score_items = rules.get("score_items", {})
    enforcement = rules.get("enforcement_rules", {})
    output_fmt  = rules.get("output_format", {})

    core_block = "[핵심 규칙]\n" + "\n".join(f"- {r}" for r in core_rules)

    ITEM_ORDER = ["quality", "visibility", "action"]
    score_blocks = []
    for idx, key in enumerate(ITEM_ORDER, start=1):
        item = score_items.get(key)
        if not item:
            continue
        item_copy = dict(item)
        item_copy["label"] = f"{idx}. {item.get('label', key)}"
        score_blocks.append(_build_score_item_section(key, item_copy))

    criteria_block = "[평가 기준]\n총 3개 항목으로 평가한다.\n\n" + "\n\n".join(score_blocks)

    category_block, matched_categories = _build_category_action_guide(category_info)
    category_block = category_block.strip() or None

    enforcement_block = _build_enforcement_section(enforcement)

    output_block = _build_output_section(output_fmt)

    user_prompt = "\n\n".join(filter(None, [
        "동영상 리뷰를 10점 만점으로 채점하고 반드시 JSON 객체로만 응답하세요.",
        core_block,
        criteria_block,
        category_block,
        enforcement_block,
        output_block,
    ]))

    return system_role, user_prompt, matched_categories


# ==========================================
# 점수 방어 검증
# ==========================================
def validate_and_fix_scores(data: dict) -> dict:
    """AI가 반환한 점수를 검증하고 범위 초과 시 클램핑"""
    score_ranges = _get_score_ranges()
    fixed = dict(data)

    for field, (lo, hi) in score_ranges.items():
        val = fixed.get(field)
        if not isinstance(val, int):
            try:
                val = int(val)
            except (TypeError, ValueError):
                val = lo
        fixed[field] = max(lo, min(hi, val))

    # total_score 재계산 (AI 계산 오류 방지)
    fixed["total_score"] = sum(fixed.get(f, 0) for f in score_ranges)
    return fixed


# ==========================================
# 카테고리 예외 규칙 — 최소 행동성 점수 보장 (v1.3)
# ==========================================
def _extract_min_action_from_rule_text(rule_text: str) -> int:
    """
    규칙 텍스트에서 최소 행동성 점수를 추출한다.
    
    매칭 패턴:
      - "행동성 2점"          → 2
      - "행동성 2점 이상"      → 2
      - "행동성 3점 수준"      → 3
      - "행동성 긍정 반영 (2점 이상)" → 2
      - "최소 2점"             → 2
      - "기본 점수(2점)"       → 2
      - "최고점 수준(4점)"     → 4
      - "→ 행동성 2점"         → 2
    
    여러 점수가 나오면 가장 높은 값을 반환 (리뷰어에게 유리하게).
    점수를 찾지 못하면 0 반환 (보정 없음).
    """
    scores = []
    
    # 패턴1: "행동성 X점" (뒤에 "이상", "수준", 공백, 쉼표 등이 올 수 있음)
    for m in re.finditer(r'행동성\s*(\d)점', rule_text):
        scores.append(int(m.group(1)))
    
    # 패턴2: "최소 X점"
    for m in re.finditer(r'최소\s*(\d)점', rule_text):
        scores.append(int(m.group(1)))
    
    # 패턴3: 괄호 안 점수 "(X점)", "(X점 이상)", "(X점 수준)" 등
    for m in re.finditer(r'\((\d)점', rule_text):
        scores.append(int(m.group(1)))
    
    # 패턴4: "→ 행동성 X점" 또는 "→ X점" (applied_category_rules 형식)
    for m in re.finditer(r'→\s*행동성?\s*(\d)점', rule_text):
        scores.append(int(m.group(1)))
    
    return max(scores) if scores else 0


def _build_category_min_action_map() -> dict:
    """
    five_point_rules.json 의 카테고리 규칙에서
    각 규칙 텍스트의 최소 행동성 점수를 미리 추출하여 맵을 구성한다.
    
    반환 예시:
    {
        "의류/패션": {
            "비닐 포장 상태라도 손으로 옷감의 신축성(고무줄 등)을 당겨보거나 만져보는 찰나의 장면이 있다면 행동성 2점 수준으로 판단한다": 2,
            "소재 질감을 직접 손으로 보여주는 장면은 행동성 긍정 반영 (2점 이상)": 2,
            "착용 후 전신 촬영이 있으면 행동성 긍정 반영 (3점 이상)": 3,
        },
        ...
    }
    """
    rules = _load_rules()
    category_data = rules.get("category_rules", {})
    result = {}
    
    for cat_key, cat in category_data.items():
        if cat_key.startswith("_") or not isinstance(cat, dict):
            continue
        if cat_key == "식품":
            continue
        
        cat_label = cat_key.replace("_", "/")
        rule_texts = cat.get("rules", [])
        cat_map = {}
        
        for rule_text in rule_texts:
            min_score = _extract_min_action_from_rule_text(rule_text)
            if min_score > 0:
                cat_map[rule_text] = min_score
        
        if cat_map:
            result[cat_label] = cat_map
    
    return result


# 카테고리별 최소 점수 맵 (모듈 로드 시 1회만 생성)
_category_min_action_map: dict = {}


def _get_category_min_action_map() -> dict:
    """카테고리 최소 점수 맵을 반환 (지연 초기화)"""
    global _category_min_action_map
    if not _category_min_action_map:
        _category_min_action_map = _build_category_min_action_map()
    return _category_min_action_map


def _is_negative_rule(rule_text: str) -> bool:
    """
    규칙 텍스트가 '부정형'(해당 규칙이 적용되지 않았음을 의미)인지 판단한다.
    
    AI가 가끔 카테고리 규칙을 언급하면서도 "없음", "미적용", "미확인" 등으로
    실제로는 규칙이 적용되지 않았음을 표시하는 경우가 있다.
    이런 규칙은 보정 대상에서 제외해야 한다.
    
    부정형 판단 기준:
      - "없음" + "→ 행동성 1점" 조합 → 규칙 미적용
      - "미적용", "미확인", "해당 없음" 포함 → 규칙 미적용
      - 단, "없음"만 있다고 무조건 부정형은 아님 (예: "포장 없이"는 정상)
    
    판단 방식: 규칙 텍스트의 → 이후 점수가 1점 이하이면 부정형으로 간주.
    카테고리 규칙이 실제로 적용되었다면 최소 2점 이상이어야 하기 때문.
    """
    if not rule_text:
        return False
    
    # 명시적 부정형 키워드
    negative_keywords = ["미적용", "미확인", "해당 없음", "해당없음"]
    if any(kw in rule_text for kw in negative_keywords):
        return True
    
    # "없음" + "→ 행동성 1점" 패턴 → 규칙 미적용을 의미
    if "없음" in rule_text:
        # → 이후 점수가 1점이면 부정형
        m = re.search(r'→\s*행동성?\s*(\d)점', rule_text)
        if m and int(m.group(1)) <= 1:
            return True
    
    return False


def enforce_category_min_action(data: dict) -> dict:
    """
    카테고리 예외 규칙이 적용된 경우, 행동성 점수가 규칙의 최소 점수 미만이면
    최소 점수로 상향 조정한다.
    
    [v1.4 수정사항]
      - 이전: 부정형 규칙("없음/미적용")만 있는 경우 → 보정 없음 (문제 원인)
      - 변경: 예외카테고리 매칭 + AI가 0/1점 부여한 경우 → 무조건 최소 2점 보정
      - 이유: AI가 "없음→행동성 1점" 등으로 처리해도, 카테고리 자체가 예외카테고리면
            기본 2점은 보장되어야 함 (규칙 내용 중 가장 낮은 점수 = 2점)
    
    핵심 원칙:
      - AI가 applied_category_rules에 규칙을 명시했다 = 해당 카테고리 영상임을 인정
      - 예외카테고리 영상은 무조건 행동성 2점 이상을 받아야 함
      - 긍정형 규칙(구체적 장면 인식)이 있으면 그 점수 우선 적용
      - 부정형("없음/미적용")만 있어도 예외카테고리면 기본 2점 보정
    
    적용 방식:
      - AI 응답의 applied_category_rules 각 항목에서 점수 추출
      - 긍정형 규칙에서 점수 추출 성공 → 그 최댓값 적용
      - 긍정형 없고 부정형만 있으며 action ≤ 1 → 예외카테고리 기본 2점 보정
      - total_score 재계산
    """
    fixed = dict(data)
    applied_rules = fixed.get("applied_category_rules", [])
    
    # 규칙이 적용되지 않은 영상 → AI 점수 그대로 (2단계에서 처리)
    if not applied_rules or not isinstance(applied_rules, list):
        return fixed
    
    positive_rules = []
    max_min_score_from_positive = 0
    has_negative_only = False
    
    for rule_text in applied_rules:
        if not isinstance(rule_text, str):
            continue
        
        if _is_negative_rule(rule_text):
            log(f"  [카테고리 규칙] 부정형 규칙 발견: {rule_text[:80]}")
            has_negative_only = True
            continue
        
        positive_rules.append(rule_text)
        min_score = _extract_min_action_from_rule_text(rule_text)
        if min_score > max_min_score_from_positive:
            max_min_score_from_positive = min_score
    
    current_action = fixed.get("action", 0)
    if not isinstance(current_action, int):
        try:
            current_action = int(current_action)
        except (TypeError, ValueError):
            current_action = 0
    
    # ── Case A: 긍정형 규칙이 있고 점수 추출됨 → 기존 로직대로 상향 ──
    if max_min_score_from_positive > 0 and current_action < max_min_score_from_positive:
        old_action = current_action
        fixed["action"] = max_min_score_from_positive
        
        score_ranges = _get_score_ranges()
        fixed["total_score"] = sum(fixed.get(f, 0) for f in score_ranges)
        
        reason = fixed.get("reason", {})
        if isinstance(reason, dict):
            orig_reason = reason.get("action_reason", "")
            reason["action_reason"] = (
                f"{orig_reason}\n[카테고리 규칙 보정] 행동성 {old_action}점 → {max_min_score_from_positive}점 "
                f"(적용 규칙: {'; '.join(positive_rules)})"
            ).strip()
            fixed["reason"] = reason
        
        log(f"  [카테고리 규칙 보정] 행동성 {old_action} → {max_min_score_from_positive} "
            f"(규칙: {positive_rules})")
        
        return fixed
    
    # ── Case B: 부정형 규칙만 존재하는 경우 → 보정 없음 ──
    # AI가 "없음/미적용" 등 부정형으로 판단했다는 것은
    # 해당 영상에 카테고리 규칙이 실제로 적용되지 않는다는 AI의 판단이므로
    # 강제 보정 없이 AI 점수를 그대로 유지한다.
    if has_negative_only:
        log(f"  [카테고리 규칙] 부정형 규칙만 존재 → AI 점수({current_action}점) 유지 (보정 없음)")

    return fixed


def enforce_category_min_action_from_match(data: dict, matched_categories: list) -> dict:
    """
    [v1.5 변경] 키워드 매칭만으로 보정하는 로직을 비활성화.

    이전(v1.4)에는 프롬프트에 카테고리 규칙이 포함(키워드 매칭)되었는데
    AI가 applied_category_rules를 비어있거나 부정형으로 반환하면
    코드에서 강제로 최소 2점을 보정했음.

    문제점:
      - AI가 해당 영상에 카테고리 규칙이 실제로 적용되지 않는다고 판단(1점)했는데도
        키워드 매칭만으로 2점으로 보정되는 오류 발생
      - 예: "카테고리 규칙 적용 → 행동성 2점 (코드 보정)"이 붙으면서 1점 → 2점 상향

    원칙:
      - 보정은 반드시 AI가 applied_category_rules에 긍정형 규칙을 명시한 경우에만 수행
      - 키워드 매칭(matched_categories)은 프롬프트 구성에만 사용, 점수 보정에 사용 안 함
      - 1단계 enforce_category_min_action(Case A)에서 긍정형 규칙 보정을 처리함
    """
    # 키워드 매칭만으로 보정하지 않음 — AI 판단 그대로 반환
    return dict(data)


# ==========================================
# AI API 호출
# ==========================================
async def call_5point_ai_api(grid_b64_list, system_prompt, user_prompt):
    loop = asyncio.get_event_loop()
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in grid_b64_list
    ]
    response = await loop.run_in_executor(
        None,
        lambda: openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}] + image_messages},
            ],
            max_completion_tokens=1500,
            temperature=0.0,
            response_format={"type": "json_object"},
        ),
    )
    res_clean = (
        response.choices[0].message.content.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    data = json.loads(res_clean)
    return data


# ==========================================
# 프레임 그리드 생성
# ==========================================
def make_grid_b64(frames_bgr, border_size=2):
    if not frames_bgr:
        return []
    n    = len(frames_bgr)
    cols = 2 if n <= 4 else (3 if n <= 6 else 4)
    rows = (n + cols - 1) // cols
    h, w, c = frames_bgr[0].shape
    grid_h   = h * rows + border_size * (rows - 1)
    grid_w   = w * cols + border_size * (cols - 1)
    grid_img = np.full((grid_h, grid_w, c), 255, dtype=np.uint8)

    for i, frame in enumerate(frames_bgr):
        r, c_idx = i // cols, i % cols
        y_start  = r * (h + border_size)
        x_start  = c_idx * (w + border_size)
        grid_img[y_start : y_start + h, x_start : x_start + w] = frame

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, buf = cv2.imencode(".jpg", grid_img, encode_param)
    return [base64.b64encode(buf).decode("utf-8")]


# ==========================================
# AI 10점제 채점
# ==========================================
async def analyze_5point_with_ai(frames_bgr, category_info: str = ""):
    system_prompt, user_prompt, matched_categories = build_5point_prompt(category_info)
    grid_b64 = make_grid_b64(frames_bgr, border_size=2)

    for attempt in range(3):
        try:
            result = await call_5point_ai_api(grid_b64, system_prompt, user_prompt)
            validated = validate_and_fix_scores(result)
            if "applied_category_rules" not in validated:
                validated["applied_category_rules"] = []
            if not isinstance(validated.get("applied_category_rules"), list):
                validated["applied_category_rules"] = []
            
            # 카테고리 규칙 보정 (2단계)
            # 1단계: AI가 규칙을 인식한 경우 → 규칙 최소 점수 보장
            validated = enforce_category_min_action(validated)
            
            # 2단계: 프롬프트에 카테고리 규칙이 포함되었는데 AI가 무시한 경우
            #        → 코드에서 직접 카테고리 규칙의 최소 점수 적용
            validated = enforce_category_min_action_from_match(validated, matched_categories)
            
            return validated
        except Exception as e:
            err_msg = str(e).lower()
            if "content_filter" in err_msg or "policy" in err_msg:
                log("  [AI 텍스트 필터 통과됨] 스킵합니다.")
                return None
            log(f"  [AI API 에러 발생] 시도 {attempt + 1}/3: {e}")
            await asyncio.sleep(2 ** attempt)

    return None


# ==========================================
# 3단계 10점제 평가 실행 (외부에서 호출하는 메인 함수)
# ==========================================
async def run_5point_phase(frames_bgr: list, category_info: str = "") -> dict:
    """3단계 10점제 평가 실행"""
    result = await analyze_5point_with_ai(frames_bgr, category_info)

    if not result:
        return {
            "quality":    0,
            "visibility": 0,
            "action":     0,
            "total_score": 0,
            "applied_category_rules": [],
            "reason": {
                "quality_reason":    "AI 평가 실패",
                "visibility_reason": "",
                "action_reason":     "",
                "overall_reason":    "AI API 호출 오류",
            },
            "eval_stage": "10점제(API실패)",
        }

    return {**result, "eval_stage": "10점제"}