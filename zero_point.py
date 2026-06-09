import json
import asyncio
import numpy as np
import cv2
from collections import Counter

from config import openai_client, GPT_MODEL, log
from video_utils import create_frame_grid_b64


# ==========================================
# CV 화면 재촬영 감지 — 내부 헬퍼 함수들
# ==========================================
def _has_uniform_dark_border(gray, border_ratio=0.06):
    h, w       = gray.shape
    bh         = max(4, int(h * border_ratio))
    bw         = max(4, int(w * border_ratio))
    top        = float(np.mean(gray[:bh, :]))
    bottom     = float(np.mean(gray[h-bh:, :]))
    left       = float(np.mean(gray[:, :bw]))
    right      = float(np.mean(gray[:, w-bw:]))
    inner      = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    center_mean = float(np.mean(inner))
    border_mean = (top + bottom + left + right) / 4.0
    all_dark    = all(v < 60 for v in [top, bottom, left, right])
    contrast    = center_mean - border_mean
    return all_dark and contrast > 30


def _has_moire_pattern(gray):
    h, w      = gray.shape
    f         = np.fft.fft2(gray.astype(np.float32))
    fshift    = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    total_energy = float(np.sum(magnitude))
    if total_energy < 1e-6:
        return False
    cy, cx       = h // 2, w // 2
    Y, X         = np.ogrid[:h, :w]
    dist         = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r        = min(cx, cy)
    mid_band_mask = (dist >= max_r * 0.15) & (dist <= max_r * 0.40)
    mid_energy    = float(np.sum(magnitude[mid_band_mask]))
    return (mid_energy / total_energy) > 0.35


def _has_statusbar_pattern(gray):
    h, w           = gray.shape
    bar_h          = max(6, int(h * 0.05))
    region         = gray[:bar_h, :]
    edges          = cv2.Canny(region, 30, 90)
    edge_density   = float(np.sum(edges > 0)) / edges.size
    std_brightness = float(np.std(region))
    return edge_density > 0.25 and std_brightness < 40


def _has_screen_in_screen(gray):
    h, w        = gray.shape
    frame_area  = h * w
    blur        = cv2.GaussianBlur(gray, (5, 5), 0)
    edges       = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(cnt)
        if not (frame_area * 0.30 < area < frame_area * 0.90):
            continue
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [approx], 255)
        inner_pixels = gray[mask == 255]
        if len(inner_pixels) < 100:
            continue
        if float(np.mean(inner_pixels)) > 80 and float(np.std(inner_pixels)) < 60:
            return True
    return False


# ==========================================
# CV 기반 화면 재촬영 감지 (통합)
# ==========================================
def detect_screen_recapture_cv(frames_bgr: list):
    """OpenCV 기반 화면 재촬영 감지 v2.0 — 프레임별 투표 방식"""
    if not frames_bgr:
        return None

    votes, detected_signals = [], []
    for frame in frames_bgr:
        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_signals = []
        if _has_uniform_dark_border(gray): frame_signals.append("dark_border")
        if _has_screen_in_screen(gray):    frame_signals.append("screen_in_screen")
        if _has_moire_pattern(gray):       frame_signals.append("moire_pattern")
        if _has_statusbar_pattern(gray):   frame_signals.append("statusbar")

        strong     = [s for s in frame_signals if s in ("dark_border", "screen_in_screen")]
        weak       = [s for s in frame_signals if s in ("moire_pattern", "statusbar")]
        frame_vote = bool(strong) or (len(weak) >= 2)
        votes.append(frame_vote)
        if frame_vote:
            detected_signals.extend(frame_signals)

    if not votes:
        return None
    vote_ratio = sum(votes) / len(votes)
    if vote_ratio >= 0.40:
        main_signal = Counter(detected_signals).most_common(1)[0][0] if detected_signals else "unknown"
        confidence  = min(0.95, 0.60 + vote_ratio * 0.35)
        return (main_signal, confidence)
    return None


def check_frame_consistency(frames_bgr: list):
    """프레임 간 급격한 변화(화면 전환) 감지"""
    if len(frames_bgr) < 2:
        return None
    inconsistencies = []
    for i in range(1, len(frames_bgr)):
        diff       = cv2.absdiff(frames_bgr[i-1], frames_bgr[i])
        diff_ratio = np.sum(diff > 30) / diff.size
        if diff_ratio > 0.3:
            inconsistencies.append(i)
    return inconsistencies if inconsistencies else None


# ==========================================
# 부적합 프롬프트 생성
# ==========================================
def build_zero_point_prompt(product_info_str: str, no_info_note: str,
                             cv_hint: str, rules: dict,
                             bundle_deal_mode: str = "normal",
                             review_text: str = "") -> str:
    """zero_point_rules.json 기반 프롬프트 생성"""
    items         = rules.get("zero_point_items", {})
    version       = rules.get("_version", "1.15")

    prompt = f"""[0점 감지 테스트 - v{version}]
{product_info_str}
{no_info_note}
{cv_hint}
이 영상이 0점 처리 대상인지 판단하세요.
아래 5가지 항목 중 하나라도 해당하면 0점입니다.

[0점 감지 항목]
"""
    for idx, (key, item) in enumerate(items.items(), 1):
        desc     = item.get("desc", "")
        criteria = item.get("criteria", "")
        prompt  += f"{idx}. {key}: {desc}\n"
        if criteria:
            prompt += f"   ▶ {criteria}\n"
        if "core_principle" in item:
            prompt += f"   ▶ {item['core_principle']}\n"
        if "v1.8_core_rules" in item:
            prompt += "   ▶ [v1.8 핵심 규칙 - 절대 준수 사항]:\n"
            for rule_key, rule_val in item["v1.8_core_rules"].items():
                prompt += f"   **{rule_val}**\n"
        if "v1.8_forced_rules" in item:
            fr = item["v1.8_forced_rules"]
            prompt += f"   ▶ {fr.get('desc', '')}:\n"
            for case in fr.get("cases", []):
                prompt += f"   - {case}\n"
        if "v1.7_same_category_options" in item:
            opt = item["v1.7_same_category_options"]
            prompt += f"   ▶ {opt.get('desc', '')}:\n"
            for rule in opt.get("rules", []):
                prompt += f"   - {rule}\n"
        if "packaging_handling" in item:
            pkg = item["packaging_handling"]
            prompt += f"   ▶ {pkg.get('desc', '')}:\n"
            for rule in pkg.get("rules", []):
                prompt += f"   - {rule}\n"
        if "priority" in item:
            pri = item["priority"]
            prompt += f"   ▶ {pri.get('desc', '[판단 우선순위]')}:\n"
            for pkey, pval in pri.items():
                if pkey != "desc":
                    prompt += f"   {pkey}: {pval}\n"
        if "v1.8_priority" in item:
            pri = item["v1.8_priority"]
            prompt += f"   ▶ {pri.get('desc', '[판단 우선순위]')}:\n"
            for pkey, pval in pri.items():
                if pkey != "desc":
                    prompt += f"   {pkey}: {pval}\n"
        if "v1.8_exclusions" in item:
            exc = item["v1.8_exclusions"]
            prompt += f"   ▶ {exc.get('desc', '')}:\n"
            for case in exc.get("cases", []):
                prompt += f"   - {case}\n"
        if "v1.8_enforcement" in item:
            enf = item["v1.8_enforcement"]
            prompt += f"   ▶ {enf.get('desc', '')}:\n"
            for rule in enf.get("rules", []):
                prompt += f"   - {rule}\n"
        if "detection_types" in item:
            dt = item["detection_types"]
            prompt += "   ▶ [진짜 화면 재촬영만 true] 아래 징후가 모두 명확히 보여야 true:\n"
            for dtkey, dtval in dt.items():
                prompt += f"   {dtkey.upper()}. {dtval.get('desc', '')}:\n"
                for sign in dtval.get("signs", []):
                    prompt += f"      · {sign}\n"
        examples = item.get("examples", {})
        if examples:
            if "true" in examples:
                prompt += "   ▶ true 예시:\n"
                for ex in examples["true"]:
                    prompt += f"   - {ex}\n"
            if "false" in examples:
                prompt += "   ▶ false 예시:\n"
                for ex in examples["false"]:
                    prompt += f"   - {ex}\n"
            if "pass" in examples:
                prompt += "   ▶ pass 예시:\n"
                for ex in examples["pass"]:
                    prompt += f"   - {ex}\n"
        cautions = item.get("cautions", [])
        if cautions:
            for c in cautions:
                prompt += f"   - 주의: {c}\n"
        
        if "category_specific_false_cases" in item:
            csfr = item["category_specific_false_cases"]
            prompt += f"\n   ▶ {csfr.get('desc', '')}:\n"
            for cat_key, cases in csfr.items():
                if cat_key == "desc":
                    continue
                prompt += f"   [{cat_key}]:\n"
                for case in cases:
                    prompt += f"   - {case}\n"
        
        if "false_patterns_to_prevent" in item:
            fpp = item["false_patterns_to_prevent"]
            prompt += f"\n   ▶ {fpp.get('desc', '')}\n"
            for pattern_key, pattern_cases in fpp.items():
                if pattern_key == "desc":
                    continue
                prompt += f"   [{pattern_key}]:\n"
                for case in pattern_cases:
                    prompt += f"   - {case}\n"
        
        if "frame_judgment_rule" in item:
            fjr = item["frame_judgment_rule"]
            prompt += f"\n   ▶ {fjr.get('desc', '')}:\n"
            prompt += f"   **{fjr.get('rule', '')}**\n"
            prompt += f"   - 이유: {fjr.get('reason', '')}\n"
            prompt += f"   - 원칙: {fjr.get('principle', '')}\n"
            fjr_examples = fjr.get("examples", {})
            if "false_case" in fjr_examples:
                prompt += "   ▶ false (정상) 케이스:\n"
                for ex in fjr_examples["false_case"]:
                    prompt += f"   - {ex}\n"
            if "true_case" in fjr_examples:
                prompt += "   ▶ true (다른 상품) 케이스:\n"
                for ex in fjr_examples["true_case"]:
                    prompt += f"   - {ex}\n"
            fjr_enf = fjr.get("enforcement", [])
            if fjr_enf:
                prompt += "   ▶ [강제 사항]:\n"
                for enf in fjr_enf:
                    prompt += f"   - {enf}\n"
        
        prompt += "\n"

    prompt += """[중요 - 보수적 판단 강화]
1. is_wrong_product 판단:
   - 카테고리가 같으면 기본적으로 false로 판단하세요.
   - 색상·사이즈·용량 차이만 있으면 false로 판단하세요.
   - 같은 카테고리 내 다른 옵션은 같은 상품군으로 판단하세요.
   - 명백히 다른 카테고리/브랜드가 확실한 경우에만 true로 판단하세요.
   - 의심스러우면 무조건 false로 판단하세요!

2. is_unverifiable 판단:
   - 영상 일부라도 상품이 어느 정도 식별된다면 false로 판단하세요.
   - 포장 상태만 보여도 카테고리와 일치하면 false로 판단하세요.
   - 상품이 전혀 보이지 않거나 완전히 식별 불가능한 경우에만 true로 판단하세요.
   - 의심스러우면 무조건 false로 판단하세요!

3. is_wrong_product와 is_unverifiable 판단 시 반드시 카테고리 정보를 활용하세요.

[특별 케이스 처리 규칙 - 절대 준수!]
1. 택배 상자/송장만 보이는 경우:
   → is_delivery_box_only = true, is_wrong_product = false, is_unverifiable = false
   → 무조건 pass 처리!

2. 포장지/박스/비닐만 보이는 경우:
   → 카테고리가 일치하면 is_wrong_product = false, is_unverifiable = false
   → 포장 상태만 보여도 정상으로 간주!

3. 착용샷/사용샷이 보이는 경우:
   → 카테고리가 일치하면 is_wrong_product = false
   → 착용/사용 장면이 있으면 정상으로 간주!

4. 의심스러운 경우:
   → 무조건 false로 판단하세요!
   → 오판 방지가 최우선입니다!

"""

    prompt += """
[★★모음전/세트/다옵션 상품 절대 면책 규칙 (최우선 적용)★★]
1. 구매 옵션(p_opt) 정보가 존재하거나 상품명에 "모음", "종", "세트", "골라담기", "선택", "패키지"가 포함된 경우, **공식 상품 대표 이미지와의 일치 여부는 완전히 무시하십시오.**
2. 영상 속 상품이 '구매 옵션'에 묘사된 단일 상품이거나, 대카테고리에 속하는 상품이라면 무조건 `is_wrong_product = false` 입니다.
3. [절대 금지 사유] "대표 이미지와 다르다", "여러 상품이 묶여있는 이미지인데 하나만 나온다"는 이유로 true 판정을 내리는 것을 엄격히 금지합니다.

[★★고객 리뷰 본문 기반 면책 (최우선 적용)★★]
제공된 '고객 리뷰 본문'에 다음 맥락이 하나라도 있으면 시각적 차이를 무시하고 무조건 is_wrong_product = false 로 처리하세요.
1. 이전 구매 상품 비교 / 2. 사은품, 덤, 서비스 / 3. 농산물 상태 변화 / 4. 혼합 배송 언급

[출력 형식]
JSON으로만 출력:
{"is_wrong_product": false, "wrong_product_reason": "", "is_irrelevant": false, "is_delivery_box_only": false, "is_screen_recapture": false, "screen_recapture_type": "", "is_unverifiable": false, "unverifiable_reason": ""}"""

    return prompt


# ==========================================
# AI 부적합 감지
# ==========================================
async def detect_zero_point_features(frames_bgr: list, p_img_b64: str,
                                      p_name: str, p_opt: str, category_info: str,
                                      zero_rules: dict, bundle_deal_mode: str = "normal",
                                      review_text: str = "") -> dict | None:
    """2단계 부적합 AI 감지"""
    cv_result   = detect_screen_recapture_cv(frames_bgr)
    cv_hint     = ""
    if cv_result:
        cv_type, cv_conf = cv_result
        cv_hint = f"\n[CV 사전 감지 결과] 화면 재촬영 징후 발견: {cv_type} (신뢰도 {cv_conf:.0%})\n"
    else:
        cv_hint = "\n[CV 사전 감지 결과] 화면 재촬영 징후 없음 (일반 상품 리뷰 영상으로 판단됨)\n"

    inconsistencies = check_frame_consistency(frames_bgr)
    if inconsistencies:
        cv_hint += f"[CV 사전 감지 결과] 급격한 화면 전환 감지: {len(inconsistencies)}개 구간\n"

    img_msgs = []
    if p_img_b64:
        clean_b64 = p_img_b64.replace('\n', '').replace('\r', '')
        img_msgs.append({"type": "text", "text": "공식 상품 대표 이미지 (참고용. 모음딜/다옵션의 경우 무시할 것):"})
        img_msgs.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}", "detail": "high"}})

    img_msgs.append({"type": "text", "text": "유저 리뷰 영상 프레임:"})
    grid_b64 = create_frame_grid_b64(frames_bgr, cell_w=320, cell_h=240, quality=80)
    if grid_b64:
        clean_grid = grid_b64.replace('\n', '').replace('\r', '')
        img_msgs.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{clean_grid}", "detail": "high"}})

    product_info_lines = []
    
    # [v1.15] p_opt(옵션)를 가장 위로 올리고 강력한 지시 추가
    if p_opt:
        product_info_lines.append(f"🔴[가장 중요한 판단 기준] 고객이 실제 선택한 구매 옵션: '{p_opt}'🔴")
        product_info_lines.append("-> [명령] 대표 이미지나 메인 상품명과 다르게 생겼더라도, 위 '구매 옵션'에 기재된 상품이 영상에 보인다면 무조건 100% 정상(false)입니다. 절대로 대표 이미지를 기준으로 오판하지 마세요.")
    
    if p_name:
        product_info_lines.append(f"공식 상품명(메인): '{p_name}'")
    else:
        product_info_lines.append("공식 상품명: 알 수 없음")
        
    if category_info:
        product_info_lines.append(f"상품 카테고리: {category_info}")
    
    if review_text:
        product_info_lines.append(f"★[매우 중요] 고객 리뷰 본문: '{review_text[:1000]}'")
        product_info_lines.append("-> (리뷰 본문에 사은품/덤/이전 상품 비교/색상 변화 등이 언급되었다면 영상과 상품이 달라 보여도 무조건 정상(false) 처리하세요.)")
    
    product_info_str = "\n".join(product_info_lines)

    no_info_note = ""
    if not p_name and not p_img_b64:
        no_info_note = ("※ 상품명과 상품 이미지가 없습니다. 카테고리 정보와 영상 내용만으로 반드시 판단하세요. "
                        "정보 부족을 이유로 판단을 회피하지 마세요.\n")

    prompt = build_zero_point_prompt(product_info_str, no_info_note, cv_hint, zero_rules, bundle_deal_mode, review_text)

    system_role = zero_rules.get(
        "system_role",
        "당신은 전자상거래 리뷰 영상의 0점 감지를 수행하는 한국어 전문 심사위원입니다. ONLY output a valid JSON object. No markdown, no explanation."
    )

    for attempt in range(3):
        try:
            def _call(sm=system_role, pm=prompt, im=img_msgs):
                return openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": sm},
                        {"role": "user",   "content": [{"type": "text", "text": pm}] + im}
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=2000,
                    temperature=0.0
                )
            res = await asyncio.get_event_loop().run_in_executor(None, _call)
            return json.loads(res.choices[0].message.content.strip())
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(3)
                continue
            log(f"[부적합 LLM 에러] {e}")
    return None


# ==========================================
# 부적합 판정 로직
# ==========================================
def analyze_zero_point(features: dict) -> tuple[str, str, list]:
    """부적합 판정 로직"""
    zero_reasons = []

    if features.get("is_wrong_product", False):
        reason = features.get("wrong_product_reason", "")
        zero_reasons.append("is_wrong_product")
        return "0", f"0점 - 다른 상품: {reason}", zero_reasons

    if features.get("is_irrelevant", False):
        zero_reasons.append("is_irrelevant")
        return "0", "0점 - 상품과 무관한 영상", zero_reasons

    if features.get("is_delivery_box_only", False):
        return "pass", "택배 상자만 노출 (상품 도착 확인)", []

    if features.get("is_screen_recapture", False):
        zero_reasons.append("is_screen_recapture")
        return "0", "0점 - 화면 재촬영", zero_reasons

    if features.get("is_unverifiable", False):
        reason = features.get("unverifiable_reason", "")
        zero_reasons.append("is_unverifiable")
        detail = f" ({reason})" if reason else ""
        return "0", f"0점 - 상품 확인 불가 영상{detail}", zero_reasons

    return "pass", "0점 감지 안됨 (정상 또는 10점제 평가 대상)", []


# ==========================================
# 2단계 부적합 탐지 실행 (외부에서 호출하는 메인 함수)
# ==========================================
async def run_zero_point_phase(frames_bgr: list, duration_sec: float,
                                p_img_b64: str, p_name: str, p_opt: str,
                                category_info: str, review_id: str,
                                member_id: str, zero_rules: dict,
                                review_text: str = "") -> dict:
    """2단계 부적합 탐지 실행
    
    주의: 중복 영상 감지는 Phase 1(precompute_video_groups)에서 사전 수행되므로
          이 함수에서는 중복 감지를 수행하지 않습니다.
    """
    from video_utils import get_bundle_deal_check_mode
    
    bundle_deal_mode = get_bundle_deal_check_mode(p_name)
    
    features = await detect_zero_point_features(
        frames_bgr, p_img_b64, p_name, p_opt, category_info, zero_rules, bundle_deal_mode, review_text
    )
    if not features:
        return {
            "zero_score":    "Check",
            "zero_reason":   "부적합 AI 감지 실패",
            "zero_features": []
        }

    zero_score, zero_reason, zero_features = analyze_zero_point(features)
    return {
        "zero_score":    zero_score,
        "zero_reason":   zero_reason,
        "zero_features": zero_features
    }