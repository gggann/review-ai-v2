"""
video_utils.py — 영상 다운로드·프레임 추출·이미지 처리·중복 감지
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
영상 관련 모든 작업을 담당합니다.
· 로컬/LAN 메타데이터 로드
· 영상 다운로드
· 프레임 추출
· 이미지 base64 변환
· 중복 영상 감지

현재 영상 처리는 로컬 경로와 LAN URL만 사용합니다.
"""

import os
import re
import json
import asyncio
import base64
import random
import io
import numpy as np
import cv2
import aiohttp
import aiofiles
import imagehash
from PIL import Image

from config import (
    log,
    get_metadata_path,
    get_video_path,
)

# ==========================================
# contNo 추출 및 로컬 메타데이터 로드
# ==========================================
def extract_cont_no(page_url: str) -> str | None:
    """URL에서 contNo 파라미터 값을 추출"""
    if not page_url:
        return None
    m = re.search(r'contNo=(\d+)', page_url, re.IGNORECASE)
    return m.group(1) if m else None


async def load_local_metadata(session, cont_no: str) -> dict | None:
    """
    metadata.json 읽기 — 로컬 디스크 또는 LAN HTTP.
    실패 시 None 반환.
    """
    path_or_url = get_metadata_path(cont_no)

    if not path_or_url.startswith("http"):
        # 로컬 디스크에서 직접 읽기
        if os.path.exists(path_or_url):
            try:
                with open(path_or_url, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data and isinstance(data, dict) and "product" in data:
                    return data
            except Exception:
                pass
        return None
    else:
        # LAN HTTP로 요청
        try:
            async with session.get(path_or_url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    data = await r.json()
                    if data and isinstance(data, dict) and "product" in data:
                        return data
        except Exception:
            pass
        return None


# ==========================================
# 리뷰 페이지 메타데이터 수집
# ==========================================

try:
    from lxml import html as lxml_html
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False


async def fetch_review_metadata(session, page_url: str):
    """
    리뷰 페이지에서 영상URL·상품이미지·상품명·옵션·본문 수집.
    로컬 경로 + 외부 URL fallback.
    """
    local_links    = []
    external_links = []
    p_img, p_name, p_opt, p_text = "", "", "", ""

    if page_url.startswith("http://"):
        page_url = "https://" + page_url[7:]

    # ── 로컬 영상 경로 등록 ──
    match = re.search(r'contNo=(\d+)', page_url, re.IGNORECASE)
    if match:
        cont_no = match.group(1)
        local_path = get_video_path(cont_no)
        if os.path.exists(local_path):
            local_links.append(local_path)
            log(f"  [fetch] 로컬 파일 등록: {local_path}")

    headers_req = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.11st.co.kr/",
    }

    # ── HTML 크롤링 — 메타데이터 + 외부 영상 URL 수집 ──
    try:
        async with session.get(
            page_url,
            headers=headers_req,
            timeout=aiohttp.ClientTimeout(total=15),
            allow_redirects=True,
            ssl=False
        ) as resp:
            log(f"  [fetch] {page_url[-60:]} → status={resp.status} finalURL={str(resp.url)[-60:]}")
            if resp.status == 200:
                html_text = await resp.text(errors="replace")

                if LXML_AVAILABLE:
                    try:
                        tree = lxml_html.fromstring(html_text)

                        for attr in ("src", "data-src", "data-url", "data-video-src"):
                            vals = tree.xpath(f'//*[@id="review-detail-video"]/source/@{attr}')
                            for v in vals:
                                v = str(v).strip()
                                if v and v.startswith("http") and not v.startswith("blob"):
                                    external_links.append(v)
                            if external_links:
                                break

                        if not external_links:
                            for attr in ("src", "data-src", "data-url", "data-video-url", "data-video-src"):
                                vals = tree.xpath(f'//*[@id="review-detail-video"]/@{attr}')
                                for v in vals:
                                    v = str(v).strip()
                                    if v and v.startswith("http") and not v.startswith("blob"):
                                        external_links.append(v)
                                if external_links:
                                    break

                        if not external_links:
                            for attr in ("src", "data-src", "data-url"):
                                vals = tree.xpath(f'//source/@{attr}')
                                for v in vals:
                                    v = str(v).strip()
                                    if v and v.startswith("http") and not v.startswith("blob"):
                                        if any(ext in v.lower() for ext in [".mp4", ".m3u8", "video"]):
                                            external_links.append(v)
                                if external_links:
                                    break

                        if external_links:
                            log(f"  [fetch] 외부 영상 URL {len(external_links)}개 수집 "
                                f"(fallback용): {external_links[0][:80]}")

                        # ── 메타데이터 파싱 ──
                        img_nodes = tree.xpath('//img/@src')
                        if img_nodes:
                            for src in img_nodes:
                                src = str(src).strip()
                                if src and src.startswith("http"):
                                    p_img = src
                                    break
                        name_nodes = tree.xpath('//a[@target="blank"]//text()')
                        if name_nodes:
                            p_name = " ".join([n.strip() for n in name_nodes if n.strip()][:3])
                        opt_nodes = tree.xpath('//*[@id="review-detail-option"]//text()')
                        if opt_nodes:
                            p_opt = " ".join([n.strip() for n in opt_nodes if n.strip()])
                        txt_nodes = tree.xpath('//p[@class="cont_text"]//text()')
                        if txt_nodes:
                            p_text = " ".join([n.strip() for n in txt_nodes if n.strip()])[:500]

                    except Exception as xe:
                        log(f"  [fetch] lxml 파싱 오류: {xe}")

                # ── 정규식 fallback ──
                if not external_links:
                    m1 = re.search(r'name=["\']movieUrl["\'][^>]*value=["\']([^"\']+)["\']', html_text)
                    if not m1:
                        m1 = re.search(r'value=["\']([^"\']+\.mp4[^"\']*)["\']', html_text, re.IGNORECASE)
                    if m1:
                        external_links.append(m1.group(1))

                if not external_links:
                    found = re.findall(r'https?://[^\s"\'<>]+?(?:\.mp4|\.m3u8)[^\s"\'<>]*',
                                       html_text, re.IGNORECASE)
                    found = [v for v in found if not v.startswith("blob")]
                    external_links.extend(found)

    except Exception as e:
        log(f"  [fetch] 요청 예외: {type(e).__name__}: {e}")

    # ── 최종 링크 조합: 로컬 먼저, 외부 URL은 fallback ──
    seen  = set()
    final = []
    for url in local_links + external_links:
        url_clean = url.strip('`').strip("'").strip('"')
        if url_clean and not url_clean.startswith("blob") and url_clean not in seen:
            seen.add(url_clean)
            final.append(url_clean)
    final = final[:5]

    log(f"  [fetch] 최종 링크 {len(final)}개: {final}")
    return final, p_img, p_name, p_opt, p_text


# ==========================================
# 이미지 다운로드 및 base64 변환
# ==========================================
async def get_image_b64(session, url: str) -> str:
    if not url: return ""
    if url.startswith("//"): url = "https:" + url
    try:
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10) as r:
            if r.status == 200:
                arr = np.frombuffer(await r.read(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    scale   = 300 / max(img.shape[:2])
                    resized = cv2.resize(img, (max(1, int(img.shape[1]*scale)),
                                               max(1, int(img.shape[0]*scale))))
                    return base64.b64encode(
                        cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                    ).decode("utf-8")
    except Exception:
        pass
    return ""


# ==========================================
# 영상 다운로드
# ==========================================
async def download_video_direct(session, url: str, save_path: str) -> bool:
    """
    단일 URL 다운로드 (3회 재시도).
    로컬 파일 경로가 들어오면 그대로 복사.
    로컬 경로와 LAN URL을 동일한 인터페이스로 처리합니다.
    """
    # 로컬 파일 경로 → 바로 복사, 없으면 실패
    if os.path.isabs(url) or (len(url) >= 2 and url[1] == ':'):
        if os.path.exists(url):
            try:
                import shutil
                await asyncio.to_thread(shutil.copy2, url, save_path)
                return os.path.exists(save_path) and os.path.getsize(save_path) > 1000
            except Exception as e:
                log(f"  [download] 로컬 복사 오류: {e}")
        else:
            log(f"  [download] 로컬 파일 없음: {url[-60:]}")
        return False

    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15"
    ]
    is_lan = not url.startswith("https://")  # 로컬/LAN은 항상 짧은 타임아웃
    for attempt in range(3):
        try:
            headers = {
                "User-Agent": random.choice(agents),
                "Referer":    "https://www.11st.co.kr/",
                "Accept":     "video/webm,video/ogg,video/*;q=0.9,application/ogg;q=0.7,audio/*;q=0.6,*/*;q=0.5",
                "Connection": "keep-alive"
            }
            timeout_sec = 30 if is_lan else 90
            async with session.get(url, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
                if r.status in (200, 206):
                    async with aiofiles.open(save_path, 'wb') as f:
                        async for chunk in r.content.iter_chunked(1024 * 512):
                            await f.write(chunk)
                    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                        return True
                    # 파일이 너무 작으면 손상된 것으로 판단
                    log(f"  [download] 파일 크기 미달 (손상 의심) — {url[-60:]}")
                else:
                    log(f"  [download] HTTP {r.status} — {url[-60:]}")
        except asyncio.TimeoutError:
            log(f"  [download] Timeout {attempt+1}/3 — {url[-60:]}")
        except Exception as e:
            log(f"  [download] 오류 {attempt+1}/3: {type(e).__name__}: {e}")
        if attempt < 2:
            await asyncio.sleep(3 * (attempt + 1))
    return False


# ==========================================
# 프레임 추출 (1회 추출 → 2·3단계 공유)
# ==========================================
def _mse(a, b):
    return np.sum((a.astype("float") - b.astype("float")) ** 2) / float(a.shape[0] * a.shape[1])


def extract_frames(video_path: str, max_frames: int = 8) -> tuple:
    """
    통합 프레임 추출 — 영상 길이 기반 동적 프레임 수 결정
    · 2단계(부적합) / 3단계(5점제 평가) 공통 사용 (1회 추출)
    · 가중 샘플링: 전반 40% + 후반 50% + 마지막 5%
    · MSE 기반 중복 제거 후 균등 재선택

    Returns: (status, frames_or_None, duration_sec, avg_motion)
      status: "Pass" | "ZeroSecond" | "StaticImage" | "BlackScreen" | "Error"
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error", None, 0.0, 0.0

    fc           = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = (fc / fps) if fps > 0 else 0.0

    # 1초 미만 영상 → 0점 처리
    if fc <= 0 or fps <= 0 or duration_sec < 1.0:
        cap.release()
        return "ZeroSecond", None, duration_sec, 0.0

    # ── 정지 이미지 감지: 3프레임(첫/중간/끝) 중심부(안쪽 40%) MSE ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_first, first_frame = cap.read()
    if ret_first and fc > 1:
        mid = fc // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret_mid, mid_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, fc - 1)
        ret_last, last_frame = cap.read()
        if ret_last:
            g_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            g_last  = cv2.cvtColor(last_frame,  cv2.COLOR_BGR2GRAY)
            g_mid   = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY) if ret_mid else None
            if g_first.shape != g_last.shape:
                h, w = g_first.shape
                g_last = cv2.resize(g_last, (w, h))
            if g_mid is not None and g_first.shape != g_mid.shape:
                h, w = g_first.shape
                g_mid = cv2.resize(g_mid, (w, h))
            # 중심부(안쪽 40%)만 비교
            h, w = g_first.shape
            mh, mw = int(h * 0.30), int(w * 0.30)
            def _cmse(a, b):
                return _mse(a[mh:h-mh, mw:w-mw], b[mh:h-mh, mw:w-mw])
            first_last = _cmse(g_first, g_last)
            center_mse = first_last
            if g_mid is not None:
                center_mse = max(first_last, _cmse(g_first, g_mid), _cmse(g_mid, g_last))
            if center_mse < 30:
                cap.release()
                return "StaticImage", None, duration_sec, 0.0

    # ── 영상 길이 기반 동적 프레임 수 ──
    if duration_sec < 15:
        target_frames = 8
    elif duration_sec < 30:
        target_frames = 10
    elif duration_sec < 60:
        target_frames = 12
    else:
        target_frames = 15

    max_frames = max(max_frames, target_frames)

    def read_frame(pos: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(pos, fc - 1)))
        ret, frame = cap.read()
        if not ret: return None
        scale = 512 / max(frame.shape[:2])
        return cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

    # ── 가중 샘플링 (전반 40% + 후반 50% + 마지막 5%) ──
    early_count     = int(target_frames * 0.4)
    late_count      = int(target_frames * 0.5)
    early_positions = [int(fc * (k + 0.5) / (early_count + 1)) for k in range(early_count)]
    late_positions  = [int(fc * (0.5 + 0.45 * k / late_count)) for k in range(late_count)]
    ending_pos      = int(fc * 0.95)
    positions       = sorted(set(early_positions + late_positions + [ending_pos]))
    if 0 not in positions:
        positions.insert(0, 0)

    raw_frames = []
    for pos in positions:
        f = read_frame(pos)
        if f is not None:
            raw_frames.append(f)
    cap.release()

    if not raw_frames:
        return "Error", None, duration_sec, 0.0

    # ── MSE 기반 중복 제거 ──
    deduped = [raw_frames[0]]
    for i in range(1, len(raw_frames)):
        g_prev = cv2.cvtColor(deduped[-1],   cv2.COLOR_BGR2GRAY)
        g_curr = cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2GRAY)
        if _mse(g_prev, g_curr) > 80:
            deduped.append(raw_frames[i])

    if len(deduped) > max_frames:
        step    = (len(deduped) - 1) / (max_frames - 1)
        indices = [round(step * k) for k in range(max_frames - 1)]
        indices.append(len(deduped) - 1)
        deduped = [deduped[idx] for idx in sorted(set(indices))]

    avg_bright = np.mean([cv2.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))[0] for f in deduped])
    if avg_bright < 5.0:
        return "BlackScreen", None, duration_sec, 0.0

    motion_scores = []
    for i in range(1, len(deduped)):
        g_prev = cv2.cvtColor(deduped[i-1], cv2.COLOR_BGR2GRAY)
        g_curr = cv2.cvtColor(deduped[i],   cv2.COLOR_BGR2GRAY)
        motion_scores.append(_mse(g_prev, g_curr))
    avg_motion = np.mean(motion_scores) if motion_scores else 0.0

    return "Pass", deduped, duration_sec, avg_motion


# ==========================================
# 프레임 그리드 이미지 생성 (부적합 탐지용)
# ==========================================
def create_frame_grid_b64(frames_bgr: list, cell_w: int = 256, cell_h: int = 192,
                           quality: int = 72) -> str:
    """추출된 프레임을 그리드 이미지로 합쳐 base64 반환"""
    if not frames_bgr: return ""
    cols = min(3, len(frames_bgr))
    rows = (len(frames_bgr) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (20, 20, 20))
    for idx, frame in enumerate(frames_bgr):
        r, c = divmod(idx, cols)
        img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((cell_w, cell_h), Image.LANCZOS)
        grid.paste(img, (c * cell_w + (cell_w - img.width) // 2,
                         r * cell_h + (cell_h - img.height) // 2))
    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==========================================
# 모음딜 상품 감지
# ==========================================
BUNDLE_DEAL_KEYWORDS = [
    "모음전", "종", "골라담기", "랜덤박스", "기획전",
    "패키지", "세트", "선택", "묶음", "특가세트",
    "한정판매", "추천상품", "베스트", "인기상품"
]

def is_bundle_deal_product(product_name: str) -> bool:
    if not product_name:
        return False
    product_name_lower = product_name.lower()
    for keyword in BUNDLE_DEAL_KEYWORDS:
        if keyword in product_name_lower:
            return True
    return False


def get_bundle_deal_check_mode(product_name: str) -> str:
    if is_bundle_deal_product(product_name):
        return "option_or_category"
    return "normal"