"""
test_extract_frames.py — 抽帧耗时测试
=====================================
테스트파일.xlsx 의 영상만 대상으로 extract_frames 만 돌려 시간을 측정합니다.
API 호출, 다운로드, 엑셀 저장 등은 전부 제외.
"""

import os
import sys
import time
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_LINK = r"E:\nginx-1.20.2\dist\review"
TEMP_VIDEO_DIR = os.path.join(BASE_DIR, "temp_videos")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from video_utils import extract_frames


def get_video_path(cont_no: str) -> str:
    return f"{VIDEO_LINK}/{cont_no}/{cont_no}.mp4"


def extract_cont_no(page_url: str) -> str | None:
    if not page_url:
        return None
    m = re.search(r'contNo=(\d+)', str(page_url), re.IGNORECASE)
    return m.group(1) if m else None


def main():
    excel_path = os.path.join(BASE_DIR, "테스트파일.xlsx")
    if not os.path.exists(excel_path):
        print(f"[ERROR] 파일 없음: {excel_path}")
        return

    # ── 헤더 형식 확인 (main.py 와 동일 로직) ──
    temp_df = pd.read_excel(excel_path, nrows=2, header=None)
    if "동영상 리뷰 목록" in str(temp_df.iloc[0, 0]):
        df = pd.read_excel(excel_path, header=1)
    else:
        df = pd.read_excel(excel_path, header=0)

    print(f"[INFO] 총 행수: {len(df)}")
    print(f"[INFO] 컬럼: {df.columns.tolist()}")

    # URL 컬럼 자동 탐지
    url_col = None
    for col in df.columns:
        col_str = str(col).strip()
        if 'URL' in col_str.upper() or 'url' in col_str:
            url_col = col
            break
    if not url_col:
        print("[ERROR] URL 컬럼을 찾을 수 없습니다.")
        return

    results = []
    total_extract_time = 0.0
    success_count = 0
    fail_count = 0

    for i, row in df.iterrows():
        url = row.get(url_col)
        if pd.isna(url) or not str(url).startswith("http"):
            continue

        cont_no = extract_cont_no(str(url))
        if not cont_no:
            continue

        video_path = get_video_path(cont_no)
        if not os.path.exists(video_path):
            print(f"[{i+1}] 파일 없음 스킵: {video_path}")
            fail_count += 1
            continue

        print(f"[{i+1}] {cont_no}.mp4 抽帧 시작...")
        t0 = time.time()
        status, frames, duration_sec, avg_motion = extract_frames(video_path)
        t1 = time.time()
        elapsed = t1 - t0
        total_extract_time += elapsed

        frame_cnt = len(frames) if frames else 0
        if status == "Pass":
            success_count += 1
            print(f"      ✓ Pass | frames={frame_cnt} | duration={duration_sec:.1f}s | time={elapsed:.2f}s")
        else:
            fail_count += 1
            print(f"      ✗ {status} | duration={duration_sec:.1f}s | time={elapsed:.2f}s")

        results.append({
            "index": i,
            "cont_no": cont_no,
            "status": status,
            "frames": frame_cnt,
            "duration_sec": duration_sec,
            "avg_motion": avg_motion,
            "time_sec": elapsed,
        })

    # ── Fallback: Excel 里的视频都不在本地时，扫描现有视频 ──
    if not results:
        print(f"\n[INFO] Excel 中的视频在本地不存在，扫描 {VIDEO_LINK} 下的现有视频...")
        local_videos = []
        for item in sorted(os.listdir(VIDEO_LINK)):
            item_path = os.path.join(VIDEO_LINK, item)
            if os.path.isdir(item_path):
                mp4_path = os.path.join(item_path, f"{item}.mp4")
                if os.path.exists(mp4_path):
                    local_videos.append((item, mp4_path))
        print(f"[INFO] 本地找到 {len(local_videos)} 个视频，取前 10 个测试")

        for idx, (cont_no, video_path) in enumerate(local_videos[:10], 1):
            print(f"[{idx}] {cont_no}.mp4 抽帧 시작...")
            t0 = time.time()
            status, frames, duration_sec, avg_motion = extract_frames(video_path)
            t1 = time.time()
            elapsed = t1 - t0
            total_extract_time += elapsed

            frame_cnt = len(frames) if frames else 0
            if status == "Pass":
                success_count += 1
                print(f"      ✓ Pass | frames={frame_cnt} | duration={duration_sec:.1f}s | time={elapsed:.2f}s")
            else:
                fail_count += 1
                print(f"      ✗ {status} | duration={duration_sec:.1f}s | time={elapsed:.2f}s")

            results.append({
                "index": idx,
                "cont_no": cont_no,
                "status": status,
                "frames": frame_cnt,
                "duration_sec": duration_sec,
                "avg_motion": avg_motion,
                "time_sec": elapsed,
            })

    # ── 요약 ──
    print("\n" + "=" * 55)
    print("📊 抽帧耗时测试汇总")
    print("=" * 55)
    print(f"   测试视频数: {len(results)}")
    print(f"   成功(Pass): {success_count}")
    print(f"   失败/跳过: {fail_count}")
    print(f"   总耗时:     {total_extract_time:.2f}s")
    if results:
        avg = total_extract_time / len(results)
        print(f"   平均每条:   {avg:.2f}s")
    print("=" * 55)


if __name__ == "__main__":
    main()
