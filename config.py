"""
config.py — 설정값·경로·API 클라이언트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 모듈이 공통으로 사용하는 설정값을 여기에 모아둡니다.
경로·모델·동시처리 수 등을 바꿀 때 이 파일만 수정하면 됩니다.
"""

import os
import time
import json
import glob
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
from pandas.core.indexes.base import F

# ==========================================
# 환경 변수 로드
# ==========================================
# 1. 고정 경로 시도
dotenv_path = r"D:\zubong\test\python\.env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    # 2. 현재 작업 디렉토리의 .env 시도
    load_dotenv()

# ==========================================
# 경로 설정 (현재 스크립트 위치 기준)
# ==========================================
# 현재 파일(config.py)이 있는 폴더 경로를 BASE_DIR로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_VIDEO_DIR = os.path.join(BASE_DIR, "temp_videos")
AI_INPUT_DIR = BASE_DIR   # 입력 파일을 main.py와 같은 폴더에 두기 위해 수정
AI_OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ==========================================
# 파일 경로 지정 (어제 날짜의 리뷰동영상 파일 자동 탐색)
# 예: D:\Python\리뷰BO크롤링\리뷰동영상_20260412_20260412.xls
yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
today_str = datetime.now().strftime('%Y%m%d')

# ==========================================
# 실행 설정 (INPUT_EXCEL_PATH보다 먼저 정의 필요)
# ==========================================
MAX_CONCURRENT_TASKS = 10
TEST_MODE = False          # ✅ 테스트 모드 활성화
TEST_LIMIT = 30                   # 테스트 시 처리할 데이터 수
TEST_FILE_PATH = os.path.join(BASE_DIR, "테스트파일.xlsx")  # 테스트 파일 경로

# 검색 경로 목록 (고정 경로 + 현재 실행 경로)
search_dirs = [
    r"D:\Python\리뷰BO크롤링",
    os.getcwd(),  # 현재 실행 경로 추가
    BASE_DIR,    # 스크립트 위치 경로
]

INPUT_EXCEL_PATH = ""
found_in = ""

# ── 테스트 모드 우선 처리 ──
_CONFIG_INFO = []  # main.py에서 출력할 시작 메시지
if TEST_MODE and os.path.exists(TEST_FILE_PATH):
    INPUT_EXCEL_PATH = TEST_FILE_PATH
    _CONFIG_INFO.append(f"[INFO] 테스트 모드: 테스트파일.xlsx 사용")
else:
    # ── 일반 모드: 어제 날짜 파일 자동 검색 ──
    for target_dir in search_dirs:
        if not os.path.isdir(target_dir):
            continue
        search_pattern = os.path.join(target_dir, f"리뷰동영상*{yesterday_str}*.xls*")
        matches = glob.glob(search_pattern)
        if matches:
            INPUT_EXCEL_PATH = matches[0]
            found_in = target_dir
            _CONFIG_INFO.append(f"[INFO] 어제 날짜({yesterday_str}) 파일 발견: {INPUT_EXCEL_PATH}")
            break

    if not INPUT_EXCEL_PATH:
        _CONFIG_INFO.append(f"[경고] 어제 날짜({yesterday_str}) 파일을 찾지 못했습니다.")
        _CONFIG_INFO.append(f"[경고] 검색된 경로: {search_dirs}")

# 검수 완료 시 오늘 날짜 폴더 생성
OUTPUT_DIR = os.path.join(AI_OUTPUT_DIR, today_str)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"리뷰동영상검수_{today_str}.xlsx")
ZERO_RULES_FILE = os.path.join(BASE_DIR, "zero_point_rules.json")

os.makedirs(AI_INPUT_DIR, exist_ok=True)
os.makedirs(AI_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 오늘 날짜 폴더 생성
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# ==========================================
# OpenAI API 설정
# ==========================================
openai_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)
GPT_MODEL = "gpt-4o"

# ==========================================
# 비디오 소스 설정
# ==========================================
VIDEO_LINK = r"E:\nginx-1.20.2\dist\review" # "http://10.240.129.13/review"

def get_video_path(cont_no: str) -> str:
    """contNo 기반 영상 경로 반환"""
    return f"{VIDEO_LINK}/{cont_no}/{cont_no}.mp4"

def get_metadata_path(cont_no: str) -> str:
    """contNo 기반 메타데이터 경로 반환"""
    return f"{VIDEO_LINK}/{cont_no}/metadata.json"


# ==========================================
# 중복 영상 감지 설정
# ==========================================
DUPLICATE_HASH_THRESHOLD = 5
DUPLICATE_DURATION_MARGIN = 1

# ── 하이브리드 해시 설정 ──
HASH_CACHE_DIR = os.path.join(BASE_DIR, "hash_cache")
HASH_CACHE_PATH = os.path.join(HASH_CACHE_DIR, "_hash_cache.json")
QUICK_HASH_BYTES = 1024 * 1024           # Quick MD5: 파일 앞 1MB만 읽기
FRAME_SAMPLE_POSITIONS = [0.1, 0.5, 0.9] # 프레임 샘플링 위치 (10%/50%/90%)
HASH_FRAME_SIZE = (128, 128)             # 프레임 리사이즈 크기
CPU_WORKERS = max(1, os.cpu_count() // 2) # 프레임 해시 CPU 병렬 수
HASH_REPORT_DIR = os.path.join(BASE_DIR, "hash_report")


# ==========================================
# 공통 유틸리티
# ==========================================
def log(msg: str):
    """로그 출력 (나중에 파일 로깅 등으로 확장 가능)"""
    print(msg)


def load_zero_point_rules() -> dict:
    """zero_point_rules.json 로드"""
    path = ZERO_RULES_FILE
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero_point_rules.json")
        if os.path.exists(alt):
            path = alt
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log(f"[경고] zero_point_rules.json 파싱 오류: {e}")
    log("[경고] zero_point_rules.json 을 찾지 못했습니다.")
    return {}
