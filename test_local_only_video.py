import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class LocalOnlyVideoFlowTests(unittest.TestCase):
    def test_main_process_uses_single_lan_url_and_returns_video_error_directly(self):
        main = importlib.import_module("main")

        async def fake_load_local_metadata(session, cont_no):
            return {
                "product": {"name": "상품", "option": "옵션"},
                "review": {"text": "리뷰"},
                "status": {"video_ok": True, "image_ok": False},
                "video": {"local_url": r"E:\\review\\12345\\12345.mp4"},
            }

        attempted_urls = []

        async def fake_download_video_direct(session, url, save_path):
            attempted_urls.append(url)
            return True

        def fake_extract_frames(path):
            return "Error", None, 0.0, 0.0

        sem = asyncio.Semaphore(1)
        row_data = (0, "review-1", "https://example.test/review?contNo=12345", "cat", 0, "member-1")

        with patch.object(main, "load_local_metadata", fake_load_local_metadata), \
             patch.object(main, "get_image_b64", lambda session, url: ""), \
             patch.object(main, "download_video_direct", fake_download_video_direct), \
             patch.object(main, "extract_frames", fake_extract_frames), \
             patch.object(main.os.path, "exists", lambda path: False):
            result = asyncio.run(main.process_one_review(sem, SimpleNamespace(), row_data, {}, None))

        self.assertEqual(attempted_urls, [f"http://{main.LAN_SERVER_IP}/review/12345/12345.mp4"])
        self.assertEqual(result["review_id"], "review-1")
        self.assertEqual(result["eval_stage"], "영상(오류)")
        self.assertEqual(result["total_score"], 0)

    def test_hash_resolve_video_path_uses_only_lan_url(self):
        hash_utils = importlib.import_module("hash_utils")

        attempted_urls = []

        async def fake_download_video_direct(session, url, save_path):
            attempted_urls.append(url)
            return False

        sem = asyncio.Semaphore(1)
        row_data = (0, "review-1", "https://example.test/review?contNo=12345", "cat", 0, "member-1")

        with patch.object(hash_utils, "download_video_direct", fake_download_video_direct):
            result = asyncio.run(hash_utils._resolve_video_path(SimpleNamespace(), row_data, sem))

        self.assertEqual(attempted_urls, [f"http://{hash_utils.LAN_SERVER_IP}/review/12345/12345.mp4"])
        self.assertFalse(result["success"])

    def test_hash_report_accepts_six_field_task_tuples(self):
        hash_utils = importlib.import_module("hash_utils")

        saved_paths = []

        class ColumnDimensions(dict):
            def __missing__(self, key):
                value = SimpleNamespace(width=None)
                self[key] = value
                return value

        class FakeWorksheet:
            def __init__(self):
                self.title = ""
                self.column_dimensions = ColumnDimensions()

            def cell(self, row, column, value=None):
                return SimpleNamespace(value=value, fill=None, font=None, alignment=None)

        class FakeWorkbook:
            def __init__(self):
                self.active = FakeWorksheet()

            def save(self, path):
                saved_paths.append(path)

        task_data = [(0, "review-1", "https://example.test/review?contNo=12345", "cat", 0, "member-1")]
        records = [{"review_id": "review-1", "member_id": "member-1", "success": False}]

        with patch.object(hash_utils, "Workbook", FakeWorkbook), \
             patch.object(hash_utils.os, "makedirs", lambda *args, **kwargs: None):
            stats = hash_utils.save_hash_report(records, {}, task_data)

        self.assertEqual(stats["fail"], 1)
        self.assertTrue(saved_paths)


if __name__ == "__main__":
    unittest.main()
