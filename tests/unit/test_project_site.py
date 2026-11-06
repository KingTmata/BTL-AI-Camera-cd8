import http.client
import importlib.util
import json
import tempfile
import threading
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "docs" / "project_site" / "server.py"
SPEC = importlib.util.spec_from_file_location("project_site_server", SERVER_PATH)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class PageStructureTests(unittest.TestCase):
    def test_sections_stay_inside_main_and_navigation_targets_exist(self):
        class PageParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.stack = []
                self.ids = set()
                self.links = []
                self.errors = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if attrs.get("id"):
                    self.ids.add(attrs["id"])
                if tag == "a" and attrs.get("href", "").startswith("#"):
                    self.links.append(attrs["href"][1:])
                if tag == "section" and "main" not in self.stack:
                    self.errors.append("Section outside main: " + attrs.get("id", ""))
                if tag not in {
                    "area",
                    "base",
                    "br",
                    "col",
                    "embed",
                    "hr",
                    "img",
                    "input",
                    "link",
                    "meta",
                    "param",
                    "source",
                    "track",
                    "wbr",
                }:
                    self.stack.append(tag)

            def handle_endtag(self, tag):
                if not self.stack or self.stack[-1] != tag:
                    self.errors.append("Unexpected closing tag: " + tag)
                if tag in self.stack:
                    while self.stack.pop() != tag:
                        pass

        parser = PageParser()
        parser.feed((SERVER_PATH.parent / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(parser.errors, [])
        self.assertEqual(parser.stack, [])
        self.assertFalse(set(parser.links) - parser.ids)
        self.assertTrue(
            {"code-list", "code-detail", "future-detail", "start", "sources"} <= parser.ids
        )


class StaticPathTests(unittest.TestCase):
    def test_rejects_paths_outside_site_and_unlisted_files(self):
        self.assertIsNone(server.resolve_static_path("/../tasks/plan.md"))
        self.assertIsNone(server.resolve_static_path("/server.py"))
        self.assertEqual(
            server.resolve_static_path("/index.html"), SERVER_PATH.parent / "index.html"
        )


class ProbeAccessTests(unittest.TestCase):
    def test_probe_is_available_only_to_loopback_clients(self):
        self.assertTrue(server.is_loopback_address("127.0.0.1"))
        self.assertTrue(server.is_loopback_address("::1"))
        self.assertFalse(server.is_loopback_address("192.168.1.25"))
        denied = server.probe_access_error(
            "192.168.1.25", "http://192.168.1.25:8765", "192.168.1.25:8765", "application/json"
        )
        self.assertEqual(denied[0], 403)
        self.assertEqual(denied[1]["code"], "loopback_only")
        rebinding = server.probe_access_error(
            "127.0.0.1", "http://attacker.example:8765", "attacker.example:8765", "application/json"
        )
        self.assertEqual(rebinding[1]["code"], "origin_rejected")
        wrong_type = server.probe_access_error(
            "127.0.0.1", "http://127.0.0.1:8765", "127.0.0.1:8765", "text/plain"
        )
        self.assertEqual(wrong_type[0], 415)

    def test_invalid_null_and_unc_paths_return_structured_errors(self):
        self.assertEqual(server.probe_local_video("bad\x00path")["code"], "invalid_path")
        self.assertEqual(
            server.probe_local_video(r"\\server\share\clip.mp4")["code"], "unsupported_path"
        )
        self.assertEqual(server.probe_local_video(r"C:\missing\clip.mp4")["code"], "file_not_found")

    def test_lan_urls_include_usable_ipv4_but_reject_link_local_and_ipv6(self):
        self.assertTrue(server._is_usable_lan_ipv4("192.168.1.20"))
        self.assertFalse(server._is_usable_lan_ipv4("169.254.10.20"))
        self.assertFalse(server._is_usable_lan_ipv4("2001:db8::1"))


class MetadataTests(unittest.TestCase):
    def test_normalizes_ffprobe_container_and_video_stream(self):
        payload = {
            "format": {
                "format_name": "mpeg",
                "duration": "3390",
                "size": "762314752",
                "bit_rate": "1800000",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "profile": "Main",
                    "level": 120,
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "25/1",
                    "pix_fmt": "yuv420p",
                    "color_range": "pc",
                    "color_space": "bt709",
                    "bit_rate": "1764000",
                }
            ],
        }
        result = server.normalize_probe_result(
            json.dumps(payload), "VCP2_204_20260810_V01.mp4", 762314752
        )
        self.assertEqual(result["container"], "mpeg")
        self.assertEqual(result["container_formats"], "mpeg")
        self.assertEqual(result["codec"], "hevc")
        self.assertEqual(result["level_label"], "L4.0")
        self.assertEqual(result["fps"], 25.0)
        self.assertEqual(result["duration_seconds"], 3390.0)
        self.assertIn("extension .mp4", " ".join(result["warnings"]))


class OfficialContentTests(unittest.TestCase):
    def test_extracts_only_the_current_official_project_section(self):
        result = server.extract_official_project(
            (ROOT / "tasks" / "plan.md").read_text(encoding="utf-8")
        )
        self.assertEqual(
            result["title"],
            "Nghiên cứu xây dựng hệ thống nhận dạng và đánh giá tính tương tác của sinh viên trong quá trình học tập trên lớp với hệ sinh thái AI-Native",
        )
        self.assertGreaterEqual(len(result["research_items"]), 6)
        self.assertFalse(any("GAN" in item for item in result["research_items"]))


class HttpSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.ProjectSiteServer(
            ("127.0.0.1", 0), server.ProjectSiteHandler, "127.0.0.1"
        )
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.httpd.server_port

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        status, content_type = response.status, response.getheader("Content-Type")
        connection.close()
        return status, content_type, data

    def test_static_allowlist_serves_asset_and_hides_server_and_repository_files(self):
        status, content_type, body = self.request("GET", "/assets/project-overview.png")
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("image/png"))
        self.assertTrue(body.startswith(b"\x89PNG"))
        for path in ("/server.py", "/%2e%2e/.git/config"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("HEAD", "/server.py")[0], 404)

    def test_code_catalog_and_source_endpoint_use_curated_allowlist(self):
        status, content_type, body = self.request("GET", "/api/code")
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("application/json"))
        modules = json.loads(body)["modules"]
        ids = {item["id"] for item in modules}
        self.assertIn("video-cli", ids)
        self.assertIn("inference-detector", ids)
        self.assertIn("site-server", ids)
        self.assertTrue(all(item["file_exists"] and item["line_count"] > 0 for item in modules))
        cli = next(item for item in modules if item["id"] == "video-cli")
        self.assertIn("main", cli["symbols"])

        status, _, body = self.request("GET", "/api/code/video-cli")
        source = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(source["path"], "src/classroom_ai/video/__main__.py")
        self.assertIn("def main", source["content"])
        self.assertLessEqual(len(source["content"].encode("utf-8")), server.MAX_CODE_BYTES)

        for path in (
            "/api/code/../../tasks/plan.md",
            "/api/code/%2e%2e",
            "/api/code/site-server/../../tasks/plan.md",
            "/api/code/not-a-module",
        ):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 404)
                self.assertFalse("content" in json.loads(body))

    def test_same_origin_json_probe_reaches_path_handler_after_quote_cleanup(self):
        probe_json = json.dumps(
            {
                "format": {"format_name": "mpeg", "duration": "2", "size": "12"},
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "hevc",
                        "avg_frame_rate": "25/1",
                        "width": 1920,
                        "height": 1080,
                    }
                ],
            }
        )
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "Origin": f"http://127.0.0.1:{self.port}",
            "Content-Type": "application/json",
        }
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            body = json.dumps({"path": f'"{video.name}"'})
            with mock.patch.object(
                server.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=probe_json, stderr=""),
            ) as probe:
                status, _, data = self.request("POST", "/api/probe", body, headers)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["codec"], "hevc")
        self.assertEqual(probe.call_args.args[0][-1], video.name)
        self.assertEqual(probe.call_args.args[0][-2], "json")

    def test_probe_rejects_cross_origin_non_json_and_oversized_requests(self):
        json_body = json.dumps({"path": "clip.mp4"})
        common = {"Host": f"127.0.0.1:{self.port}"}
        status, _, data = self.request(
            "POST",
            "/api/probe",
            json_body,
            {**common, "Origin": "http://attacker.example", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(data)["code"], "origin_rejected")
        status, _, data = self.request(
            "POST",
            "/api/probe",
            json_body,
            {**common, "Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "text/plain"},
        )
        self.assertEqual(status, 415)
        status, _, data = self.request(
            "POST",
            "/api/probe",
            "{}" + " " * 16384,
            {
                **common,
                "Origin": f"http://127.0.0.1:{self.port}",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(data)["code"], "payload_size")


if __name__ == "__main__":
    unittest.main()
