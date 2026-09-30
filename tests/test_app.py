import unittest

from streamlit.testing.v1 import AppTest

from src.inference.detector import ROOT


@unittest.skipUnless((ROOT / "weights/yolo26n.pt").is_file() and
                     (ROOT / "data/reference/coco128/images/train2017/000000000283.jpg").is_file(),
                     "Cần tải weights và COCO128 để chạy integration test")
class AppTests(unittest.TestCase):
    def test_real_inference_filters_and_stale_results(self):
        app = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.button(key="analyze").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertGreater(int(app.metric[1].value), 0, "Ảnh fixture phải có dự đoán bottle")
        app.multiselect[0].set_value([]).run()
        self.assertFalse(app.error)
        self.assertTrue(all(int(m.value) == 0 for m in app.metric[:4]))
        app.slider[0].set_value(0.5).run()
        self.assertEqual(len(app.metric), 0, "Đổi confidence phải ẩn kết quả cũ")
        app.radio[0].set_value("Ảnh của bạn").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
