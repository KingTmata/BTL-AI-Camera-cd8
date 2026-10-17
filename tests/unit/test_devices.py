import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


class DeviceTests(unittest.TestCase):
    def test_cpu_build_does_not_offer_gpu(self):
        from classroom_ai.inference.devices import cuda_devices, validate_device

        with patch("classroom_ai.inference.devices.torch.version.cuda", None):
            devices, reason = cuda_devices()
            self.assertEqual(devices, {})
            self.assertIn("CPU", reason)
            self.assertEqual(validate_device("cpu"), "cpu")
            with self.assertRaises(ValueError):
                validate_device("cuda:0")

    def test_only_gpus_that_complete_cuda_kernel_are_offered(self):
        from classroom_ai.inference.devices import cuda_devices

        with (
            patch("classroom_ai.inference.devices.torch.version.cuda", "13.0"),
            patch("classroom_ai.inference.devices.torch.cuda.is_available", return_value=True),
            patch("classroom_ai.inference.devices.torch.cuda.device_count", return_value=2),
            patch(
                "classroom_ai.inference.devices.torch.cuda.get_device_name",
                side_effect=["RTX 5060", "broken GPU"],
            ),
            patch(
                "classroom_ai.inference.devices.torch.ones",
                side_effect=[Mock(), RuntimeError("incompatible kernel")],
            ),
        ):
            devices, reason = cuda_devices()
            self.assertEqual(list(devices), ["cuda:0"])
            self.assertIn("RTX 5060", devices["cuda:0"])
            self.assertIn("incompatible kernel", reason)

    def test_model_cache_is_separate_for_each_device(self):
        from classroom_ai.interfaces.web.app import load_inspector

        load_inspector.clear()
        try:
            with patch(
                "classroom_ai.interfaces.web.app.ImageInspector", side_effect=[object(), object()]
            ) as create:
                cpu = load_inspector("weights.pt", 1, "cpu")
                gpu = load_inspector("weights.pt", 1, "cuda:0")
                self.assertIsNot(cpu, gpu)
                self.assertIs(cpu, load_inspector("weights.pt", 1, "cpu"))
                self.assertEqual(create.call_args_list[-1].kwargs["device"], "cuda:0")
        finally:
            load_inspector.clear()

    def test_sidebar_offers_cpu_and_probed_gpu_then_recovers_if_gpu_disappears(self):
        app = AppTest.from_string("""
from unittest.mock import patch
import streamlit as st
from classroom_ai.interfaces.web.app import main
devices = {} if st.session_state.get('gpu_missing') else {'cuda:0': 'GPU 0 · RTX 5060 (CUDA)'}
with patch('classroom_ai.interfaces.web.app.available_gpus', return_value=(devices, '')):
    main()
""").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox(key="inference_device").value, "cpu")
        app.selectbox(key="inference_device").select("cuda:0").run()
        self.assertEqual(app.selectbox(key="inference_device").value, "cuda:0")
        app.session_state["gpu_missing"] = True
        app.run()
        self.assertEqual(app.selectbox(key="inference_device").value, "cpu")
        self.assertTrue(any("không còn khả dụng" in item.value for item in app.warning))

    def test_image_device_change_invalidates_result_until_reanalysis(self):
        app = AppTest.from_string("""
from unittest.mock import Mock, patch
import numpy as np
from classroom_ai.interfaces.web.app import main
def inspector(path, stamp, device):
    return Mock(predict=Mock(return_value={'device': device, 'processing_ms': 1, 'detections': []}))
with patch('classroom_ai.interfaces.web.app.available_gpus', return_value=({'cuda:0': 'RTX 5060'}, '')), \\
     patch('classroom_ai.interfaces.web.app.sample_source', return_value=(np.zeros((64,64,3), dtype=np.uint8), 'sample', None)), \\
     patch('classroom_ai.interfaces.web.app.load_inspector', side_effect=inspector):
    main()
""").run()
        app.button(key="analyze").click().run()
        self.assertEqual(app.session_state["inspection"]["device"], "cpu")
        app.selectbox(key="inference_device").select("cuda:0").run()
        self.assertTrue(any("cần chạy lại" in item.value for item in app.info))
        self.assertEqual(len(app.metric), 0)
        app.button(key="analyze").click().run()
        self.assertEqual(app.session_state["inspection"]["device"], "cuda:0")
        self.assertEqual(len(app.exception), 0)

    def test_webcam_device_change_waits_for_old_worker_to_stop(self):
        for stopped in (True, False):
            app = AppTest.from_string(
                """
from pathlib import Path
from unittest.mock import Mock
import streamlit as st
from classroom_ai.interfaces.web.app import webcam_panel
checkpoint = Path('configs/app.yaml')
camera = Mock()
camera.stop.return_value = STOPPED
st.session_state.camera_session = camera
st.session_state.camera_config = (0, str(checkpoint), checkpoint.stat().st_mtime_ns, .25, 640, None, 1, 'cpu')
webcam_panel(checkpoint, .25, [], Mock(), device='cuda:0')
assert camera.stop.called
camera.start.assert_not_called()
camera.snapshot.assert_not_called()
assert ('camera_session' not in st.session_state) == STOPPED
""".replace("STOPPED", repr(stopped))
            ).run()
            with self.subTest(stopped=stopped):
                self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
