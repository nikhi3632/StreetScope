#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <streetscope/frame_pipeline/frame_loop.h>
#include <streetscope/inference/coreml_detector.h>
#include <streetscope/inference/detection.h>
#include <videotoolbox_decoder.h>

#include <cstring>
#include <stdexcept>

namespace py = pybind11;
using streetscope::FrameLoop;
using streetscope::FrameLoopConfig;
using streetscope::ProcessedFrame;

// --- Zero-copy numpy helpers via py::capsule ---

/// Move a vector to the heap and return a numpy array that references its data.
/// The capsule destructor frees the vector when numpy is garbage collected.
static py::array_t<uint8_t> vec_to_numpy_3d(
    std::vector<uint8_t>& vec, int height, int width, int channels
) {
    auto* heap = new std::vector<uint8_t>(std::move(vec));
    auto capsule = py::capsule(heap, [](void* p) {
        delete static_cast<std::vector<uint8_t>*>(p);
    });
    return py::array_t<uint8_t>(
        {height, width, channels},
        {width * channels, channels, 1},
        heap->data(),
        capsule
    );
}

static py::array_t<uint8_t> vec_to_numpy_2d(
    std::vector<uint8_t>& vec, int height, int width
) {
    auto* heap = new std::vector<uint8_t>(std::move(vec));
    auto capsule = py::capsule(heap, [](void* p) {
        delete static_cast<std::vector<uint8_t>*>(p);
    });
    return py::array_t<uint8_t>(
        {height, width},
        {width, 1},
        heap->data(),
        capsule
    );
}

// --- Module definition ---

PYBIND11_MODULE(streetscope_pipeline, m) {
    m.doc() = "StreetScope C++ frame pipeline (macOS VideoToolbox decoder + FrameLoop)";

    // --- ProcessedFrame ---
    py::class_<ProcessedFrame>(m, "ProcessedFrame")
        .def_readonly("width", &ProcessedFrame::width)
        .def_readonly("height", &ProcessedFrame::height)
        .def_readonly("frame_number", &ProcessedFrame::frame_number)
        .def_readonly("timestamp_s", &ProcessedFrame::timestamp_s)
        .def_readonly("media_pts_s", &ProcessedFrame::media_pts_s)
        .def_property_readonly("bgr", [](ProcessedFrame& self) {
            return vec_to_numpy_3d(self.bgr_data, self.height, self.width, 3);
        }, "Raw BGR frame as numpy (H,W,3). Zero-copy via capsule.")
        .def_property_readonly("mask", [](ProcessedFrame& self) {
            return vec_to_numpy_2d(self.mask_data, self.height, self.width);
        }, "Motion mask as numpy (H,W). Zero-copy via capsule.")
        .def_property_readonly("display", [](ProcessedFrame& self) {
            return vec_to_numpy_3d(self.display_data, self.height, self.width, 3);
        }, "ISP-corrected display as numpy (H,W,3). Zero-copy via capsule.");

    // --- FrameLoop ---
    py::class_<FrameLoop>(m, "FrameLoop")
        .def(py::init<const std::string&>(), py::arg("url"))
        .def("start", &FrameLoop::start,
            "Start the decoder and processing thread.")
        .def("stop", &FrameLoop::stop,
            "Stop the processing thread and decoder.")
        .def("try_get_result", &FrameLoop::try_get_result,
            "Get next processed frame, or None if not ready.")
        .def("update_config", [](FrameLoop& self,
                float ema_alpha,
                uint8_t motion_threshold,
                const py::object& lut_obj,
                float gain_b,
                float gain_g,
                float gain_r,
                const py::object& alpha_map_obj,
                int blur_ksize) {
            FrameLoopConfig cfg;
            cfg.pipeline.ema_alpha = ema_alpha;
            cfg.pipeline.motion_threshold = motion_threshold;

            bool has_lut = !lut_obj.is_none();
            bool has_alpha_map = !alpha_map_obj.is_none();

            cfg.pipeline.apply_isp = has_lut;
            if (has_lut) {
                auto lut = lut_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
                auto lut_buf = lut.request();
                if (lut_buf.size != 256) {
                    throw std::invalid_argument("lut must have exactly 256 elements");
                }
                std::memcpy(cfg.pipeline.ae_awb.lut, lut_buf.ptr, 256);
                cfg.pipeline.ae_awb.gain_b = gain_b;
                cfg.pipeline.ae_awb.gain_g = gain_g;
                cfg.pipeline.ae_awb.gain_r = gain_r;
                cfg.pipeline.af_blur_ksize =
                    (has_alpha_map && blur_ksize > 0) ? blur_ksize : 0;
            }

            if (has_alpha_map) {
                auto alpha_map = alpha_map_obj.cast<
                    py::array_t<float, py::array::c_style>>();
                auto am_buf = alpha_map.request();
                cfg.alpha_map.assign(
                    static_cast<const float*>(am_buf.ptr),
                    static_cast<const float*>(am_buf.ptr) + am_buf.size
                );
            }

            self.update_config(std::move(cfg));
        },
            py::arg("ema_alpha"),
            py::arg("motion_threshold"),
            py::arg("lut") = py::none(),
            py::arg("gain_b") = 1.0f,
            py::arg("gain_g") = 1.0f,
            py::arg("gain_r") = 1.0f,
            py::arg("alpha_map") = py::none(),
            py::arg("blur_ksize") = 5,
            "Push updated ISP config. Thread-safe.")
        .def("is_running", &FrameLoop::is_running)
        .def("frames_processed", &FrameLoop::frames_processed);

    m.def("pump_main_runloop", &streetscope::pump_main_runloop,
        py::arg("seconds") = 0.01,
        "Pump the main thread's run loop. Must be called from the main thread. "
        "Required for AVPlayer when no Cocoa event loop is running.");

    // --- Detection ---
    py::class_<streetscope::Detection>(m, "Detection")
        .def_readonly("x1", &streetscope::Detection::x1)
        .def_readonly("y1", &streetscope::Detection::y1)
        .def_readonly("x2", &streetscope::Detection::x2)
        .def_readonly("y2", &streetscope::Detection::y2)
        .def_readonly("confidence", &streetscope::Detection::confidence)
        .def_readonly("class_id", &streetscope::Detection::class_id)
        .def_property_readonly("class_name", [](const streetscope::Detection& d) {
            return streetscope::coco_class_name(d.class_id);
        })
        .def_property_readonly("bbox", [](const streetscope::Detection& d) {
            return py::make_tuple(d.x1, d.y1, d.x2, d.y2);
        })
        .def_property_readonly("area", &streetscope::Detection::area)
        .def("__repr__", [](const streetscope::Detection& d) {
            return "<Detection " + std::string(streetscope::coco_class_name(d.class_id)) +
                   " conf=" + std::to_string(d.confidence) +
                   " bbox=(" + std::to_string(d.x1) + "," + std::to_string(d.y1) +
                   "," + std::to_string(d.x2) + "," + std::to_string(d.y2) + ")>";
        });

    // --- CoreMLDetector ---
    py::class_<streetscope::CoreMLDetector>(m, "CoreMLDetector")
        .def(py::init<const std::string&, float, float>(),
            py::arg("model_path"),
            py::arg("conf_threshold") = 0.25f,
            py::arg("iou_threshold") = 0.45f)
        .def("detect", [](streetscope::CoreMLDetector& self,
                          py::array_t<uint8_t, py::array::c_style> frame,
                          bool vehicles_only) {
            auto buf = frame.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::invalid_argument("frame must be (H, W, 3) uint8");
            }
            auto h = static_cast<int>(buf.shape[0]);
            auto w = static_cast<int>(buf.shape[1]);
            return self.detect(static_cast<const uint8_t*>(buf.ptr), w, h, vehicles_only);
        }, py::arg("frame"), py::arg("vehicles_only") = false,
           "Run sync detection on a BGR frame. Returns list of Detection.")
        .def("submit", [](streetscope::CoreMLDetector& self,
                          py::array_t<uint8_t, py::array::c_style> frame,
                          bool vehicles_only) {
            auto buf = frame.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::invalid_argument("frame must be (H, W, 3) uint8");
            }
            auto h = static_cast<int>(buf.shape[0]);
            auto w = static_cast<int>(buf.shape[1]);
            self.submit(static_cast<const uint8_t*>(buf.ptr), w, h, vehicles_only);
        }, py::arg("frame"), py::arg("vehicles_only") = false,
           "Submit frame for async detection.")
        .def("try_get_result", &streetscope::CoreMLDetector::try_get_result,
            "Poll for async result. Returns DetectionResult or None.")
        .def_property_readonly("conf_threshold", &streetscope::CoreMLDetector::conf_threshold)
        .def_property_readonly("iou_threshold", &streetscope::CoreMLDetector::iou_threshold)
        .def_property_readonly("input_size", &streetscope::CoreMLDetector::input_size);

    // --- DetectionResult ---
    py::class_<streetscope::DetectionResult>(m, "DetectionResult")
        .def_readonly("detections", &streetscope::DetectionResult::detections)
        .def_readonly("inference_ms", &streetscope::DetectionResult::inference_ms)
        .def_readonly("frame_number", &streetscope::DetectionResult::frame_number);
}
