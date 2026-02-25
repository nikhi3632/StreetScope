#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <streetscope/frame_pipeline/frame_loop.h>
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
}
