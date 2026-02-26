#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <streetscope/simd/accumulator.h>
#include <streetscope/simd/subtractor.h>
#include <streetscope/simd/isp_correction.h>
#include <streetscope/simd/pipeline.h>
#include <streetscope/plan_executor/plan_executor.h>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;
using namespace streetscope::simd;

static void py_accumulate_ema(
    py::array_t<float, py::array::c_style> frame,
    py::array_t<float, py::array::c_style> background,
    float alpha
) {
    auto frame_buf = frame.request();
    auto bg_buf = background.request(true);

    if (frame_buf.size != bg_buf.size) {
        throw std::invalid_argument("frame and background must have the same size");
    }

    AccumulatorConfig config{alpha, static_cast<int>(frame_buf.size)};
    accumulate_ema_neon(
        static_cast<const float*>(frame_buf.ptr),
        static_cast<float*>(bg_buf.ptr),
        config
    );
}

static void py_accumulate_ema_scalar(
    py::array_t<float, py::array::c_style> frame,
    py::array_t<float, py::array::c_style> background,
    float alpha
) {
    auto frame_buf = frame.request();
    auto bg_buf = background.request(true);

    if (frame_buf.size != bg_buf.size) {
        throw std::invalid_argument("frame and background must have the same size");
    }

    AccumulatorConfig config{alpha, static_cast<int>(frame_buf.size)};
    accumulate_ema_scalar(
        static_cast<const float*>(frame_buf.ptr),
        static_cast<float*>(bg_buf.ptr),
        config
    );
}

static py::array_t<uint8_t> py_subtract_background(
    py::array_t<uint8_t, py::array::c_style> frame,
    py::array_t<uint8_t, py::array::c_style> background,
    uint8_t threshold
) {
    auto frame_buf = frame.request();
    auto bg_buf = background.request();

    if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
        throw std::invalid_argument("frame must be (H, W, 3) uint8");
    }
    if (frame_buf.size != bg_buf.size) {
        throw std::invalid_argument("frame and background must have the same size");
    }

    int height = static_cast<int>(frame_buf.shape[0]);
    int width = static_cast<int>(frame_buf.shape[1]);

    auto mask = py::array_t<uint8_t>({height, width});
    auto mask_buf = mask.request(true);

    SubtractorConfig config{threshold, width, height};
    subtract_background_neon(
        static_cast<const uint8_t*>(frame_buf.ptr),
        static_cast<const uint8_t*>(bg_buf.ptr),
        static_cast<uint8_t*>(mask_buf.ptr),
        config
    );

    return mask;
}

static py::array_t<uint8_t> py_subtract_background_scalar(
    py::array_t<uint8_t, py::array::c_style> frame,
    py::array_t<uint8_t, py::array::c_style> background,
    uint8_t threshold
) {
    auto frame_buf = frame.request();
    auto bg_buf = background.request();

    if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
        throw std::invalid_argument("frame must be (H, W, 3) uint8");
    }
    if (frame_buf.size != bg_buf.size) {
        throw std::invalid_argument("frame and background must have the same size");
    }

    int height = static_cast<int>(frame_buf.shape[0]);
    int width = static_cast<int>(frame_buf.shape[1]);

    auto mask = py::array_t<uint8_t>({height, width});
    auto mask_buf = mask.request(true);

    SubtractorConfig config{threshold, width, height};
    subtract_background_scalar(
        static_cast<const uint8_t*>(frame_buf.ptr),
        static_cast<const uint8_t*>(bg_buf.ptr),
        static_cast<uint8_t*>(mask_buf.ptr),
        config
    );

    return mask;
}

static py::array_t<uint8_t> py_apply_ae_awb(
    py::array_t<uint8_t, py::array::c_style> frame,
    py::array_t<uint8_t, py::array::c_style> lut,
    float gain_b, float gain_g, float gain_r
) {
    auto frame_buf = frame.request();
    auto lut_buf = lut.request();

    if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
        throw std::invalid_argument("frame must be (H, W, 3) uint8");
    }
    if (lut_buf.size != 256) {
        throw std::invalid_argument("lut must have exactly 256 elements");
    }

    int height = static_cast<int>(frame_buf.shape[0]);
    int width = static_cast<int>(frame_buf.shape[1]);

    auto output = py::array_t<uint8_t>({height, width, 3});
    auto output_buf = output.request(true);

    AEAWBConfig config{};
    std::memcpy(config.lut, lut_buf.ptr, 256);
    config.gain_b = gain_b;
    config.gain_g = gain_g;
    config.gain_r = gain_r;
    config.width = width;
    config.height = height;

    apply_ae_awb_neon(
        static_cast<const uint8_t*>(frame_buf.ptr),
        static_cast<uint8_t*>(output_buf.ptr),
        config
    );

    return output;
}

static py::array_t<uint8_t> py_apply_ae_awb_scalar(
    py::array_t<uint8_t, py::array::c_style> frame,
    py::array_t<uint8_t, py::array::c_style> lut,
    float gain_b, float gain_g, float gain_r
) {
    auto frame_buf = frame.request();
    auto lut_buf = lut.request();

    if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
        throw std::invalid_argument("frame must be (H, W, 3) uint8");
    }
    if (lut_buf.size != 256) {
        throw std::invalid_argument("lut must have exactly 256 elements");
    }

    int height = static_cast<int>(frame_buf.shape[0]);
    int width = static_cast<int>(frame_buf.shape[1]);

    auto output = py::array_t<uint8_t>({height, width, 3});
    auto output_buf = output.request(true);

    AEAWBConfig config{};
    std::memcpy(config.lut, lut_buf.ptr, 256);
    config.gain_b = gain_b;
    config.gain_g = gain_g;
    config.gain_r = gain_r;
    config.width = width;
    config.height = height;

    apply_ae_awb_scalar(
        static_cast<const uint8_t*>(frame_buf.ptr),
        static_cast<uint8_t*>(output_buf.ptr),
        config
    );

    return output;
}

static py::array_t<uint8_t> py_apply_af_blend(
    py::array_t<uint8_t, py::array::c_style> frame,
    py::array_t<uint8_t, py::array::c_style> blurred,
    py::array_t<float, py::array::c_style> alpha_map
) {
    auto frame_buf = frame.request();
    auto blurred_buf = blurred.request();
    auto alpha_buf = alpha_map.request();

    if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
        throw std::invalid_argument("frame must be (H, W, 3) uint8");
    }
    if (frame_buf.size != blurred_buf.size) {
        throw std::invalid_argument("frame and blurred must have the same size");
    }

    int height = static_cast<int>(frame_buf.shape[0]);
    int width = static_cast<int>(frame_buf.shape[1]);

    if (alpha_buf.size != static_cast<ssize_t>(height) * width) {
        throw std::invalid_argument("alpha_map must have H*W elements");
    }

    auto output = py::array_t<uint8_t>({height, width, 3});
    auto output_buf = output.request(true);

    AFBlendConfig config{width * height};
    apply_af_blend_neon(
        static_cast<const uint8_t*>(frame_buf.ptr),
        static_cast<const float*>(alpha_buf.ptr),
        static_cast<const uint8_t*>(blurred_buf.ptr),
        static_cast<uint8_t*>(output_buf.ptr),
        config
    );

    return output;
}

static py::array_t<uint8_t> py_apply_af_blend_scalar(
    py::array_t<uint8_t, py::array::c_style> frame,
    py::array_t<uint8_t, py::array::c_style> blurred,
    py::array_t<float, py::array::c_style> alpha_map
) {
    auto frame_buf = frame.request();
    auto blurred_buf = blurred.request();
    auto alpha_buf = alpha_map.request();

    if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
        throw std::invalid_argument("frame must be (H, W, 3) uint8");
    }
    if (frame_buf.size != blurred_buf.size) {
        throw std::invalid_argument("frame and blurred must have the same size");
    }

    int height = static_cast<int>(frame_buf.shape[0]);
    int width = static_cast<int>(frame_buf.shape[1]);

    if (alpha_buf.size != static_cast<ssize_t>(height) * width) {
        throw std::invalid_argument("alpha_map must have H*W elements");
    }

    auto output = py::array_t<uint8_t>({height, width, 3});
    auto output_buf = output.request(true);

    AFBlendConfig config{width * height};
    apply_af_blend_scalar(
        static_cast<const uint8_t*>(frame_buf.ptr),
        static_cast<const float*>(alpha_buf.ptr),
        static_cast<const uint8_t*>(blurred_buf.ptr),
        static_cast<uint8_t*>(output_buf.ptr),
        config
    );

    return output;
}

static py::tuple py_process_frame(
    py::array_t<uint8_t, py::array::c_style> frame,
    py::array_t<float, py::array::c_style> background,
    float alpha,
    uint8_t threshold,
    const py::object& lut_obj,
    float gain_b,
    float gain_g,
    float gain_r,
    const py::object& alpha_map_obj
) {
    auto frame_buf = frame.request();
    auto bg_buf = background.request(true);

    if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
        throw std::invalid_argument("frame must be (H, W, 3) uint8");
    }

    int height = static_cast<int>(frame_buf.shape[0]);
    int width = static_cast<int>(frame_buf.shape[1]);
    int pixels = width * height;
    int bytes = pixels * 3;

    if (bg_buf.size != static_cast<ssize_t>(bytes)) {
        throw std::invalid_argument("background must match frame size (H*W*3 float32)");
    }

    PipelineConfig config{};
    config.ema_alpha = alpha;
    config.motion_threshold = threshold;
    config.width = width;
    config.height = height;

    // ISP: AE+AWB always runs; identity LUT + unity gains when not provided
    bool has_lut = !lut_obj.is_none();
    bool has_alpha_map = !alpha_map_obj.is_none();

    if (has_lut) {
        auto lut = lut_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
        auto lut_buf = lut.request();
        if (lut_buf.size != 256) {
            throw std::invalid_argument("lut must have exactly 256 elements");
        }
        std::memcpy(config.ae_awb.lut, lut_buf.ptr, 256);
        config.ae_awb.gain_b = gain_b;
        config.ae_awb.gain_g = gain_g;
        config.ae_awb.gain_r = gain_r;
    } else {
        for (int i = 0; i < 256; i++) config.ae_awb.lut[i] = static_cast<uint8_t>(i);
        config.ae_awb.gain_b = 1.0f;
        config.ae_awb.gain_g = 1.0f;
        config.ae_awb.gain_r = 1.0f;
    }
    config.ae_awb.width = width;
    config.ae_awb.height = height;

    auto mask = py::array_t<uint8_t>({height, width});
    auto mask_buf = mask.request(true);

    auto display = py::array_t<uint8_t>({height, width, 3});
    auto display_buf = display.request(true);

    // AF always runs; use all-zeros alpha map (fully sharp) when not provided
    std::vector<float> default_alpha;
    const float* alpha_map_ptr = nullptr;
    if (has_alpha_map) {
        auto alpha_map = alpha_map_obj.cast<py::array_t<float, py::array::c_style>>();
        auto am_buf = alpha_map.request();
        if (am_buf.size != static_cast<ssize_t>(pixels)) {
            throw std::invalid_argument("alpha_map must have H*W elements");
        }
        alpha_map_ptr = static_cast<const float*>(am_buf.ptr);
    } else {
        default_alpha.resize(static_cast<size_t>(pixels), 0.0f);
        alpha_map_ptr = default_alpha.data();
    }

    process_frame(
        static_cast<const uint8_t*>(frame_buf.ptr),
        static_cast<float*>(bg_buf.ptr),
        static_cast<uint8_t*>(mask_buf.ptr),
        static_cast<uint8_t*>(display_buf.ptr),
        alpha_map_ptr,
        config
    );

    return py::make_tuple(mask, display);
}

PYBIND11_MODULE(streetscope_simd, m) {
    m.doc() = "StreetScope SIMD kernels";

    m.def("accumulate_ema", &py_accumulate_ema,
        py::arg("frame"), py::arg("background"), py::arg("alpha"),
        "EMA background accumulation. Modifies background in-place.");

    m.def("subtract_background", &py_subtract_background,
        py::arg("frame"), py::arg("background"), py::arg("threshold"),
        "Background subtraction (NEON). Returns binary motion mask.");

    m.def("accumulate_ema_scalar", &py_accumulate_ema_scalar,
        py::arg("frame"), py::arg("background"), py::arg("alpha"),
        "EMA background accumulation (scalar). Modifies background in-place.");

    m.def("subtract_background_scalar", &py_subtract_background_scalar,
        py::arg("frame"), py::arg("background"), py::arg("threshold"),
        "Background subtraction (scalar). Returns binary motion mask.");

    m.def("apply_ae_awb", &py_apply_ae_awb,
        py::arg("frame"), py::arg("lut"),
        py::arg("gain_b"), py::arg("gain_g"), py::arg("gain_r"),
        "Fused auto exposure + auto white balance (NEON). Returns corrected BGR frame.");

    m.def("apply_ae_awb_scalar", &py_apply_ae_awb_scalar,
        py::arg("frame"), py::arg("lut"),
        py::arg("gain_b"), py::arg("gain_g"), py::arg("gain_r"),
        "Fused auto exposure + auto white balance (scalar). Returns corrected BGR frame.");

    m.def("apply_af_blend", &py_apply_af_blend,
        py::arg("frame"), py::arg("blurred"), py::arg("alpha_map"),
        "AF detail blend (NEON). Returns sharpened BGR frame.");

    m.def("apply_af_blend_scalar", &py_apply_af_blend_scalar,
        py::arg("frame"), py::arg("blurred"), py::arg("alpha_map"),
        "AF detail blend (scalar). Returns sharpened BGR frame.");

    m.def("process_frame", &py_process_frame,
        py::arg("frame"), py::arg("background"),
        py::arg("alpha"), py::arg("threshold"),
        py::arg("lut") = py::none(),
        py::arg("gain_b") = 1.0f, py::arg("gain_g") = 1.0f, py::arg("gain_r") = 1.0f,
        py::arg("alpha_map") = py::none(),
        "Fused pipeline: EMA + subtract + ISP (AE+AWB+AF).\n"
        "Returns (mask, display_frame).\n"
        "ISP always runs. lut=None uses identity LUT. alpha_map=None uses zero alpha (fully sharp).");

    py::class_<streetscope::PlanExecutor>(m, "PlanExecutor")
        .def(py::init([](const py::bytes& plan_data) {
            std::string data = plan_data;
            return std::make_unique<streetscope::PlanExecutor>(
                reinterpret_cast<const uint8_t*>(data.data()),
                data.size()
            );
        }), py::arg("plan_data"), "Create a PlanExecutor from a plan binary.")

        .def("run_frame", [](streetscope::PlanExecutor& self,
                py::array_t<uint8_t, py::array::c_style> frame,
                py::array_t<float, py::array::c_style> background,
                float alpha, uint8_t threshold,
                const py::object& lut_obj,
                float gain_b, float gain_g, float gain_r,
                const py::object& alpha_map_obj,
                bool timing
            ) {
                auto frame_buf = frame.request();
                auto bg_buf = background.request(true);  // writable

                if (frame_buf.ndim != 3 || frame_buf.shape[2] != 3) {
                    throw std::invalid_argument("frame must be (H, W, 3) uint8");
                }

                int height = static_cast<int>(frame_buf.shape[0]);
                int width = static_cast<int>(frame_buf.shape[1]);
                int pixels = width * height;

                if (width != self.width() || height != self.height()) {
                    throw std::invalid_argument("frame dimensions must match plan dimensions");
                }

                // Build PipelineConfig
                PipelineConfig config{};
                config.ema_alpha = alpha;
                config.motion_threshold = threshold;
                config.width = width;
                config.height = height;

                bool has_lut = !lut_obj.is_none();
                bool has_alpha_map = !alpha_map_obj.is_none();

                if (has_lut) {
                    auto lut = lut_obj.cast<py::array_t<uint8_t, py::array::c_style>>();
                    auto lut_buf = lut.request();
                    if (lut_buf.size != 256) {
                        throw std::invalid_argument("lut must have exactly 256 elements");
                    }
                    std::memcpy(config.ae_awb.lut, lut_buf.ptr, 256);
                    config.ae_awb.gain_b = gain_b;
                    config.ae_awb.gain_g = gain_g;
                    config.ae_awb.gain_r = gain_r;
                } else {
                    for (int i = 0; i < 256; i++) config.ae_awb.lut[i] = static_cast<uint8_t>(i);
                    config.ae_awb.gain_b = 1.0f;
                    config.ae_awb.gain_g = 1.0f;
                    config.ae_awb.gain_r = 1.0f;
                }
                config.ae_awb.width = width;
                config.ae_awb.height = height;

                auto mask = py::array_t<uint8_t>({height, width});
                auto mask_buf = mask.request(true);

                auto display = py::array_t<uint8_t>({height, width, 3});
                auto display_buf = display.request(true);

                // AF always runs; use all-zeros alpha map (fully sharp) when not provided
                std::vector<float> default_alpha;
                const float* alpha_map_ptr = nullptr;
                if (has_alpha_map) {
                    auto alpha_map = alpha_map_obj.cast<py::array_t<float, py::array::c_style>>();
                    auto am_buf = alpha_map.request();
                    if (am_buf.size != static_cast<ssize_t>(pixels)) {
                        throw std::invalid_argument("alpha_map must have H*W elements");
                    }
                    alpha_map_ptr = static_cast<const float*>(am_buf.ptr);
                } else {
                    default_alpha.resize(static_cast<size_t>(pixels), 0.0f);
                    alpha_map_ptr = default_alpha.data();
                }

                // Optional timing
                std::vector<uint64_t> times;
                uint64_t* times_ptr = nullptr;
                if (timing) {
                    times.resize(self.num_stages(), 0);
                    times_ptr = times.data();
                }

                self.run_frame(
                    static_cast<const uint8_t*>(frame_buf.ptr),
                    static_cast<float*>(bg_buf.ptr),
                    static_cast<uint8_t*>(mask_buf.ptr),
                    static_cast<uint8_t*>(display_buf.ptr),
                    alpha_map_ptr,
                    config,
                    times_ptr
                );

                if (timing) {
                    // Return (mask, display, times_list)
                    py::list py_times;
                    for (auto t : times) py_times.append(t);
                    return py::make_tuple(mask, display, py::object(py_times));
                }
                return py::make_tuple(mask, display, py::object(py::none()));
            },
            py::arg("frame"), py::arg("background"),
            py::arg("alpha"), py::arg("threshold"),
            py::arg("lut") = py::none(),
            py::arg("gain_b") = 1.0f, py::arg("gain_g") = 1.0f, py::arg("gain_r") = 1.0f,
            py::arg("alpha_map") = py::none(),
            py::arg("timing") = false,
            "Run the execution plan for one frame.\n"
            "Returns (mask, display, stage_times_or_None)."
        )

        .def_property_readonly("width", &streetscope::PlanExecutor::width)
        .def_property_readonly("height", &streetscope::PlanExecutor::height)
        .def_property_readonly("arena_size", &streetscope::PlanExecutor::arena_size)
        .def_property_readonly("num_stages", &streetscope::PlanExecutor::num_stages);
}
