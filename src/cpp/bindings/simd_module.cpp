#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <streetscope/simd/accumulator.h>
#include <streetscope/simd/subtractor.h>
#include <streetscope/simd/isp_correction.h>
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
}
