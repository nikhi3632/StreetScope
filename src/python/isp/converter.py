"""Convert ISPParams to MetalToneMapper Reinhard parameters."""

import numpy as np

from src.python.isp.estimator import ISPParams


def isp_to_reinhard(params: ISPParams) -> dict:
    """Map ISPEstimator's scene analysis to Reinhard tone map parameters.

    ISPEstimator computes a gamma LUT and AWB gains from the background plate.
    This function derives Reinhard params from the same analysis:

    - exposure: inverse of the AE gamma. Gamma < 1 (brighten) -> exposure > 1.
      Gamma > 1 (darken) -> exposure < 1.
    - white_point: derived from exposure. Higher exposure needs higher white
      point to avoid clamping highlights.
    - gamma: 1.0 for sRGB camera feeds (already gamma-encoded).
    - gains: AWB gains pass through directly (BGR order).
    """
    # Reconstruct gamma from the LUT.
    # The LUT is built as: lut[i] = (i/255)^gamma * 255
    # Sample at mid-gray (128) to recover gamma:
    #   lut[128] / 255 = (128/255)^gamma
    #   gamma = log(lut[128]/255) / log(128/255)
    lut_mid = float(params.auto_exposure_lut[128])
    if lut_mid <= 0 or lut_mid >= 255:
        gamma = 1.0
    else:
        gamma = np.log(lut_mid / 255.0) / np.log(128.0 / 255.0)

    # Exposure is the inverse relationship: lower gamma = more brightening = higher exposure
    exposure = 1.0 / max(gamma, 0.01)

    # White point: scale with exposure so highlights compress proportionally.
    # At exposure=1 (no correction), white_point=1 (no highlight compression).
    # At exposure=2 (dark scene doubled), white_point=2 (allow wider range).
    white_point = max(exposure, 1.0)

    gains = params.auto_white_balance_gains  # [B, G, R]

    return {
        "exposure": float(np.clip(exposure, 0.2, 5.0)),
        "white_point": float(np.clip(white_point, 1.0, 5.0)),
        "gamma": 1.0,  # sRGB camera feeds are already gamma-encoded
        "gain_b": float(gains[0]),
        "gain_g": float(gains[1]),
        "gain_r": float(gains[2]),
    }
