"""Stream discovery via ffprobe. Probes an HLS URL and returns a StreamProfile."""

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamProfile:
    width: int
    height: int
    frame_rate: float
    codec: str
    pixel_format: str
    bitrate_kbps: int
    color_matrix: str
    is_live: bool

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / self.frame_rate

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def total_pixels(self) -> int:
        return self.width * self.height


def parse_frame_rate(rate_str: str) -> float | None:
    """Parse ffprobe frame rate string like '15/1' or '30000/1001'."""
    if not rate_str or rate_str == "0/0":
        return None
    parts = rate_str.split("/")
    if len(parts) == 2:
        num, den = int(parts[0]), int(parts[1])
        if den == 0:
            return None
        return num / den
    return float(rate_str)


def extract_bitrate(ffprobe_data: dict, video_stream: dict) -> int:
    """Extract bitrate in kbps from ffprobe output, checking multiple locations."""
    # Check video stream tags for variant_bitrate
    tags = video_stream.get("tags", {})
    if "variant_bitrate" in tags:
        return int(tags["variant_bitrate"]) // 1000

    # Check all streams for variant_bitrate
    for stream in ffprobe_data.get("streams", []):
        stream_tags = stream.get("tags", {})
        if "variant_bitrate" in stream_tags:
            return int(stream_tags["variant_bitrate"]) // 1000

    # Fall back to format-level bit_rate
    fmt = ffprobe_data.get("format", {})
    if "bit_rate" in fmt:
        return int(fmt["bit_rate"]) // 1000

    return 0


def parse_ffprobe_output(ffprobe_data: dict) -> StreamProfile:
    """Parse ffprobe JSON output into a StreamProfile."""
    video_stream = None
    for stream in ffprobe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError("No video stream found in ffprobe output")

    # Frame rate: prefer r_frame_rate, fall back to avg_frame_rate
    frame_rate = parse_frame_rate(video_stream.get("r_frame_rate", ""))
    if frame_rate is None:
        frame_rate = parse_frame_rate(video_stream.get("avg_frame_rate", ""))
    if frame_rate is None:
        raise ValueError("Could not determine frame rate from ffprobe output")

    # Live detection: duration "0" or missing means live
    fmt = ffprobe_data.get("format", {})
    duration_str = fmt.get("duration", "0")
    try:
        duration = float(duration_str)
        is_live = duration == 0
    except (ValueError, TypeError):
        is_live = True

    return StreamProfile(
        width=video_stream["width"],
        height=video_stream["height"],
        frame_rate=frame_rate,
        codec=video_stream["codec_name"],
        pixel_format=video_stream.get("pix_fmt", "unknown"),
        bitrate_kbps=extract_bitrate(ffprobe_data, video_stream),
        color_matrix=video_stream.get("color_space", "unknown"),
        is_live=is_live,
    )


def probe_stream(url: str, timeout: float = 10.0) -> StreamProfile:
    """Probe an HLS stream URL and return its StreamProfile."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    return parse_ffprobe_output(data)
