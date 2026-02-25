import json

import pytest

from src.python.bootstrap.stream_discovery import StreamProfile, parse_ffprobe_output

SAMPLE_FFPROBE_OUTPUT = {
    "streams": [
        {
            "index": 0,
            "codec_name": "timed_id3",
            "codec_type": "data",
            "tags": {"variant_bitrate": "231365"},
        },
        {
            "index": 1,
            "codec_name": "h264",
            "codec_type": "video",
            "profile": "Constrained Baseline",
            "width": 320,
            "height": 240,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "15/1",
            "avg_frame_rate": "15/1",
            "color_space": "bt709",
            "tags": {"variant_bitrate": "231365"},
        },
    ],
    "format": {"format_name": "hls", "duration": "0", "probe_score": 100},
}


class TestStreamProfile:
    def test_frame_budget_at_15fps(self):
        profile = StreamProfile(
            width=320,
            height=240,
            frame_rate=15.0,
            codec="h264",
            pixel_format="yuv420p",
            bitrate_kbps=231,
            color_matrix="bt709",
            is_live=True,
        )
        assert profile.frame_budget_ms == pytest.approx(66.67, abs=0.01)

    def test_frame_budget_at_30fps(self):
        profile = StreamProfile(
            width=640,
            height=480,
            frame_rate=30.0,
            codec="h264",
            pixel_format="yuv420p",
            bitrate_kbps=500,
            color_matrix="bt709",
            is_live=True,
        )
        assert profile.frame_budget_ms == pytest.approx(33.33, abs=0.01)

    def test_resolution_tuple(self):
        profile = StreamProfile(
            width=352,
            height=240,
            frame_rate=15.0,
            codec="h264",
            pixel_format="yuv420p",
            bitrate_kbps=175,
            color_matrix="smpte170m",
            is_live=True,
        )
        assert profile.resolution == (352, 240)

    def test_total_pixels(self):
        profile = StreamProfile(
            width=320,
            height=240,
            frame_rate=15.0,
            codec="h264",
            pixel_format="yuv420p",
            bitrate_kbps=231,
            color_matrix="bt709",
            is_live=True,
        )
        assert profile.total_pixels == 76800


class TestParseFFprobeOutput:
    def test_parses_video_stream(self):
        profile = parse_ffprobe_output(SAMPLE_FFPROBE_OUTPUT)
        assert profile.width == 320
        assert profile.height == 240
        assert profile.frame_rate == 15.0
        assert profile.codec == "h264"
        assert profile.pixel_format == "yuv420p"
        assert profile.color_matrix == "bt709"

    def test_extracts_bitrate_from_variant_tag(self):
        profile = parse_ffprobe_output(SAMPLE_FFPROBE_OUTPUT)
        assert profile.bitrate_kbps == 231

    def test_detects_live_stream(self):
        profile = parse_ffprobe_output(SAMPLE_FFPROBE_OUTPUT)
        assert profile.is_live is True

    def test_detects_non_live_from_finite_duration(self):
        data = json.loads(json.dumps(SAMPLE_FFPROBE_OUTPUT))
        data["format"]["duration"] = "120.5"
        profile = parse_ffprobe_output(data)
        assert profile.is_live is False

    def test_raises_on_no_video_stream(self):
        data = {
            "streams": [{"codec_type": "audio", "codec_name": "aac"}],
            "format": {"format_name": "hls"},
        }
        with pytest.raises(ValueError, match="No video stream"):
            parse_ffprobe_output(data)

    def test_parses_fractional_frame_rate(self):
        data = json.loads(json.dumps(SAMPLE_FFPROBE_OUTPUT))
        data["streams"][1]["r_frame_rate"] = "30000/1001"
        profile = parse_ffprobe_output(data)
        assert profile.frame_rate == pytest.approx(29.97, abs=0.01)

    def test_falls_back_to_avg_frame_rate(self):
        data = json.loads(json.dumps(SAMPLE_FFPROBE_OUTPUT))
        data["streams"][1]["r_frame_rate"] = "0/0"
        data["streams"][1]["avg_frame_rate"] = "12/1"
        profile = parse_ffprobe_output(data)
        assert profile.frame_rate == 12.0

    def test_missing_color_space_defaults_to_unknown(self):
        data = json.loads(json.dumps(SAMPLE_FFPROBE_OUTPUT))
        del data["streams"][1]["color_space"]
        profile = parse_ffprobe_output(data)
        assert profile.color_matrix == "unknown"

    def test_bitrate_from_format_field(self):
        data = json.loads(json.dumps(SAMPLE_FFPROBE_OUTPUT))
        del data["streams"][1]["tags"]
        data["streams"][0]["tags"] = {}
        data["format"]["bit_rate"] = "350000"
        profile = parse_ffprobe_output(data)
        assert profile.bitrate_kbps == 350
