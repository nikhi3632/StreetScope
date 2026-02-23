.PHONY: test lint fix format check play detect detect-mask background capture pipeline

# ── Testing ──────────────────────────────────────────────
test:
	.venv/bin/python -m pytest tests/ -v

test-q:
	.venv/bin/python -m pytest tests/ -q

# ── Linting ──────────────────────────────────────────────
lint:
	.venv/bin/ruff check src/ tests/ tools/

fix:
	.venv/bin/ruff check --fix src/ tests/ tools/

format:
	.venv/bin/ruff format src/ tests/ tools/

check: lint test

# ── Tools (URL required) ────────────────────────────────
URL ?= https://s52.nysdot.skyvdn.com/rtplive/R5_011/playlist.m3u8

play:
	.venv/bin/python tools/stream_player.py --url "$(URL)"

detect:
	.venv/bin/python tools/detection_viewer.py --url "$(URL)" --vehicles-only

detect-mask:
	.venv/bin/python tools/detection_viewer.py --url "$(URL)" --vehicles-only --show-mask

background:
	.venv/bin/python tools/background_viewer.py --url "$(URL)"

pipeline:
	.venv/bin/python tools/pipeline_viewer.py --url "$(URL)" --vehicles-only

capture:
	.venv/bin/python tools/capture_frames.py --url "$(URL)" --count 200

# ── Setup ────────────────────────────────────────────────
install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
