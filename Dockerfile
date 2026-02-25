FROM python:3.14-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake make g++ valgrind git \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir pybind11 \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Source is mounted at runtime: docker run -v $(PWD):/app ...
# Build artifacts go to build-docker/ (separate from macOS build/)
