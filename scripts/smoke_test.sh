#!/usr/bin/env bash
set -euo pipefail
python -m templateparity.cli --help
templateparity fixtures
templateparity demo
