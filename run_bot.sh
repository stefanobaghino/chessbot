#!/usr/bin/env bash
# Start the Lichess bot with the project's virtualenv.
cd "$(dirname "$0")" && exec .venv/bin/python -m bot.lichess_bot "$@"
