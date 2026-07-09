#!/bin/bash
set -euo pipefail

cp chat_template.jinja "$WORKSPACE_DIR/gemma4_audio_chat_template.jinja"
echo "=======> Gemma 4 audio chat template installed at $WORKSPACE_DIR/gemma4_audio_chat_template.jinja"
