#!/usr/bin/env bash
for p in $(pgrep -u $USER -f '[w]atch_resstudy3'); do kill $p 2>/dev/null; done
echo "remaining: $(pgrep -u $USER -f '[w]atch_resstudy3' | wc -l)"
