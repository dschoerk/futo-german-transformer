#!/usr/bin/env bash
# usage: thermal_guard.sh <pgrep-pattern> <max_celsius> <logfile>
# Samples CPU (Tctl) and GPU temps every 10s while a process matching the pattern runs; SIGTERMs it above the limit.
pat=$1; max=$2; log=$3
while pgrep -f "$pat" >/dev/null; do
  t=$(sensors 2>/dev/null | awk '/Tctl/ {gsub(/[+°C]/,"",$2); print int($2); exit}')
  g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null)
  echo "$(date +%T) cpu=${t}C gpu=${g}C" >> "$log"
  if { [ -n "$t" ] && [ "$t" -ge "$max" ]; } || { [ -n "$g" ] && [ "$g" -ge 88 ]; }; then
    echo "OVERHEAT cpu=$t gpu=$g, terminating '$pat'" >> "$log"; pkill -TERM -f "$pat"; exit 1
  fi
  sleep 10
done
