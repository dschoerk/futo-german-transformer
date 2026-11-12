#!/usr/bin/env bash
# usage: thermal_guard.sh <pattern> <max_cpu_celsius> <logfile> [resume_cpu_celsius] [max_gpu_celsius]
# Samples CPU (Tctl) and GPU temps every 10s while a *python* process whose command line matches <pattern> runs.
# Above the limit it PAUSES those processes (SIGSTOP) and resumes them (SIGCONT) once cooled down, so no work is lost.
# Only python processes are targeted: shells/watchers whose command line merely contains the pattern are ignored
# (pgrep -f would otherwise match this script and any waiting shell, and never exit / signal the wrong process).
pat=$1; max=$2; log=$3; resume=${4:-$((max - 10))}; gmax=${5:-88}
paused=0
targets() { for p in $(pgrep -f -- "$pat"); do [[ "$(ps -o comm= -p "$p" 2>/dev/null)" == python* ]] && echo "$p"; done; }
signal_all() { local t; t=$(targets); [ -n "$t" ] && kill "-$1" $t 2>/dev/null; }
cleanup() { [ $paused -eq 1 ] && signal_all CONT; exit 0; }
trap cleanup INT TERM
while [ -n "$(targets)" ]; do
  t=$(sensors 2>/dev/null | awk '/Tctl/ {gsub(/[+°C]/,"",$2); print int($2); exit}')
  g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null)
  echo "$(date +%T) cpu=${t}C gpu=${g}C paused=$paused" >> "$log"
  if [ $paused -eq 0 ] && { { [ -n "$t" ] && [ "$t" -ge "$max" ]; } || { [ -n "$g" ] && [ "$g" -ge "$gmax" ]; }; }; then
    echo "$(date +%T) HOT cpu=$t gpu=$g -> pausing '$pat'" >> "$log"; signal_all STOP; paused=1
  elif [ $paused -eq 1 ] && [ -n "$t" ] && [ "$t" -le "$resume" ] && { [ -z "$g" ] || [ "$g" -lt $((gmax - 5)) ]; }; then
    echo "$(date +%T) cooled cpu=$t gpu=$g -> resuming" >> "$log"; signal_all CONT; paused=0
  fi
  sleep 10
done
