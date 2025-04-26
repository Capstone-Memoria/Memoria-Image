#!/bin/bash
PID=$(pgrep -f "uvicorn main:app")
if [ -z "$PID" ]; then
  echo "Process not found."
else
  kill $PID
  echo "Process $PID killed."
fi 