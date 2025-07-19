#!/bin/bash

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -i:$port > /dev/null ; then
        return 0 # Port is in use
    else
        return 1 # Port is free
    fi
}

# Function to kill process using specific port
kill_port() {
    local port=$1
    echo "Killing process on port $port..."
    lsof -ti:$port | xargs kill -9 2>/dev/null
}

# Kill any existing process using port 8188 (ComfyUI)
if check_port 8188 ; then
    kill_port 8188
fi

# Kill any existing process using port 5002 (Flask)
if check_port 5002 ; then
    kill_port 5002
fi

# Wait for ports to be freed
sleep 2

# Start dream_layer.py
echo "Starting Dream Layer..."
cd "$(dirname "$0")"  # Change to script directory
python dream_layer.py "$@"  # Pass all command line arguments to dream_layer.py 