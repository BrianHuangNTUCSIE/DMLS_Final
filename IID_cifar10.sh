#!/bin/bash

# Check if a model name was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <ModelName>"
  exit 1
fi

MODEL_NAME=$1

# Run all commands in background with specified model
python3 IID_cifar10.py --model "$MODEL_NAME" --rounds 100 --delta 0 &
python3 IID_cifar10.py --model "$MODEL_NAME" --rounds 100 --delta 0.2 &
python3 IID_cifar10.py --model "$MODEL_NAME" --rounds 100 --delta 0.4 &
python3 IID_cifar10.py --model "$MODEL_NAME" --rounds 100 --delta 1.0 &

# Wait for all background processes to finish
wait

echo "All processes completed."