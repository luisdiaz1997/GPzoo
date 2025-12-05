#!/bin/bash
# Run all Slideseq training scripts in parallel
# Each script runs in the background with its own GPU assignment

# Conda environment python
PYTHON="/gladstone/engelhardt/home/lchumpitaz/miniconda3/envs/gpzoo/bin/python"

echo "Starting all Slideseq training scripts..."
echo "Using Python: $PYTHON"
echo ""

# GPU 0: VNNGP models
echo "Starting VNNGP (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -m gpzoo.datasets.slideseq.vnngp_nsf &
PID1=$!

echo "Starting VNNGP-MGGP (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -m gpzoo.datasets.slideseq.vnngp_mggp_nsf &
PID2=$!

# GPU 1: SVGP models
echo "Starting SVGP (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -m gpzoo.datasets.slideseq.svgp_nsf &
PID3=$!

echo "Starting SVGP-MGGP (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -m gpzoo.datasets.slideseq.svgp_mggp_nsf &
PID4=$!

echo ""
echo "All jobs started in background!"
echo "Process IDs: $PID1 $PID2 $PID3 $PID4"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "To check if processes are still running:"
echo "  ps -p $PID1,$PID2,$PID3,$PID4"
echo ""
echo "To kill all jobs:"
echo "  kill $PID1 $PID2 $PID3 $PID4"
echo ""
echo "Training logs will be saved to: models/slideseq/"
echo "TensorBoard logs: tensorboard --logdir models/slideseq/tb/"