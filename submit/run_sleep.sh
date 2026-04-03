#!/bin/bash
# Sleep script for interactive debugging on L20 cluster.
# Submit with: python submit.py --cluster carizon-4dgt-gpu --gpu 2 --sleep
echo "Sleep mode - use tmux for debugging"
echo "Code at: /running_package/SuperGluePretrainedNetwork/"
echo "Data at: /bucket/input/"
echo "Output:  /job_data/"
echo "Logs:    /job_log/"
sleep 20000m
