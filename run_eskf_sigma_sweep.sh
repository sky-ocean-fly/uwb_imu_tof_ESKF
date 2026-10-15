#!/bin/bash
# ESKF sigma_uwb sweep experiment
# Tests different sigma_uwb values in the ESKF filter while keeping
# the true measurement noise at sigma_uwb=0.033 in the data generator.

set -e

# Configuration
WS_DIR="/home/yu/catkin_workspace"
ESKF_CONFIG="${WS_DIR}/src/uwb_pose_estimation/config/uwb_imu_tof_eskf_sim.yaml"
RESULT_DIR="${WS_DIR}/src/uwb_pose_estimation/sim_result"
LOG_DIR="${WS_DIR}/sigma_sweep_logs"
SIGMA_VALUES="0.033 0.1 0.2 0.5 1 5 20"
SIM_TIMEOUT_SEC=70

# Source workspace
source /opt/ros/noetic/setup.bash
source "${WS_DIR}/devel/setup.bash"

# Backup original config
cp "${ESKF_CONFIG}" "${ESKF_CONFIG}.bak"

# Create log directory
mkdir -p "${LOG_DIR}"

# Ensure roscore is running
if ! rostopic list &>/dev/null; then
    echo "Starting roscore..."
    roscore &
    ROSTORE_PID=$!
    sleep 5
    if ! rostopic list &>/dev/null; then
        echo "ERROR: Failed to start roscore"
        exit 1
    fi
    AUTO_KILL_ROSCORE=1
fi

echo "============================================"
echo "Starting sigma_uwb sweep experiment"
echo "Values: ${SIGMA_VALUES}"
echo "============================================"

for sigma in ${SIGMA_VALUES}; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> Running experiment with sigma_uwb = ${sigma}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    # Modify the sigma_uwb value in the config file
    sed -i "s/^sigma_uwb:.*/sigma_uwb: ${sigma}  #0.033 #0.1表示有正偏置的时候/" "${ESKF_CONFIG}"
    echo "Updated config: $(grep 'sigma_uwb' ${ESKF_CONFIG})"

    # Clear old CSV files
    rm -f "${RESULT_DIR}/eskf_pose_compare1.csv"
    rm -f "${RESULT_DIR}/optimization_pose_compare1.csv"

    # Launch simulation with timeout
    echo "Launching simulation (timeout: ${SIM_TIMEOUT_SEC}s)..."
    timeout --signal=SIGINT ${SIM_TIMEOUT_SEC} roslaunch uwb_pose_estimation simulation_both.launch \
        > "${LOG_DIR}/sigma_${sigma}.log" 2>&1 || true

    # Wait a moment for nodes to shut down
    sleep 3

    # Copy the result CSV
    CSV_SRC="${RESULT_DIR}/eskf_pose_compare1.csv"
    CSV_DST="${RESULT_DIR}/eskf_sigma_${sigma}.csv"
    if [ -f "${CSV_SRC}" ]; then
        cp "${CSV_SRC}" "${CSV_DST}"
        NUM_LINES=$(tail -n +2 "${CSV_DST}" | wc -l)
        echo "Results saved: ${CSV_DST} (${NUM_LINES} data rows)"

        # Extract RMSE from the log
        echo "--- ESKF RMSE ---"
        grep -A 10 "ESKF.*误差统计\|========== ESKF" "${LOG_DIR}/sigma_${sigma}.log" | tail -12 || echo "(check log file for details)"
    else
        echo "WARNING: No CSV file found for sigma_uwb=${sigma}!"
    fi

    # Also copy optimization CSV
    OPT_CSV_SRC="${RESULT_DIR}/optimization_pose_compare1.csv"
    OPT_CSV_DST="${RESULT_DIR}/optimization_sigma_${sigma}.csv"
    if [ -f "${OPT_CSV_SRC}" ]; then
        cp "${OPT_CSV_SRC}" "${OPT_CSV_DST}"
    fi

    echo ">>> Finished sigma_uwb = ${sigma}"
done

# Restore original config
cp "${ESKF_CONFIG}.bak" "${ESKF_CONFIG}"
rm -f "${ESKF_CONFIG}.bak"
echo ""
echo "============================================"
echo "All experiments complete!"
echo "Results in: ${RESULT_DIR}/eskf_sigma_*.csv"
echo "Logs in: ${LOG_DIR}/"
echo "============================================"

# Kill roscore if we started it
if [ -n "${AUTO_KILL_ROSCORE}" ]; then
    kill ${ROSTORE_PID} 2>/dev/null || true
fi
