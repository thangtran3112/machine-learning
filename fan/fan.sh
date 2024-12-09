#!/bin/bash

# Use sudo to elevate privileges and activate the virtual environment
# CPU fan
# sudo echo 1 > /sys/devices/platform/asus-nb-wmi/hwmon/hwmon7/pwm1_enable
# throttle policy
sudo echo 1 > /sys/devices/platform/asus-nb-wmi/throttle_thermal_policy

sudo bash <<EOF
source /home/thangtran3112/fan/bin/activate
python /home/thangtran3112/fan/nvidia_fan_control.py
deactivate
EOF
