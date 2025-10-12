# leg-odometry
Proprioceptive state estimation for legged robots using IMU and joint encoders.

Files and code can be downloaded in "release"

Make sure you have the right python enviroment before running the code

Tips on how to run the code:
  I'm using Anoconda Prompt to run it, first switch to the right python enviroment, then set the working root at where the main.py is, then using this line:
  python -u main.py --config "your\path\to\go2.yaml" --bagpath "your\path\to\rosbag2_2025_02_19-22_46_51(here is the given test file)" --topic /lowstate

The result for test file should be:
<img width="600" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/5e6b3e12-aabd-4f6f-8c71-4b553476a118" />
