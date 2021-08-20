# fsds-driverless-system
Autonomous software stack built using Python3 and implemented on the Formula Student Driverless Simulator

## System Design
![System design](https://github.com/mkamalel/fsds-driverless-system/blob/main/docs/images/fsds-system.png)

Documentation of the code is provided in the python files. Pretrained Yolov4 weights for detecting cones are provided in the yolov4-mit folder.
- Original Code
  - trackmap.py
    - A class that defines a map of track using cone observations
  - path_tracking.py
    - A class that tracks a path using a Stanley Controller
  - track_explore.py
    - A class used to explore a track of cones and generate a map using FastSLAM
  - driverless.py
    - Main driverless application for use with Formula Student Driverless Simulation
- Modified Code
  - fastslam.py
    - Added labels to landmark observations
  - stanley.py
    - Added look ahead for stanley controller
- Unmodified Code
  - darknet.py


## Getting Started
### Prerequisites
- Linux
- GPU
- OpenCV
- Python3

### Python packages
- simple_pid
- matplotlib
- cv2
- numpy
- scipy
- collections
- copy
- seaborn
- scikit-learn
- pandas

To install the Formula Student Driverless Simulator, refer to this [guide](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator/blob/master/docs/getting-started.md)

To compile darknet on a user's computer, refer to this [guide](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make)

Once the requirements are fulfilled, run FSDS.sh from the Formula Student Driverless Simulator folder and run driverless.py
