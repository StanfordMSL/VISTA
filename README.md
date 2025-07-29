# VISTA
ArXiv: <https://arxiv.org/abs/2507.01125>.

Repository for VISTA, including Real-Time Semantic Gaussian Splatting, voxel grid generation, and semantic exploration & planning, presented in our paper. This repository includes scripts and ROS2 nodes required for running quadrotor experiments.

## Requirements
- A Linux machine (tested with Ubuntu 22.04)
    - This should also have a CUDA capable GPU, and fairly strong CPU.
- ROS2 Humble installed on your Linux machine
- A camera that is compatible with ROS2 Humble
- Some means by which to estimate pose of the camera (SLAM, motion capture system, etc)

## Organization 
This codebase uses git submodules to reference three different code bases:
- [Semantic-SplatBridge](https://github.com/StanfordMSL/Semantic-SplatBridge): Performs Real-Time 3D Gaussian Splatting given ROS streams of camera image and pose data.
- [VISTA-Map](https://github.com/StanfordMSL/VISTA-Map): Generates voxel grid environment with RGB, semantic, and epistemic uncertainty information from pointclouds. Uses Voxel Traversal mechanism to differentiate between unobserved, observed-free, and observed-occupied space.
- [VISTA-Plan](https://github.com/StanfordMSL/VISTA-Plan): Proposes plans in voxel grid environment by prioritizing frontier voxels and highly semantically similar regions of the environment. Proposed plans are sampled from a Gaussian Mixture Model and then scored to choose the best trajectory that has high information gain with high semantic similarity.

## Installation
Each of these submodules have their own instructions for how to use them independently in case you want to use only a part of our pipeline. If you want to replicate the full pipeline, follow these instructions.

1. Install ROS2 Humble using the [installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html). After installing ROS2 we recommend adding `source /opt/ros/humble/setup.bash` to your bashrc file. Otherwise, this command needs to be run in each terminal before running any ros2 commands.
2. Install Miniconda (or Anaconda) using the [installation guide](https://www.anaconda.com/docs/main). 
3. Create a conda environment for VISTA. Take note of the optional procedure for completely isolating the conda environment from your machine's base python site packages. For more details see this [StackOverflow post](https://stackoverflow.com/questions/25584276/how-to-disable-site-enable-user-site-for-an-environment) and [PEP-0370](https://peps.python.org/pep-0370/). 
    ```bash
    conda create --name vista -y python=3.10

    conda activate vista
    conda env config vars set PYTHONNOUSERSITE=1
    conda deactivate
    ```
4. Activate the conda environment and install Nerfstudio dependencies.
    ```bash
    # Activate conda environment, and upgrade pip
    conda activate vista
    python -m pip install --upgrade pip

    # PyTorch, Torchvision dependency
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

    # CUDA dependency (by far the easiest way to manage cuda versions)
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

    # TinyCUDANN dependency (takes a while!)
    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```
5. Clone, and install VISTA, intializing all submodules.
    ```bash
    git clone git@github.com:StanfordMSL/VISTA.git
    cd VISTA
    git submodule init
    git submodule update
    ```
6. Now you should have all three repositories that make up vista in the `VISTA/src/` directory. We'll now install the dependencies for each of the submodules starting with [Semantic-SplatBridge](https://github.com/StanfordMSL/Semantic-SplatBridge). 
    ```bash
    # Make sure your conda env is activated!
    cd Semantic-SplatBridge
    pip install -e . 
    ```
7. Install gsplat and ensure correct numpy version. 
    ```bash
    # Numpy 2.0 bricks the current install...
    pip install numpy==1.26.3

    # Uninstall the JIT version of Gsplat 1.0.0
    pip uninstall gsplat

    # Build GSPLAT 1.0.0
    pip install git+https://github.com/nerfstudio-project/gsplat.git@c7b0a383657307a13dff56cb2f832e3ab7f029fd

    # After this step you can never build GSplat again in the conda env so be careful!
    # Fix ROS2 CMake version dep
    conda install -c conda-forge gcc=12.1.0
    ```
8. Now we'll install the dependencies for [VISTA-Map](https://github.com/StanfordMSL/VISTA-Map) submodule.
    ```bash
    # Navigate to the VISTA-Map repo and pip install
    cd ../VISTA-Map
    pip install -e .
    ```
9. And the [VISTA-Plan](https://github.com/StanfordMSL/VISTA-Plan) submodule. Note that this module is a ROS2 package.
    ```bash
    # Build and source the repo
    cd ../VISTA-Plan
    colcon build --symlink-install
    source install/setup.bash

    # pip install
    cd src/vista_plan
    pip install -e .
    ```

## Run ROS2 example with rosbag
This example simulates streaming data from a robot by replaying a [ROS2 bag](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html), and using Semantic-SplatBridge to train a Splatfacto model on that data. This example is a great way to check that your installation is working correctly.

These instructions are designed to be run in three different terminal sessions. They will be reffered to as terminals 1, 2, and 3.

1. [Terminal 1] Download the example `wagon` rosbag using the provided download script. This script creates a folder `VISTA/src/Semantic-SplatBridge/rosbags`, and downloads a rosbag to that directory.
    ```bash
    # Activate conda env
    activate vista

    # Run download script
    cd VISTA/src/Semantic-SplatBridge
    python scripts/download_data.py
    ```

2. [Terminal 2] Start the `wagon` rosbag paused. The three important ROS topics in this rosbag are an image topic (`/zed/zed_node/rgb/image_rect_color/compressed`), a depth topic (`/zed/zed_node/depth/depth_registered`), and a camera pose topic (`/vrpn_mocap/fmu/pose`). These will be the data streams that will be used to train the Splatfacto model using Semantic-SplatBridge.

    ```bash
    # NOTE: for this example, Terminal 2 does not need to use the conda env.

    # Play the rosbag, but start it paused.
    cd VISTA/src/Semantic-SplatBridge/rosbags
    ros2 bag play wagon --start-paused
    ```

3. [Terminal 1] Start Semantic-SplatBridge using the Nerfstudio `ns-train` CLI.

    ```bash
    ns-train ros-depth-splatfacto --data configs/zed_mocap.json
    ```

    After some initialization a message will appear stating that `(NerfBridge) Waiting to recieve 5 images...`, at this point you should open the Nerfstudio viewer in a browser tab to visualize the Splatfacto model as it trains. Training will start after completing the next step.

4. [Terminal 3] We then need to open a terminal to run our planner. We'll need to activate our conda environment, and add the paths of the conda environment and vista repo to the python path. Note that your paths may be different from what is provided here:
    ```bash
    conda activate vista
    cd VISTA/src/VISTA-Plan
    source install/setup.bash
    export PYTHONPATH="${PYTHONPATH}:/home/<user>/miniconda3/envs/vistaplan/lib/python3.10/site-packages"
    export PYTHONPATH="${PYTHONPATH}:/home/<user>/VISTA-Map"
    ```

5. [Terminal 3] In the same terminal, run the launch file:
    ```bash
    ros2 launch vista_plan waypoint_publisher.launch.py
    ```

6. [Terminal 2] Press the SPACE key to start playing the rosbag. Once the pre-training image buffer is filled then training should commence, and the usual Nerfstudio print messages will appear in Terminal 1. After a few seconds the Nerfstudio viewer should start to show the recieved images as camera frames, and the Splatfacto model should begin be filled out.

7. After the rosbag in Terminal 2 finishes playing Semantic-SplatBridge will continue training the Splatfacto model on all of the data that it has recieved, but no new data will be added. You can use CTRL+c to kill Semantic-SplatBridge and VISTA-Plan after you are done inspecting the Splatfacto model in the viewer.


## Run the system with hardware
We use a custom [StanfordMSL Drone Hardware Platform](https://github.com/StanfordMSL/TrajBridge/wiki/3.-Drone-Hardware) that uses a PX4 flightcontroller. State machine used on this platform can be found at https://github.com/StanfordMSL/TrajBridge.

First move the robot to the desired start location. In the case of the drone, get drone in a hover in a state where it is waiting for waypoint commands.

A different config file can be created for each type of robot system being used. This is where the ros topic names are defined for the camera image and pose information streams. In the case of our drones, we use:
```bash
conda activate vista
cd VISTA/src/Semantic-SplatBridge
ns-train ros-depth-splatfacto --data configs/mocap_endor.json
```

Once the terminal in Terminal 1 shows that the Gaussian Splat is waiting to receive images, in Terminal 3, specifying the correct robot platform:
```bash
conda activate vista
cd VISTA/src/VISTA-Plan
source install/setup.bash
export PYTHONPATH="${PYTHONPATH}:/home/<user>/miniconda3/envs/vistaplan/lib/python3.10/site-packages"
export PYTHONPATH="${PYTHONPATH}:/home/<user>/VISTA-Map"
ros2 launch vista_plan waypoint_publisher.launch.py
```

## Citation
In case anyone does use VISTA as a starting point for any research please cite our paper.
```
# --------------------------- VISTA ---------------------
@article{nagami2025vista,
    title={VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting}, 
    author={Keiko Nagami and Timothy Chen and Javier Yu and Ola Shorinwa and Maximilian Adang and Carlyn Dougherty and Eric Cristofalo and Mac Schwager},
    journal={arXiv preprint arXiv:2507.01125}
    year={2025},
}
```
