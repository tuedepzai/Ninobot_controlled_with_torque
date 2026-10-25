
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
# NVIDIA GPU rendering through WSLg
ENV MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
ENV GALLIUM_DRIVER=d3d12
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
# ROS 2, Gazebo, RViz and development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-venv \
    x11-apps \
    mesa-utils \
    ros-jazzy-ros-gz \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-teleop-twist-keyboard \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-controller-manager \
    ros-jazzy-effort-controllers \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-ros2controlcli \
    ros-jazzy-rviz2 \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-slam-toolbox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws

# Load ROS in every interactive bash shell
RUN echo "source /opt/ros/jazzy/setup.bash" \
    >> /root/.bashrc && \
    echo '[ ! -f /ros2_ws/install/setup.bash ] || source /ros2_ws/install/setup.bash' \
    >> /root/.bashrc

CMD ["/bin/bash"]
