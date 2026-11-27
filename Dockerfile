FROM registry.cookcloud.ru/base_images/ros2-base-humble-amd64:v1.0.1

ENV DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-tf2-ros \
        python3-pip \
    && pip3 install --no-cache-dir "numpy<2" mujoco scipy \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -U -s /bin/bash appuser

COPY unitree_go_pkg/unitree_go /opt/unitree_go
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod 755 /entrypoint.sh \
    && chmod -R a+rX /opt/unitree_go

WORKDIR /home/appuser/ws
COPY model ./model
COPY rl_training/artifacts/best_policy_weights.npz ./policy_weights.npz
COPY safety_filter.py ./safety_filter.py
COPY odom_node.py ./odom_node.py
COPY rl_node.py ./rl_node.py
RUN chown -R appuser:appuser /home/appuser \
    && chmod 755 /home/appuser /home/appuser/ws ./safety_filter.py ./odom_node.py ./rl_node.py \
    && chmod -R a+rX /home/appuser/ws

USER appuser
ENV HOME=/tmp \
    ROS_HOME=/tmp/.ros

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ping"]
