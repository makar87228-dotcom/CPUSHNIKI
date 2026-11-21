FROM registry.cookcloud.ru/base_images/ros2-base-humble-amd64:v1.0.1

ENV DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp \
        python3-pip \
    && pip3 install --no-cache-dir "numpy<2" mujoco \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -U -s /bin/bash appuser

COPY unitree_go_pkg/unitree_go /opt/unitree_go
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod 755 /entrypoint.sh

WORKDIR /home/appuser/ws
COPY model ./model
COPY safety_filter.py ./safety_filter.py
RUN chmod 755 ./safety_filter.py && chown -R appuser:appuser /home/appuser

USER appuser
ENV HOME=/home/appuser \
    ROS_HOME=/tmp/.ros

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ping"]
