xhost +
docker run --rm --gpus all -it -p 8888:8888 --mount source=$(pwd)/assignment5_materials,target=/app,type=bind --net=host --env="DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix --volume="$HOME/.Xauthority:/home/developer/.Xauthority:rw" rl_env /bin/bash
