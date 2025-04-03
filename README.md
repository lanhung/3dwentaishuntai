1. 执行命令
```
# docker-compose.yaml中的执行命令行被注释掉了，是因为安装ffmpeg会卡住，要等很久

docker compose -f docker-compose.yaml up
```
2. 容器启动之后进入jupyter notebook，执行command.ipynb中的命令
3. 也可以进入到容器里面执行command.ipynb中的命令
```
pip install -r /tf/notebooks/requirements.txt && apt-get update && apt-get install -y ffmpeg
```