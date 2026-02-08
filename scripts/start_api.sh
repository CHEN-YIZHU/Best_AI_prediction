#!/bin/bash

# echo "安装Python依赖..."
# pip install -r requirements_api.txt


echo "启动API服务器..."
python api/api_server.py --host 0.0.0.0 --port 8000 --reload


