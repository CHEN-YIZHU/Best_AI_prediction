# bash scripts/test.sh
cd /home/users/chenyizhu/AI_predict
source /home/users/chenyizhu/miniconda3/bin/activate verl_latest_251103
export AI_PROVIDER=qwen
export AI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=sk-a5003d95f24b49ebb40c1927f126fba1
export OPENAI_MODEL=qwen-max
python /home/users/chenyizhu/AI_predict/test_workflow.py
