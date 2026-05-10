docker build -t nba556677/inference:flexgen --progress=plain .
cd vllm-server && docker build -t nba556677/chatbot:vllm --progress=plain .