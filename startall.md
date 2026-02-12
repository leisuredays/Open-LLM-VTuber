DIFY
cd /home/leisuredays/dify/docker  
docker compose up -d

MCP router (docker 사용)
cd /home/leisuredays/Open-LLM-VTuber/mcp-servers                    
python router_mcp_server.py --host 0.0.0.0 --port 8769

Mindcraft
cd /home/leisuredays/mindcraft
nvm use 20
npm start

LLMVtuber
cd /home/leisuredays/Open-LLM-VTuber
conda activate open-llm-vtuber
python run_server.py --verbose

GptSovits
cd /home/leisuredays/GPT-SoVITS
conda activate GPTSoVits
python api_v2.py

python scripts/run_discord_live.py 

frontend
cd /home/leisuredays/Open-LLM-VTuber/Open-LLM-VTuber-Web
npm run dev:web