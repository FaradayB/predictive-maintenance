#!/bin/bash
cd ~/predictive-maintenance

sleep 10

nohup python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8010 > logs/api.log 2>&1 &

nohup python3 -m streamlit run app_technician.py \
  --server.port 8501 \
  --server.address 0.0.0.0 > logs/technician.log 2>&1 &

nohup python3 -m streamlit run app_owner.py \
  --server.port 8502 \
  --server.address 0.0.0.0 > logs/owner.log 2>&1 &

echo "All services started."
