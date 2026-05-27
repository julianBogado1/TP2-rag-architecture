#!/usr/bin/env bash
curl -X POST http://localhost:8000/recommend -H 'Content-Type: application/json' -d '{"user_id": "user_001", "raw_prompt": "quiero canciones que vayan con mi mood feliz :)"}'