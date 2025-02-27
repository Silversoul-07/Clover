start-dev:
    sudo docker compose -f docker/compose.dev.yml up --build -d

stop-dev:
    sudo docker compose -f docker/compose.prod.yml down

frontend:
    cd frontend && npm run dev

backend:
    cd backend && poetry run uvicorn app.main:app --reload

ml-server:
    cd ml-server && poetry run uvicorn app.main:app --reload --port 9876