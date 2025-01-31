start:
    sudo docker compose -f docker/compose.prod.yml up --build -d

stop:
    sudo docker compose -f docker/compose.prod.yml down

start-dev:
    sudo docker compose -f docker/compose.dev.yml up --build -d

stop-dev:
    sudo docker compose -f docker/compose.dev.yml down

frontend:
    cd frontend && npm run dev

backend:
    cd backend && poetry run uvicorn app.main:app --reload