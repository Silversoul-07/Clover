.PHONY: start
start:
	sudo docker compose -f docker/compose.prod.yml up --build -d

.PHONY: stop
stop:
	sudo docker compose -f docker/compose.prod.yml down

.PHONY: start-dev
start-dev:
	sudo docker compose -f docker/compose.dev.yml up --build -d

.PHONY: stop-dev
stop-dev:
	sudo docker compose -f docker/compose.dev.yml down

.PHONY: frontend
frontend:
	cd frontend && npm run dev

.PHONY: backend
backend:
	cd backend && poetry run uvicorn app.main:app --reload

