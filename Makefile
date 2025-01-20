.PHONY: frontend
frontend:
	cd frontend && npm run dev

.PHONY: docker-start
docker-start:
	sudo docker compose up --build -d

.PHONY: docker-stop
docker-stop:
	sudo docker compose down

.PHONY: backend
backend:
	cd backend && poetry run uvicorn app.main:app --reload

.PHONY: docker-dev-start
docker-dev-start:
	sudo docker compose -f docker-compose.dev.yml up --build -d

.PHONY: docker-dev-stop
docker-dev-stop:
	sudo docker compose -f docker-compose.dev.yml down
