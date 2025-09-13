# Flight Analytics Platform Makefile

.PHONY: help setup start stop logs clean test

help:
	@echo "Available commands:"
	@echo "  setup    - Initial setup (copy .env, generate keys)"
	@echo "  start    - Start all services"
	@echo "  stop     - Stop all services"
	@echo "  logs     - View logs"
	@echo "  clean    - Clean up containers and volumes"
	@echo "  test     - Run dbt tests"

setup:
	@echo "Setting up Flight Analytics Platform..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please update with your API keys"; \
	fi
	@echo "AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here-replace-with-32-char-base64-key" >> .env
	@echo "Generated placeholder Fernet key - replace with actual key for production"

start:
	@echo "Starting services..."
	docker compose up -d
	@echo "Services started. Airflow UI: http://localhost:8080"
	@echo "Default login: admin/admin"

stop:
	@echo "Stopping services..."
	docker compose down

logs:
	docker compose logs -f

clean:
	@echo "Cleaning up..."
	docker compose down -v
	docker system prune -f

test:
	@echo "Running dbt tests..."
	docker compose exec airflow-webserver bash -c "cd /opt/dbt && dbt test --profiles-dir ."

init-db:
	@echo "Initializing Airflow database..."
	docker compose exec airflow-webserver airflow db init
	docker compose exec airflow-webserver airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com \
		--password admin