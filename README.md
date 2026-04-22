# Face Recognition Service

Internal microservice for face detection, enrollment, verification (1:1), and recognition (1:N).

## Quick Start

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Start services (PostgreSQL + Face Service)
docker compose up --build -d

# 3. Check health
curl http://localhost:8001/healthz

# 4. Check readiness (DB + model loaded)
curl http://localhost:8001/readyz

# 5. View API docs
open http://localhost:8001/docs
```

## Architecture

```
┌──────────────┐     X-API-Key      ┌─────────────────┐
│   Laravel    │ ──────────────────▶ │  Face Service   │
│  (fts-absen) │                    │  FastAPI :8001   │
│   MySQL DB   │ ◀────── JSON ───── │                  │
└──────────────┘                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │  PostgreSQL :5433 │
                                    │  + pgvector       │
                                    └──────────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/faces/detect` | Detect faces in an image |
| `POST` | `/v1/faces/enroll` | Enroll face template for a subject |
| `POST` | `/v1/faces/verify` | 1:1 verification against stored templates |
| `POST` | `/v1/faces/recognize` | 1:N recognition (top-k search) |
| `GET` | `/v1/faces/subjects/{id}` | Check enrollment status |
| `DELETE` | `/v1/faces/subjects/{id}` | Delete subject and all templates |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/readyz` | Readiness probe |

## Authentication

All endpoints (except health checks) require the `X-API-Key` header:

```bash
curl -X POST http://localhost:8001/v1/faces/detect \
  -H "X-API-Key: your-secret-key" \
  -F "image=@photo.jpg"
```

## Environment Variables

See [`.env.example`](.env.example) for all configurable values.

## Project Structure

```
face-service/
├── app/
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic request/response models
│   ├── core/
│   │   ├── config.py          # Pydantic settings (env vars)
│   │   ├── database.py        # PostgreSQL + pgvector setup
│   │   ├── logging.py         # Structured logging + request ID
│   │   └── security.py        # X-API-Key verification
│   ├── models/
│   │   └── database_models.py # SQLAlchemy ORM (subjects, embeddings, logs)
│   ├── services/
│   │   └── face_pipeline.py   # MTCNN + InceptionResnetV1 pipeline
│   └── main.py                # FastAPI app factory + lifespan
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```
