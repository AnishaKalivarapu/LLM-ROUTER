# Smart LLM Router (backend-only)

A FastAPI service that classifies each query's complexity and routes it to the cheapest model that
should handle it well — across **OpenAI and Gemini** — escalating to stronger models if a call fails.

```
                    ┌───────────────────────────────┐
 POST /v1/route ──► │ classifier: heuristic | llm | │
                    │ hybrid (heuristic → LLM)      │
                    └──────────────┬────────────────┘
                       simple / moderate / complex  (+ preference: cost|latency|quality)
                                   ▼
                  escalation chain, e.g. complex →  gemini-2.5-pro → gpt-4.1
                                   ▼
        Executor: retry transient errors → circuit breaker → escalate to next model
                                   ▼
              answer + model + cost + latency + full attempt trace
```

## Design

| Concern | How it works |
|---|---|
| **Heuristic classifier** (`classifiers/heuristic.py`) | Scores length, greetings/lookups, strong/medium keywords, code, math, multi-part asks. Score → tier; distance from the nearest threshold → confidence. No API call. |
| **Hybrid classifier** (`classifiers/hybrid.py`) | Heuristic first. If confidence < `CONFIDENCE_THRESHOLD`, a tier‑0 model labels the query. If that call fails, the heuristic result is kept — classification never fails a request. |
| **LLM classifier** (`classifiers/llm.py`) | Cheapest model only, JSON output, query wrapped in tags and treated as data (basic prompt-injection hygiene). Its token cost is included in the response. |
| **Multi-provider** (`providers/`) | Native REST adapters for OpenAI Chat Completions and Gemini `generateContent` (incl. Gemini "thinking" tokens in cost). Gemini key is sent as a header, not in the URL. |
| **Model ladder** (`config.py`) | Models have tiers 0/1/2 and prices. Only models whose provider key is set are used, so it runs with one provider or both. Override with `MODELS` JSON. |
| **Quality / latency / cost** | `preference`: `quality` bumps one tier; `cost` orders a tier by price; `latency` orders a tier by observed EWMA latency; default follows the classifier. |
| **Fallback** (`executor.py`) | Retryable errors (429/5xx/timeout) get retried with backoff on the same model; non-retryable errors (4xx, blocked, empty output) escalate immediately. Chains only go **up** in capability and can cross providers. Per-model circuit breaker skips repeatedly failing models for a cooldown. Every response includes the attempt trace. |
| **Metrics** (`metrics.py`) | Requests, p50/p95 latency, per-model cost/tokens, fallback rate, and estimated savings vs. always calling the strongest model. |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add OPENAI_API_KEY and/or GEMINI_API_KEY
uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/v1/route -H 'content-type: application/json' \
  -d '{"query":"Design a rate limiter and analyze the trade-offs","preference":"balanced"}' | jq
curl -s localhost:8000/v1/classify -H 'content-type: application/json' -d '{"query":"Hello!"}'
curl -s localhost:8000/v1/metrics
```

Endpoints: `POST /v1/route`, `POST /v1/classify`, `GET /v1/metrics`, `GET /v1/models`, `GET /health`. Interactive docs at `/docs`.

## Test & evaluate

```bash
python -m pytest                    # 54 tests, no network or keys needed (fake providers + httpx MockTransport)
python -m scripts.eval_classifier   # heuristic accuracy on data/eval_queries.json
```

The eval set is 45 hand-labelled queries — a sanity check, not a benchmark. Swap in real traffic
samples before quoting accuracy anywhere.

## Docker

```bash
docker build -t smart-llm-router . && docker run -p 8000:8000 --env-file .env smart-llm-router
```

## Notes / limitations

- Model names and prices in `config.py` are defaults — check current provider pricing and model availability.
- Metrics are in-memory and per-process (use Redis/Prometheus for multi-worker deployments).
- The complexity→tier mapping is a prior, not ground truth; the right next step is logging real traffic,
  labelling a sample, and tuning thresholds against it.
