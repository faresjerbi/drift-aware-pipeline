
from prometheus_client import Counter, Gauge, Histogram, start_http_server


# Total detected events, grouped by signal.
DRIFT_EVENTS_TOTAL = Counter(
    "drift_events_total",
    "Total number of detected drift events",
    ["signal"],
)

# Number of model evaluations and their outcomes.
ACCURACY_EVALUATIONS_TOTAL = Counter(
    "accuracy_evaluations_total",
    "Total number of model accuracy evaluations",
    ["result"],
)

# Current evaluation score for each model.
MODEL_ACCURACY_SCORE = Gauge(
    "model_accuracy_score",
    "Most recent model evaluation score (0-100)",
    ["model"],
)

# Most recently calculated embedding drift score.
LAST_DRIFT_SCORE = Gauge(
    "last_drift_score",
    "Most recently calculated embedding drift score",
)

# Pipeline decisions, grouped by status.
PIPELINE_DECISIONS_TOTAL = Counter(
    "pipeline_decisions_total",
    "Total pipeline decisions",
    ["status"],
)
# Initialize known decision statuses at zero.
for status in (
    "skipped",
    "detected",
    "processing",
    "accepted",
    "rejected",
    "failed",
):
    PIPELINE_DECISIONS_TOTAL.labels(status=status)
# Time spent evaluating a model.
MODEL_EVALUATION_DURATION = Histogram(
    "model_evaluation_duration_seconds",
    "Duration of model evaluations",
)

# Number of currently active behavior alerts.
ACTIVE_BEHAVIOR_ALERTS = Gauge(
    "active_behavior_alerts",
    "Number of currently active behavior alerts",
)


def record_drift_event(signal, score):
    DRIFT_EVENTS_TOTAL.labels(signal=signal).inc()
    LAST_DRIFT_SCORE.set(score)


def record_accuracy_evaluation(model, score, accuracy_drop_detected):
    result = "accuracy_drop" if accuracy_drop_detected else "normal"

    ACCURACY_EVALUATIONS_TOTAL.labels(result=result).inc()
    MODEL_ACCURACY_SCORE.labels(model=model).set(score)


def record_pipeline_decision(status):
    PIPELINE_DECISIONS_TOTAL.labels(status=status).inc()


def set_active_behavior_alerts(count):
    ACTIVE_BEHAVIOR_ALERTS.set(count)


def start_metrics_server(port=8000):
    start_http_server(port)
    print(f"Prometheus metrics available at http://localhost:{port}/metrics")
