import mlflow
import requests
import time

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llmops-ab-testing")

MODELS = ["tinyllama", "llama3.2:1b"]
PROMPTS = [
    "What is Kubernetes in one sentence?",
    "What is MLOps in one sentence?",
    "What is Docker in one sentence?"
]

def run_ab_test(model, prompt):
    with mlflow.start_run(run_name=f"{model}-test"):
        mlflow.log_param("model", model)
        mlflow.log_param("prompt", prompt)

        start = time.time()
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120)
        duration = round(time.time() - start, 3)

        result = response.json()
        answer = result.get("response", "")

        mlflow.log_metric("response_time_sec", duration)
        mlflow.log_metric("response_length", len(answer))
        mlflow.set_tag("ab_group", model)
        mlflow.set_tag("test_type", "comparative")

        print(f"[{model}] {duration}s → {answer[:80]}...")
        return duration

if __name__ == "__main__":
    print("=== A/B Test: tinyllama vs llama3.2:1b ===\n")
    for prompt in PROMPTS:
        print(f"Prompt: {prompt}")
        for model in MODELS:
            run_ab_test(model, prompt)
        print()
    print("✅ Done! View results at http://localhost:5000")
