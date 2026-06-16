import mlflow
import requests
import time

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llmops-ollama-tracking")

def query_ollama(model, prompt):
    with mlflow.start_run():
        mlflow.log_param("model", model)
        mlflow.log_param("prompt", prompt)

        start = time.time()
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False})
        duration = time.time() - start

        result = response.json()
        mlflow.log_metric("response_time_sec", round(duration, 3))
        mlflow.log_metric("response_length", len(result.get("response", "")))
        mlflow.set_tag("status", "success")
        print(f"Model: {model} | Time: {round(duration,2)}s")
        return result.get("response", "")

if __name__ == "__main__":
    query_ollama("tinyllama", "What is cloud computing in one sentence?")
    query_ollama("llama3.2:1b", "What is MLOps in one sentence?")
