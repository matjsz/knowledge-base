---
title: "A new GPU training ecosystem"
date: "2026-02-01"
type: "engineering"
tags: ["GPU", "PyTorch", "Prefect", "DVC", "MLflow", "Docker"]
summary: "Creating an on-premise GPU training ecosystem from scratch."
reading_time: "15 min"
citations: false
math: true
---

## Chaos

Let's start by stating that before all of this was done, things were... chaotic.

## The Context

At the moment I'm writing this article, I'm working as an MLE at ASTRA, here I train ML models, gather data, prepare pipelines, validate metrics and all the cool stuff. Up until the moment we decided to finally build our GPU training server, I was training heavy-load neural networks on my computer (ugh) and even though it worked, I hated the lack of monitoring, tracking and versioning the models had, mostly because there wasn't a dedicated enviroment for all of that.

That's when we decided it was enough. We grabbed a GPU, installed on a machine, my friend [@Gustavo](https://www.linkedin.com/in/gustavo-henrique-rodrigues-3070a5260/) installed the OS, prepared the Docker NVIDIA CUDA Toolkit for containerization with CUDA and... it works!

## The Machine

We didn't just want a fast computer; we wanted a reproducible environment. The hardware is useless if the drivers fight the container runtime every time we try to run a job.

The foundation is a dedicated server running Linux, but the critical piece of software glue is the **NVIDIA Container Toolkit**. This allows Docker containers to interact directly with the GPU. Instead of installing CUDA 11.8 or 12.1 globally and praying it doesn't break `apt-get`, we simply pass the `--gpus all` flag (or `runtime="nvidia"`) to Docker.

This allows us to treat the GPU as a fungible resource. If a project needs PyTorch 2.1 and another needs 1.13, they don't clash. They just run.

## The Architecture

To move away from the chaos, we designed a pipeline where **local** development is strictly separated from **remote** execution. The code lives in git, the data lives in MinIO (S3 compatible object storage), and the execution happens on the GPU server.

Here is the high-level architecture of the ecosystem we built:

```mermaid
graph TD
    subgraph "Local Environment"
        Dev[MLE (You)]
        Git[Git Repo]
    end

    subgraph "Infrastructure Services"
        MinIO[MinIO Object Storage]
        MLflow[MLflow Tracking Server]
        Prefect[Prefect Orchestrator]
    end

    subgraph "GPU Server"
        Agent[Prefect Worker]
        Train[Training Container]
        API[Serving API Container]
    end

    Dev -->|Push Code| MinIO
    Dev -->|Trigger| Prefect
    Prefect -->|Command| Agent
    Agent -->|Spin up| Train
    
    Train -->|Pull Data| MinIO
    Train -->|Log Metrics| MLflow
    Train -->|Push Weights| DVC
    DVC -.->|Store| MinIO
    
    Train -->|Deploy| API
    API -->|Load Model| MLflow

```

## The Pipeline

We chose **Prefect** to orchestrate the madness. It allows us to define our workflow as Python code, handle retries, and manage logs centrally.

Our `remote_training_pipeline.py` is the workhorse. While it does the training bit, it also acts as the entire lifecycle manager for the model.

### 1. Data Ingestion

We treat our MinIO bucket as the single source of truth. That way, we stream it fresh for every run. This ensures that if we update the dataset, the next training run picks it up automatically.

```python
@task(name="Pull Data")
def pull_and_prep_data(base_dir="/opt/prefect/training_data"):
    s3 = boto3.client('s3', ...)

    # We enforce a clean state every time
    if base_path.exists():
        shutil.rmtree(base_path)

    # Download logic handling train/val splits
    # ...
    return str(base_path)

```

The pipeline pulls data from our source bucket (`*-train`), splits it into training and validation sets, and structures it locally for YOLO.

### 2. Training (The Heavy Lifting)

For this specific project, we are using **Ultralytics YOLO** for image classification. The beauty of wrapping this in a Prefect task is that we can dynamically assign the device and capture outputs.

We also make sure the GPU is actually there before we start burning electricity:

```python
@task(name="GPU Check", retries=0)
def check_gpu_status():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: No GPU detected! Aborting.")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Detected: {gpu_name}")
    return 0 
```

The training task logs directly to MLflow, giving us those sweet loss curves in real-time!

### 3. The "Double Lock" Versioning Strategy

This is where I think our approach shines. We use **MLflow** for metrics and model registry, but we use **DVC (Data Version Control)** for the physical weight storage.

Why both? MLflow is great for _"Which run had the best accuracy?"_ DVC is great for _"I need the exact binary file associated with this commit hash."_

In our `push_artifacts` task, we do a handshake between the two:

1. We initialize DVC and push the `.pt` file to MinIO.
2. We grab the MD5 hash generated by DVC.
3. We log that hash into MLflow as a tag.

```python
# 1. Push to Remote Storage
subprocess.run(["dvc", "push", "artifacts/best.pt"], check=True)

# 2. Get the Hash
with open(dvc_file, 'r') as f:
    content = yaml.safe_load(f)
    file_hash = content['outs'][0]['md5']

# 3. Link it in MLflow
mlflow.set_tag("dvc_hash", file_hash)
```

This creates an unbreakable chain of custody. If the model behaves weirdly in production, I can look at the API, see the MLflow Run ID, look up the DVC hash, and trace it back to the exact training data used. Tracking is key in every ML project!

## Continuous Deployment (CD)

The final step of the pipeline isn't just "saving the model." It's **putting it to work.**

We implemented a "Hot Swap" deployment directly in the training pipeline. If the training succeeds, the pipeline instructs the Docker daemon on the server to:

1. Pull the latest API image.
2. Stop the old API container.
3. Start a new container injected with the _new_ `MLFLOW_RUN_ID`.
4. **Wait for a Health Check.**

That last part is crucial, we don't just fire and forget.

```python
@task(name="Deploy API")
def deploy_api_container(run_id: str):
    # docker run logic...
    
    print("Waiting for API health check...")
    health_url = f"http://{CONTAINER_NAME}:8000/health" # 8000 because we're talking directly to the container network, not the host network!
    
    for i in range(20):
        # Loop to check if the new model is actually serving requests
        response = requests.get(health_url, timeout=2)
        if response.status_code == 200:
            print(f"Health Check Passed!")
            return CONTAINER_NAME
            
    # If it fails, we kill it and raise an error
    container.stop()
    raise RuntimeError("Health check timed out.")

```

This ensures that we never replace a working production model with a broken one.

## Deployment: "The Code IS the Infrastructure"

One of the coolest things about Prefect is how we deploy the flow itself. We don't SSH into the server to `git pull`.

We have a `deploy_pipeline.py` script that packages our code and uploads it to MinIO as a storage block. The GPU worker simply polls for new work, and when it finds a job, it downloads the code from MinIO and executes it.

```python
minio_block = RemoteFileSystem(
    basepath=f"s3://{BUCKET_NAME}/{PREFIX}",
    # credentials ...
)

remote_training_pipeline.from_source(
    source=minio_block,
    entrypoint=f"{FLOW_FILENAME}:remote_training_pipeline"
).deploy(
    name="gatekeeper-production-gpu-training",
    work_pool_name="local-pool"
)
```

This means I can trigger a training run on the heavy GPU server from a CLI command on my lightweight laptop, without ever opening an SSH tunnel.

## Peace

Moving from local scripts to this ecosystem transformed our workflow:

1. **Observability:** I no longer guess if a model is training; I see the logs in Prefect and the metrics in MLflow.
2. **Reproducibility:** Every model in production can be traced back to the specific line of code and dataset byte that created it.
3. **Speed:** We utilize the dedicated GPU 24/7 without locking up our dev machines.

It was a bit of work to set up the Docker with CUDA support, MinIO, Prefect and DVC all glued together, but now that it flows, it feels less like chaos and more like engineering. I'm just happy that Prefect, Docker and CUDA exists. Doing miracles with that trinity.

Now I need to check out my model metrics. Don't forget to drink water!
