import subprocess
import uuid
import shutil
import os
import threading
import queue
import time

TEST_DIR = 'testset'

class InferenceJobManager:
    def __init__(self, max_containers=48):
        self.active_containers = 0
        self.max_containers = max_containers
        self.lock = threading.Lock()
        self.job_queue = queue.Queue()
        self.results = {}
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def can_run(self, user_id):
        with self.lock:
            if self.active_containers >= self.max_containers:
                return False, "Maximum number of containers reached. Your job will be queued and processed soon."
            return True, None

    def add_job(self, user_id, job_func, *args, **kwargs):
        job_id = str(uuid.uuid4())
        self.job_queue.put((job_id, user_id, job_func, args, kwargs))
        return job_id

    def _worker(self):
        while True:
            job_id, user_id, job_func, args, kwargs = self.job_queue.get()
            with self.lock:
                if self.active_containers >= self.max_containers:
                    # Wait until a slot is free
                    while self.active_containers >= self.max_containers:
                        time.sleep(1)
                self.active_containers += 1
            try:
                result = job_func(*args, **kwargs)
                self.results[job_id] = result
            except Exception as e:
                self.results[job_id] = {"error": str(e)}
            finally:
                with self.lock:
                    self.active_containers -= 1
                self.job_queue.task_done()

    def get_result(self, job_id):
        return self.results.pop(job_id, None)

job_manager = InferenceJobManager()

def prepare_test_set():
    image_paths = sorted([
        os.path.join(root, fname)
        for root, _, files in os.walk(TEST_DIR)
        for fname in sorted(files) if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    return image_paths

def prepare_image_ids(image_paths):
    filenames = []
    for path in image_paths:
        filenames.append(os.path.basename(path))
    return filenames

def _run_inference_job(USER_ID):
    print("Script started!", flush=True)
    job_id = str(uuid.uuid4())
    base_tmp = os.path.abspath("./tmp")
    os.makedirs(base_tmp, exist_ok=True)
    temp_dir = os.path.join(base_tmp, f"job-{job_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    user_py = f"inference/{USER_ID}.py"  
    model_pth = f"models/{USER_ID}.pth"  

    shutil.copy(user_py, os.path.join(temp_dir, "inference.py"))
    shutil.copy(model_pth, os.path.join(temp_dir, "model.pth"))
    shutil.copy("run_inference_job.py", os.path.join(temp_dir, "run_inference_job.py"))
    shutil.copy("utils.py", os.path.join(temp_dir, "utils.py"))
    
    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--shm-size=8g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--network", "none",
        "--cpus=10", "--memory=16g",
        "-v", f"{temp_dir}:/workspace:rw",
        "-v", f"{os.path.abspath(TEST_DIR)}:/workspace/testset:ro",
        "inference-sandbox",
        "python", "-u", "/workspace/run_inference_job.py"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        print("STDOUT:", result.stdout.decode('utf-8'))
        print("STDERR:", result.stderr.decode('utf-8'))

        with open(os.path.join(temp_dir, "logs.txt"), "wb") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        
        return {
            "stdout": result.stdout.decode('utf-8'),
            "stderr": result.stderr.decode('utf-8'),
            "temp_dir": temp_dir,
            "predictions_file": os.path.join(temp_dir, "predictions.csv")
        }
    except subprocess.TimeoutExpired:
        return {"error": "Inference job timed out, make sure your code does not exceed the time limit."}

def run_inference_in_docker(USER_ID):
    # Always queue the job for load balancing
    job_id = job_manager.add_job(USER_ID, _run_inference_job, USER_ID)
    # Wait for result (polling, could be improved with async/notify)
    while True:
        result = job_manager.get_result(job_id)
        if result is not None:
            return result
        time.sleep(1)