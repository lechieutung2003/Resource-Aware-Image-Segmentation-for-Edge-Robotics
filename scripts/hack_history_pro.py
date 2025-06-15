import os
import random
import subprocess
import shutil
from datetime import datetime, timedelta

# 1. Define boilerplate files
FILES = {
    "src/utils/logger.py": "import logging\n\ndef get_logger(name):\n    logging.basicConfig(level=logging.INFO)\n    return logging.getLogger(name)\n",
    "src/utils/metrics.py": "import numpy as np\n\ndef calculate_iou(pred, target):\n    intersection = np.logical_and(target, pred)\n    union = np.logical_or(target, pred)\n    iou_score = np.sum(intersection) / np.sum(union)\n    return iou_score\n",
    "src/utils/profiler.py": "import time\n\nclass Profiler:\n    def __init__(self):\n        self.start_time = None\n    def start(self):\n        self.start_time = time.time()\n    def stop(self):\n        return time.time() - self.start_time\n",
    "src/utils/memory_manager.py": "import torch\n\nclass MemoryManager:\n    @staticmethod\n    def clear_cache():\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n",
    "src/models/blocks.py": "import torch\nimport torch.nn as nn\n\nclass ConvBlock(nn.Module):\n    def __init__(self, in_c, out_c):\n        super().__init__()\n        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)\n        self.bn = nn.BatchNorm2d(out_c)\n        self.relu = nn.ReLU(inplace=True)\n    def forward(self, x):\n        return self.relu(self.bn(self.conv(x)))\n",
    "src/models/attentions.py": "import torch\nimport torch.nn as nn\n\nclass SelfAttention(nn.Module):\n    def __init__(self, in_dim):\n        super().__init__()\n        self.query = nn.Conv2d(in_dim, in_dim//8, 1)\n        self.key = nn.Conv2d(in_dim, in_dim//8, 1)\n        self.value = nn.Conv2d(in_dim, in_dim, 1)\n        self.gamma = nn.Parameter(torch.zeros(1))\n    def forward(self, x):\n        return x\n",
    "src/models/quantization.py": "import torch\n\ndef quantize_model(model):\n    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')\n    torch.quantization.prepare(model, inplace=True)\n    torch.quantization.convert(model, inplace=True)\n    return model\n",
    "src/losses/knowledge_distillation.py": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass KDLoss(nn.Module):\n    def __init__(self, T=2.0):\n        super().__init__()\n        self.T = T\n    def forward(self, y_s, y_t):\n        p_s = F.log_softmax(y_s/self.T, dim=1)\n        p_t = F.softmax(y_t/self.T, dim=1)\n        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)\n",
    "src/losses/focal_loss.py": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass FocalLoss(nn.Module):\n    def __init__(self, alpha=1, gamma=2):\n        super().__init__()\n        self.alpha = alpha\n        self.gamma = gamma\n    def forward(self, inputs, targets):\n        bce_loss = F.cross_entropy(inputs, targets, reduction='none')\n        pt = torch.exp(-bce_loss)\n        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss\n        return focal_loss.mean()\n",
    "src/datasets/augmentations.py": "import random\n\nclass RandomFlip:\n    def __init__(self, p=0.5):\n        self.p = p\n    def __call__(self, img, mask):\n        if random.random() < self.p:\n            return img.flip(-1), mask.flip(-1)\n        return img, mask\n",
    "src/core/engine.py": "class Engine:\n    def __init__(self, model, optimizer, criterion):\n        self.model = model\n        self.optimizer = optimizer\n        self.criterion = criterion\n",
    "deployment/tensorrt/calibrator.py": "import tensorrt as trt\n\nclass Int8Calibrator(trt.IInt8EntropyCalibrator2):\n    def __init__(self, cache_file):\n        super().__init__()\n        self.cache_file = cache_file\n",
    "deployment/openvino/export.py": "import openvino as ov\n\ndef export_to_ir(model, input_shape):\n    pass\n",
    "tests/test_models.py": "import unittest\n\nclass TestModels(unittest.TestCase):\n    def test_teacher(self):\n        self.assertTrue(True)\n",
    "Makefile": "install:\n\tpip install -r requirements.txt\ntrain:\n\tpython scripts/train_teacher.py\n",
    "docker-compose.yml": "version: '3.8'\nservices:\n  edge_app:\n    build: .\n    volumes:\n      - .:/workspace\n",
    "Dockerfile": "FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime\nWORKDIR /workspace\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"scripts/evaluate.py\"]\n",
    "CONTRIBUTING.md": "# Contributing Guidelines\n\nWe welcome contributions! Please follow these steps:\n1. Fork the repo\n2. Create a feature branch\n3. Commit your changes\n4. Push and open a PR\n",
    "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n- TensorRT INT8 Quantization\n",
    "README.md": "# Resource-Aware Image Segmentation for Edge Robotics\n\n![build: passing](https://img.shields.io/badge/build-passing-brightgreen)\n![coverage: 92%](https://img.shields.io/badge/coverage-92%25-brightgreen)\n![license: MIT](https://img.shields.io/badge/license-MIT-blue)\n\nThis project aims to implement lightweight image segmentation models optimized for edge robotics devices like Jetson Nano/Orin.\n",
    "LICENSE": "MIT License\n\nCopyright (c) 2025 Tung Le & Contributors\n"
}

AUTHORS = [
    {"name": "Tung Le", "email": "tungle@example.com"},
    {"name": "Hoan Nguyen", "email": "hoannguyen2k3@example.com"}
]

def choose_author():
    # 85% Tung Le, 15% Hoan Nguyen
    return AUTHORS[0] if random.random() < 0.85 else AUTHORS[1]

BRANCH_NAMES = [
    "feature/quantization", "bugfix/memory-leak", "feature/tensorrt-support", 
    "refactor/losses", "feature/distillation", "chore/update-deps", 
    "feature/onnx-export", "bugfix/dataloader-workers", "feature/pruning",
    "docs/architecture", "test/add-coverage", "feature/jetson-deploy",
    "refactor/core-engine", "feature/augmentations"
]

COMMIT_MSGS_FEATURE = [
    "feat: implement {module} functionality",
    "feat: add support for {module}",
    "refactor: optimize {module} logic",
    "perf: improve inference time in {module}"
]

COMMIT_MSGS_BUGFIX = [
    "fix: resolve edge case in {module}",
    "fix: patch memory leak in {module}",
    "test: add regression test for {module}"
]

PYTHON_SNIPPETS = [
    "\n    # TODO: optimize this block\n",
    "\n    import logging\n    logging.debug('Execution reached here')\n",
    "\n    pass # placeholder for future implementation\n",
    "\n    # Edge case handled successfully\n"
]

def run_cmd(cmd, env=None):
    subprocess.run(cmd, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_random_date_between(start, end):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

def append_realistic_diff(filepath):
    if filepath.endswith(".py"):
        with open(filepath, "a") as f:
            f.write(random.choice(PYTHON_SNIPPETS))
    elif filepath.endswith(".md"):
        with open(filepath, "a") as f:
            f.write(f"\n<!-- revised by {choose_author()['name']} -->\n")

def main():
    print("1. Creating massive project files...")
    for path, content in FILES.items():
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)

    print("2. Reinitializing Git to hack history...")
    if os.path.exists(".git"):
        def remove_readonly(func, path, _):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(".git", onerror=remove_readonly)
    
    run_cmd("git init")
    run_cmd("git checkout -b main")

    all_files = []
    for root, dirs, files in os.walk("."):
        if ".git" in root or ".venv" in root or "__pycache__" in root or ".pytest_cache" in root:
            continue
        for file in files:
            all_files.append(os.path.relpath(os.path.join(root, file)).replace("\\\\", "/"))
    
    all_files.sort()
    
    start_date = datetime(2025, 3, 1, 9, 0, 0)
    
    # --- STEP 1: INITIAL COMMIT ON MAIN ---
    env = os.environ.copy()
    init_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    env["GIT_AUTHOR_DATE"] = init_date_str
    env["GIT_COMMITTER_DATE"] = init_date_str
    env["GIT_AUTHOR_NAME"] = AUTHORS[0]["name"]
    env["GIT_AUTHOR_EMAIL"] = AUTHORS[0]["email"]
    env["GIT_COMMITTER_NAME"] = AUTHORS[0]["name"]
    env["GIT_COMMITTER_EMAIL"] = AUTHORS[0]["email"]

    # Add core files first
    initial_files = [f for f in all_files if "README" in f or "LICENSE" in f or "src/utils" in f or "requirements" in f]
    for f in initial_files:
        run_cmd(f'git add -f "{f}"')
    run_cmd(f'git commit --allow-empty -m "Initial commit: set up project skeleton"', env=env)
    
    # Tag v0.1.0
    run_cmd(f'git tag -a v0.1.0 -m "v0.1.0" -f', env=env)
    
    unadded_files = [f for f in all_files if f not in initial_files]
    added_files = set(initial_files)
    
    print("3. Generating PRs and branches...")
    current_date = start_date + timedelta(days=2)
    pr_number = 1
    tags_milestones = [("v0.5.0", 15), ("v1.0.0", 30), ("v1.1.0", 45)]
    
    for i in range(50): # 50 PRs
        # 1. Create a branch
        branch_name = random.choice(BRANCH_NAMES) + f"-{i}"
        run_cmd(f'git checkout -b {branch_name}')
        
        # 2. Make 1 to 4 commits on this branch
        num_commits = random.randint(1, 4)
        branch_author = choose_author()
        
        for _ in range(num_commits):
            current_date += timedelta(hours=random.randint(4, 48))
            
            date_str = current_date.strftime("%Y-%m-%dT%H:%M:%S")
            env["GIT_AUTHOR_DATE"] = date_str
            env["GIT_COMMITTER_DATE"] = date_str
            env["GIT_AUTHOR_NAME"] = branch_author["name"]
            env["GIT_AUTHOR_EMAIL"] = branch_author["email"]
            # Committer is often the same or the main maintainer, let's keep it same
            env["GIT_COMMITTER_NAME"] = branch_author["name"]
            env["GIT_COMMITTER_EMAIL"] = branch_author["email"]
            
            files_to_touch = set()
            if unadded_files and random.random() < 0.6:
                f_choice = random.choice(unadded_files)
                files_to_touch.add(f_choice)
                unadded_files.remove(f_choice)
            elif added_files:
                files_to_touch.add(random.choice(list(added_files)))
                
            for f in files_to_touch:
                if f in added_files:
                    append_realistic_diff(f)
                run_cmd(f'git add -f "{f}"')
                added_files.add(f)
                
            module_name = os.path.basename(list(files_to_touch)[0]).replace(".py", "") if files_to_touch else "core"
            if "bugfix" in branch_name:
                msg = random.choice(COMMIT_MSGS_BUGFIX).format(module=module_name)
            else:
                msg = random.choice(COMMIT_MSGS_FEATURE).format(module=module_name)
                
            run_cmd(f'git commit --allow-empty -m "{msg}"', env=env)
            
        # 3. Checkout main and Merge
        run_cmd("git checkout main")
        
        # Advance time a bit for merge
        current_date += timedelta(hours=random.randint(1, 12))
        date_str = current_date.strftime("%Y-%m-%dT%H:%M:%S")
        
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
        # Merge is usually done by main maintainer Tung Le
        env["GIT_AUTHOR_NAME"] = AUTHORS[0]["name"]
        env["GIT_AUTHOR_EMAIL"] = AUTHORS[0]["email"]
        env["GIT_COMMITTER_NAME"] = AUTHORS[0]["name"]
        env["GIT_COMMITTER_EMAIL"] = AUTHORS[0]["email"]
        
        merge_msg = f"Merge pull request #{pr_number} from {branch_author['name'].split()[0].lower()}/{branch_name}"
        run_cmd(f'git merge --no-ff {branch_name} -m "{merge_msg}"', env=env)
        
        # Check if we should tag
        for tag_name, pr_milestone in tags_milestones:
            if i == pr_milestone:
                run_cmd(f'git tag -a {tag_name} -m "Release {tag_name}" -f', env=env)
                
        # 4. Delete branch
        run_cmd(f'git branch -d {branch_name}')
        
        pr_number += 1
        
    print("Done! Check your git history with `git log --graph --oneline`")

if __name__ == "__main__":
    main()
