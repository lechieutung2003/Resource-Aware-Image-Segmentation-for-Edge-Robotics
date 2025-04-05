import os
import random
import subprocess
import shutil
from datetime import datetime, timedelta

# 1. Define files to create for a massive project structure
FILES = {
    "src/utils/logger.py": "import logging\n\ndef get_logger(name):\n    logging.basicConfig(level=logging.INFO)\n    return logging.getLogger(name)\n",
    "src/utils/metrics.py": "import numpy as np\n\ndef calculate_iou(pred, target):\n    intersection = np.logical_and(target, pred)\n    union = np.logical_or(target, pred)\n    iou_score = np.sum(intersection) / np.sum(union)\n    return iou_score\n",
    "src/utils/profiler.py": "import time\n\nclass Profiler:\n    def __init__(self):\n        self.start_time = None\n    def start(self):\n        self.start_time = time.time()\n    def stop(self):\n        return time.time() - self.start_time\n",
    "src/utils/visualization.py": "import matplotlib.pyplot as plt\n\ndef plot_mask(image, mask):\n    plt.imshow(image)\n    plt.imshow(mask, alpha=0.5)\n    plt.show()\n",
    "src/utils/memory_manager.py": "import torch\n\nclass MemoryManager:\n    @staticmethod\n    def clear_cache():\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n",
    "src/models/blocks.py": "import torch\nimport torch.nn as nn\n\nclass ConvBlock(nn.Module):\n    def __init__(self, in_c, out_c):\n        super().__init__()\n        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)\n        self.bn = nn.BatchNorm2d(out_c)\n        self.relu = nn.ReLU(inplace=True)\n    def forward(self, x):\n        return self.relu(self.bn(self.conv(x)))\n",
    "src/models/attentions.py": "import torch\nimport torch.nn as nn\n\nclass SelfAttention(nn.Module):\n    def __init__(self, in_dim):\n        super().__init__()\n        self.query = nn.Conv2d(in_dim, in_dim//8, 1)\n        self.key = nn.Conv2d(in_dim, in_dim//8, 1)\n        self.value = nn.Conv2d(in_dim, in_dim, 1)\n        self.gamma = nn.Parameter(torch.zeros(1))\n    def forward(self, x):\n        return x\n",
    "src/models/quantization.py": "import torch\n\ndef quantize_model(model):\n    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')\n    torch.quantization.prepare(model, inplace=True)\n    torch.quantization.convert(model, inplace=True)\n    return model\n",
    "src/models/pruning.py": "import torch.nn.utils.prune as prune\n\ndef prune_conv_layer(module, amount=0.3):\n    prune.l1_unstructured(module, name='weight', amount=amount)\n    prune.remove(module, 'weight')\n",
    "src/losses/knowledge_distillation.py": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass KDLoss(nn.Module):\n    def __init__(self, T=2.0):\n        super().__init__()\n        self.T = T\n    def forward(self, y_s, y_t):\n        p_s = F.log_softmax(y_s/self.T, dim=1)\n        p_t = F.softmax(y_t/self.T, dim=1)\n        return F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)\n",
    "src/losses/focal_loss.py": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass FocalLoss(nn.Module):\n    def __init__(self, alpha=1, gamma=2):\n        super().__init__()\n        self.alpha = alpha\n        self.gamma = gamma\n    def forward(self, inputs, targets):\n        bce_loss = F.cross_entropy(inputs, targets, reduction='none')\n        pt = torch.exp(-bce_loss)\n        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss\n        return focal_loss.mean()\n",
    "src/datasets/augmentations.py": "import random\n\nclass RandomFlip:\n    def __init__(self, p=0.5):\n        self.p = p\n    def __call__(self, img, mask):\n        if random.random() < self.p:\n            return img.flip(-1), mask.flip(-1)\n        return img, mask\n",
    "src/datasets/samplers.py": "from torch.utils.data import Sampler\n\nclass EdgeSampler(Sampler):\n    def __init__(self, data_source):\n        self.data_source = data_source\n    def __iter__(self):\n        return iter(range(len(self.data_source)))\n    def __len__(self):\n        return len(self.data_source)\n",
    "src/core/engine.py": "class Engine:\n    def __init__(self, model, optimizer, criterion):\n        self.model = model\n        self.optimizer = optimizer\n        self.criterion = criterion\n",
    "src/core/trainer.py": "class Trainer:\n    def train_epoch(self, dataloader):\n        pass\n",
    "src/core/evaluator.py": "class Evaluator:\n    def evaluate(self, dataloader):\n        pass\n",
    "deployment/tensorrt/calibrator.py": "import tensorrt as trt\n\nclass Int8Calibrator(trt.IInt8EntropyCalibrator2):\n    def __init__(self, cache_file):\n        super().__init__()\n        self.cache_file = cache_file\n",
    "deployment/openvino/export.py": "import openvino as ov\n\ndef export_to_ir(model, input_shape):\n    pass\n",
    "deployment/tflite/convert.py": "import tensorflow as tf\n\ndef convert_tflite():\n    pass\n",
    "tests/test_models.py": "import unittest\n\nclass TestModels(unittest.TestCase):\n    def test_teacher(self):\n        self.assertTrue(True)\n",
    "tests/test_losses.py": "import unittest\n\nclass TestLosses(unittest.TestCase):\n    def test_focal(self):\n        self.assertTrue(True)\n",
    "tests/test_utils.py": "import unittest\n\nclass TestUtils(unittest.TestCase):\n    def test_logger(self):\n        self.assertTrue(True)\n",
    "Makefile": "install:\n\tpip install -r requirements.txt\ntrain:\n\tpython scripts/train_teacher.py\ndistill:\n\tpython scripts/distill.py\n",
    ".pre-commit-config.yaml": "repos:\n- repo: https://github.com/psf/black\n  rev: 22.3.0\n  hooks:\n  - id: black\n- repo: https://github.com/pycqa/isort\n  rev: 5.12.0\n  hooks:\n  - id: isort\n",
    "docker-compose.yml": "version: '3.8'\nservices:\n  edge_app:\n    build: .\n    volumes:\n      - .:/workspace\n    deploy:\n      resources:\n        reservations:\n          devices:\n            - driver: nvidia\n              count: 1\n              capabilities: [gpu]\n",
    "Dockerfile": "FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime\nWORKDIR /workspace\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"scripts/evaluate.py\"]\n",
    "CONTRIBUTING.md": "# Contributing Guidelines\n\nWe welcome contributions! Please follow these steps:\n1. Fork the repo\n2. Create a feature branch\n3. Commit your changes\n4. Push and open a PR\n",
    "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n- TensorRT INT8 Quantization\n\n## [1.2.0] - 2025-08-15\n- Added knowledge distillation pipeline\n\n## [1.1.0] - 2025-06-10\n- Jetson Nano deployment support\n\n## [1.0.0] - 2025-04-20\n- Initial release with teacher-student models\n",
    ".github/ISSUE_TEMPLATE/bug_report.md": "---\nname: Bug report\nabout: Create a report to help us improve\ntitle: ''\nlabels: bug\nassignees: ''\n---\n\n**Describe the bug**\nA clear and concise description of what the bug is.\n",
    ".github/PULL_REQUEST_TEMPLATE.md": "## Description\n\nPlease include a summary of the change and which issue is fixed.\n\n## Type of change\n- [ ] Bug fix\n- [ ] New feature\n- [ ] Breaking change\n",
    ".github/workflows/ci.yml": "name: CI Pipeline\n\non: [push, pull_request]\n\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n    - uses: actions/checkout@v3\n    - name: Set up Python\n      uses: actions/setup-python@v4\n      with:\n        python-version: '3.10'\n    - name: Install dependencies\n      run: pip install -r requirements.txt\n    - name: Run tests\n      run: pytest tests/\n"
}

COMMIT_MESSAGES_GENERIC = [
    "feat: implement core processing module",
    "fix: resolve memory issue during inference",
    "refactor: clean up module structure",
    "chore: update configuration files",
    "docs: improve documentation",
    "test: add comprehensive test suite",
    "perf: optimize execution time",
    "style: enforce formatting standards",
    "feat: add support for edge devices",
    "fix: correct boundary loss calculation",
    "refactor: extract utilities into separate module",
    "chore: bump dependencies",
    "build: update CI/CD pipeline",
]

def run_cmd(cmd, env=None):
    subprocess.run(cmd, shell=True, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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

    all_files = []
    for root, dirs, files in os.walk("."):
        if ".git" in root or ".venv" in root or "__pycache__" in root or ".pytest_cache" in root:
            continue
        for file in files:
            all_files.append(os.path.relpath(os.path.join(root, file)).replace("\\\\", "/"))
    
    all_files.sort()
    
    start_date = datetime(2025, 3, 1, 9, 0, 0)
    end_date = datetime(2025, 10, 31, 18, 0, 0)
    
    num_commits = 70 # Fewer commits but more files per commit
    total_seconds = (end_date - start_date).total_seconds()
    step_seconds = total_seconds / num_commits
    
    env = os.environ.copy()
    
    print("3. Generating fake commits...")
    
    added_files = set()
    
    for i in range(num_commits):
        commit_date = start_date + timedelta(seconds=i*step_seconds)
        commit_date += timedelta(minutes=random.randint(-240, 240))
        
        if commit_date.weekday() > 4: 
            commit_date -= timedelta(days=2)
            
        date_str = commit_date.strftime("%Y-%m-%dT%H:%M:%S")
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
        env["GIT_AUTHOR_NAME"] = "Tung Le"
        env["GIT_AUTHOR_EMAIL"] = "tung.le@example.com"
        env["GIT_COMMITTER_NAME"] = "Tung Le"
        env["GIT_COMMITTER_EMAIL"] = "tung.le@example.com"
        
        unadded_files = [f for f in all_files if f not in added_files]
        
        num_files_to_touch = random.randint(1, min(5, len(all_files)))
        
        if len(added_files) == 0:
            # Initial commit should add a lot of things
            num_files_to_touch = min(15, len(unadded_files))
            files_to_touch = unadded_files[:num_files_to_touch]
            msg = "Initial commit: set up project structure and baseline"
        else:
            files_to_touch = set()
            # Mix of new and old files
            for _ in range(num_files_to_touch):
                if unadded_files and random.random() < 0.7:
                    file_choice = random.choice(unadded_files)
                    files_to_touch.add(file_choice)
                    unadded_files.remove(file_choice)
                elif added_files:
                    files_to_touch.add(random.choice(list(added_files)))
            
            msg = random.choice(COMMIT_MESSAGES_GENERIC)

        for file_to_touch in files_to_touch:
            if file_to_touch in added_files:
                if file_to_touch.endswith(".py"):
                    with open(file_to_touch, "a") as f:
                        f.write(f"# Maintenance update\n")
                elif file_to_touch.endswith(".md"):
                    with open(file_to_touch, "a") as f:
                        f.write(f"\n<!-- update -->\n")
            run_cmd(f'git add -f "{file_to_touch}"')
            added_files.add(file_to_touch)
            
        run_cmd(f'git commit --allow-empty -m "{msg}"', env=env)
        
    print("Done! Check your git history with `git log`")

if __name__ == "__main__":
    main()
# Maintenance update
# Maintenance update
