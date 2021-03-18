# Multi-Task-Learning-PyTorch
## Installation
```bash
$ git clone git@github.com:tomoino/Multi-Task-Learning-PyTorch.git
```

## Usage
### Setup
```bash
$ cd Multi-Task-Learning-PyTorch
$ sh docker/build.sh
$ sh docker/run.sh
$ sh docker/exec.sh
```

### Training
```bash
$ python train.py
```
#### Grid Search
You can run train.py with multiple different configurations.
```bash
$ python train.py -m \
    project.train.batch_size=16,32 \
    project.train.optimizer.lr=0.01,0.001
```
#### Evaluation
```bash
$ python train.py eval=True project.model.initial_ckpt=best_ckpt.pth
```

### Check the results
You can use MLflow to check the results of your experiment.
Access http://localhost:8888/ from your browser.
If necessary, you can edit env.sh to change the port.

## Structure
```bash
$ tree -I "datasets|mlruns|__pycache__|outputs|multirun"
.
├── README.md
├── configs
│   ├── config.yaml
│   ├── hydra
│   │   └── job_logging
│   │       └── custom.yaml
│   ├── project
│   │   └── default.yaml
│   └── supported_info.py
├── data
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset
│   │   ├── cifar10.py
│   │   └── omniglot.py
│   ├── helper.py
│   └── sampler
│       └── balanced_batch_sampler.py
├── docker
│   ├── Dockerfile
│   ├── build.sh
│   ├── env.sh
│   ├── exec.sh
│   ├── init.sh
│   ├── requirements.txt
│   └── run.sh
├── metrics
│   ├── __init__.py
│   └── classification_metric.py
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── helper.py
│   └── networks
│       ├── resnet18.py
│       └── simple_cnn.py
├── train.py
└── trainers
    ├── __init__.py
    ├── base_trainer.py
    └── default_trainer.py
```
