# PIDAutoTune: Reinforcement Learning Sandbox

## Overview
**PIDAutoTune** is an experimental machine learning sandbox designed to tackle a specific, real-world control problem: auto-tuning Proportional-Integral-Derivative (PID) values in a highly dynamic, three-dimensional system. 

This repository serves as a proof-of-concept and a hands-on learning environment for applying advanced Reinforcement Learning (RL) algorithms to continuous control challenges. It is built to explore how statistical modeling, Python programming, and machine learning can optimize dynamic systems, rather than acting as a dedicated, production-ready robotics framework.

## Project Motivation
The primary goal of this project is to experiment with different RL training architectures to find a programmatic, algorithmic solution to the tedious process of manual PID tuning. It is a space for testing out theoretical ML concepts on a simulated physical system to observe real-time convergence and performance differences between state-of-the-art RL models.

## Core Components (`controllers/` Directory)
The core logic of the learning algorithms and the environment simulation are isolated within the `controllers/` directory:

* **`droneRobot.py`**: The custom environment and simulation script. This defines the physics, state observations, reward functions, and the base PID controller structure that the AI agent interacts with and attempts to optimize.
* **`Train_PPO.py`**: Training loop utilizing **Proximal Policy Optimization (PPO)**. A reliable baseline for stable, on-policy reinforcement learning.
* **`Train_SAC.py`**: Training loop utilizing **Soft Actor-Critic (SAC)**. An off-policy algorithm that maximizes a trade-off between expected return and entropy, often highly effective for continuous, complex control tasks like multidimensional PID tuning.
* **`Train_TDP.py`**: An additional experimental training script (utilizing Twin Delayed or temporal-difference-based methods) designed to compare sample efficiency and policy convergence against the PPO and SAC models.

## Getting Started

### Prerequisites
* Python 3.x
* PyTorch
* Additional dependencies listed in `requirements.txt`

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/tam2077/PIDAutoTune.git](https://github.com/tam2077/PIDAutoTune.git)
    cd PIDAutoTune
    ```
2.  Set up your environment (Conda is recommended):
    ```bash
    conda create --name pid_autotune python=3.9
    conda activate pid_autotune
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
To begin an experimental training session, navigate to the `controllers` directory and execute your preferred RL algorithm. For example:

```bash
cd controllers
python Train_SAC.py
```
You can monitor training metrics, reward progression, and agent performance via TensorBoard (logs are saved in Training/Logs/).

```Disclaimer
This project is currently a work-in-progress experimental sandbox. It is strictly intended for the educational exploration of machine learning algorithms and is not designed or validated for use in critical, real-world flight systems or production environments.
```

