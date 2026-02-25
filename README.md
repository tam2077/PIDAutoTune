# PIDAutoTune: Reinforcement Learning Sandbox

## Overview
This repository contains documentation and code for developing a program that can auto-tune Proportional-Integral-Derivative (PID) values in a three-dimensional, highly dynamic system, such as a drone. 

**PIDAutoTune** serves as an experimental machine learning sandbox and a proof of concept. It is built to explore how statistical modeling, Python programming, and machine learning can optimize dynamic systems. It acts as a hands-on learning environment for applying advanced Reinforcement Learning (RL) algorithms rather than a dedicated, production-ready robotics framework.

[![Drone Image](https://raw.githubusercontent.com/tam2077/PIDAutoTune/main/drone.png)](https://raw.githubusercontent.com/tam2077/PIDAutoTune/main/drone.png)

## Documentation
The full documentation for this project is available at:  
[https://tam2077.github.io/PIDAutoTune/index.html](https://tam2077.github.io/PIDAutoTune/index.html)

## Project Motivation
The primary goal of this project is to experiment with different RL training architectures to find an algorithmic solution to the tedious process of manual PID tuning. It is a space for testing out theoretical ML concepts on a simulated physical system to observe real-time convergence and performance differences between state-of-the-art RL models.

## Core Components (`controllers/` Directory)
The core logic of the learning algorithms and the environment simulation are isolated within the `controllers/` folder:

* **`droneRobot.py`**: The custom environment and simulation script. This defines the physics, state observations, reward functions, and the base PID controller structure that the AI agent interacts with.
* **`Train_PPO.py`**: Training loop utilizing **Proximal Policy Optimization (PPO)**. A reliable baseline for stable, on-policy reinforcement learning.
* **`Train_SAC.py`**: Training loop utilizing **Soft Actor-Critic (SAC)**. An off-policy algorithm that maximizes a trade-off between expected return and entropy, often highly effective for continuous control tasks.
* **`Train_TDP.py`**: An additional experimental training script designed to compare sample efficiency and policy convergence against the PPO and SAC models.

## Requirements
This project requires **PyTorch**. You can find the installation instructions for your specific system hardware [here](https://pytorch.org/get-started/locally/).

## Getting Started & Usage

1. **Clone the repository:**
   git clone https://github.com/tam2077/PIDAutoTune.git
   cd PIDAutoTune

2. **Set up the environment:** It is recommended to create a Conda environment before installing dependencies.
   conda create --name pid_autotune python=3.9
   conda activate pid_autotune

3. **Install dependencies:**
   pip install -r requirements.txt

4. **Run training:** Execute your preferred training method from the `controllers` folder.
   cd controllers
   python Train_SAC.py

### TensorBoard Output Example
You can monitor training metrics, reward progression, and agent performance via TensorBoard (logs are saved in `Training/Logs/`). Here is an example of the training output:

[![TensorBoard Example](https://raw.githubusercontent.com/tam2077/PIDAutoTune/main/tensorboard_output_example.png)](https://raw.githubusercontent.com/tam2077/PIDAutoTune/main/tensorboard_output_example.png)

## Disclaimer
This project is currently a work in progress and an experimental sandbox. It is strictly intended for the educational exploration of machine learning algorithms and is **not** intended or validated for use in critical systems or real-world flight environments.
