  DroneRobot Documentation

DroneRobot Documentation
========================

This documentation provides details about the DroneRobot simulation project.

Overview
--------

The **DroneRobot** project simulates and controls a robotic drone in a Webots environment. It includes functionality for drone motion control, environment interaction, and reinforcement learning.

Features
--------

*   PID controller for precise drone motion control.
*   Integrated observation and action spaces for environment interaction.
*   Reward-based mechanism for task evaluation.
*   Supports TensorBoard for monitoring progress.
*   Reset and step functionality for episodic simulations.

Dependencies
------------

*   **Webots** (R2020b or later) for simulation.
*   **Python 3.6+** for scripting and control.
*   Required Python libraries:
    *   NumPy
    *   OpenAI Gym
    *   TensorFlow (optional, for TensorBoard)

Usage
-----

1.  Set up the Webots environment and install the required dependencies.
2.  Initialize the `DroneRobot` class to control the drone in the simulation.
3.  Use the provided methods such as `step`, `reset`, and `apply_action` for interaction and control.
4.  Utilize the reward system and TensorBoard for reinforcement learning tasks.

Authors
-------

The project uses several different opensource libraries. All rights reserved by their respective owners

© 2024 DroneRobot Project.