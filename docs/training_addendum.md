# Training for Spin-up #

To train the model for balancing during spin-up in this vibration control simulation, modify the training configuration to enable motor spin-up instead of keeping it "off". The current setup in `run_training()` sets the motor profile to "off" for both training and evaluation environments, which prevents spin-up dynamics. Change this to use a profile like "ramp" or "trapezoidal" to simulate the motor accelerating to target spin rate over time.

For example, update the training code to:
- Set `train_cfg.motor = MotorConfig(profile="ramp")` instead of "off"
- Adjust the curriculum callback to include spin-up phases, potentially starting with low spin rates and ramping up

The existing SAC reward function already penalizes transverse angular velocity (wobble), so the agent will learn to use water tank actuators to minimize vibrations as spin rate increases. Train with the curriculum that progressively adds imbalance disturbances while the motor spins up.

To summarize results in the database, use the existing `ExperimentRecorder` which stores episode-level metrics (total reward, final nutation angle, CM offset) and per-timestep data (omega components, CM offset magnitude, etc.). For aggregation:

- Use `get_reward_curve()` to plot reward progression over episodes
- Use `get_nutation_curve()` to track nutation reduction
- Query episode summaries with SQLAlchemy to compute averages, standard deviations, or best-performing episodes
- For spin-up specific metrics, add custom queries to analyze omega_z (spin rate) evolution and vibration stability during ramp-up phases

The database schema supports detailed telemetry, so you can filter timesteps by time ranges corresponding to spin-up periods and compute metrics like maximum nutation during acceleration.


# Training with background processes #
To kick off a training run in the background and periodically check its progress in this vibration control project, follow these steps. The training uses the SAC (Soft Actor-Critic) algorithm via Stable Baselines3, orchestrated through the `habitat_sim/scripts/train_agent.py` script.

### 1. **Starting the Training Run in the Background**
The training script (`habitat_sim/scripts/train_agent.py`) runs synchronously by default, blocking the terminal until completion. To run it in the background:

- **On Windows (PowerShell, which is your default shell):**
  Use `Start-Job` to run the script as a background job.
  ```
  $job = Start-Job -ScriptBlock { python habitat_sim/scripts/train_agent.py --config path/to/config.json --timesteps 1000000 --log-dir ./logs/training_run }
  ```
  - Replace `path/to/config.json` with your experiment config file (e.g., `habitat_sim/configs/reference_config.json`).
  - Adjust other arguments like `--timesteps`, `--n-envs`, etc., as needed (see the script's help with `python habitat_sim/scripts/train_agent.py --help`).
  - The job runs asynchronously, and you can continue using the terminal.

- **Alternative: Use `&` if in a bash-like environment (e.g., via WSL or Git Bash):**
  ```
  python habitat_sim/scripts/train_agent.py --config path/to/config.json --timesteps 1000000 --log-dir ./logs/training_run &
  ```

The training will log progress to the console (if not redirected) and save models/checkpoints to the specified `--log-dir`.

### 2. **Periodically Interrogating Progress**
The training process provides several ways to monitor progress without interrupting it:

- **Check Job Status (PowerShell):**
  ```
  Get-Job -Id $job.Id
  ```
  - This shows if the job is running, completed, or failed. For detailed output, use `Receive-Job -Id $job.Id -Keep` (keeps the output buffered).

- **Monitor Log Files:**
  - Training logs are written to the `--log-dir` (e.g., `./logs/training_run`).
  - Look for files like `progress.csv` or TensorBoard logs in `tb/` subdirectory (if TensorBoard is installed).
  - The script uses `log_interval=10`, so progress is logged every 10 training steps. Check the console output or log files for updates like:
    ```
    ---------------------------------
    | rollout/           |          |
    |    ep_len_mean     | 1e+03    |
    |    ep_rew_mean     | -1.23    |
    | time/              |          |
    |    fps             | 123      |
    |    iterations      | 10       |
    |    time_elapsed    | 456      |
    |    total_timesteps | 10000    |
    ---------------------------------
    ```
  - Periodically run `Get-Content ./logs/training_run/progress.csv | Select-Object -Last 10` to see recent logs.

- **Check for Checkpoints and Models:**
  - Checkpoints are saved every `checkpoint_freq` steps (default from config) to `./logs/training_run/checkpoints/`.
  - The best model is saved to `./logs/training_run/best_model.zip` when evaluation improves.
  - Run a script periodically to list files: `Get-ChildItem ./logs/training_run/checkpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 5` to see recent checkpoints.

- **TensorBoard Monitoring (if installed):**
  - If TensorBoard is available, start it in another terminal: `tensorboard --logdir ./logs/training_run/tb`.
  - Access via browser at `http://localhost:6006` to view real-time metrics like rewards, losses, and episode lengths.

- **Database Recording (if enabled):**
  - If using `--db path/to/database.db`, progress is recorded to an SQLite database. Query it periodically with SQL tools or scripts to check metrics.

- **Automated Polling:**
  - Write a simple PowerShell script to check progress every few minutes:
    ```
    while ($true) {
        $latestCheckpoint = Get-ChildItem ./logs/training_run/checkpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Host "Latest checkpoint: $($latestCheckpoint.Name) at $($latestCheckpoint.LastWriteTime)"
        Start-Sleep -Seconds 300  # Check every 5 minutes
    }
    ```

Training can take hours or days depending on `total_timesteps`. If the job fails, check the job's error output with `Receive-Job -Id $job.Id`. For more details, refer to the [training addendum](docs/training_addendum.md) or config files.
