1. **Install Python 3.9+** and ensure `pip` is available.
2. **Install dependencies.**  The game requires `pygame` for graphics and `tkinter` for the GUI.  On most Linux distributions `tkinter` is provided by the system package manager (e.g. `apt install python3‑tk`).  Use `pip` to install the remaining packages:

   ```bash
   pip install pygame pandas
   ```

3. **Run the game.**  Navigate into the `pacman_deployment` directory and execute:

   ```bash
   python3 PAC‑MAN_patched_v23_10_headless_pump_guard.py
   ```

   When prompted, you can create a new simulation or load the provided `Final Simulation.qpac` to continue training.  At the end of each episode the updated Q‑table and metrics will be saved alongside the original files.

4. **Headless training.**  If you want to train without opening a game window (for example on a server), set the `render` parameter to `0` when creating a new simulation.  You must also instruct SDL to use a dummy video driver so that `pygame` can run without a display.  Before running the script, set the environment variable:

   ```bash
   export SDL_VIDEODRIVER=dummy
   ```

## Deploying on a Website or Cloud Storage

The Q‑table (`*.qpac`) and metrics CSV (`*_metrics.csv`) are ordinary files that can be served from any web server or cloud storage provider.  After running training locally, simply upload these files to your chosen platform (for example GitHub, AWS S3, Google Drive or your personal web hosting).  Visitors can then download the files to continue training or analyse the metrics.

For automated deployments you can modify the Python script to save updated Q‑tables directly into a cloud bucket by replacing the `save_simulation_file` and `_append_metrics_row` functions with code that writes to your storage API.

## Notes

* The game uses `tkinter` for its GUI.  If you encounter `ModuleNotFoundError: No module named 'tkinter'` it means that the Tk libraries are missing on your system.  On Debian/Ubuntu run `sudo apt install python3‑tk` to install them.
* Training can take a long time.  The provided `Final Simulation.qpac` has been trained for roughly 5 000 episodes and demonstrates reasonable behaviour.  Further training may improve the agent’s performance.

If you plan to embed the game in a web page, consider rewriting the interface using a web framework (e.g. Flask or Streamlit) and exposing a REST API for uploading/downloading Q‑tables.  The core logic (the `Game` and `QLearningAgent` classes) can be reused without the `tkinter` components.