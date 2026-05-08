# Flight-Delay

This project analyzes flight delay data at DCA (Washington National Airport) and attempts to model the flight and weather history to classify whether a flight is delayed or not.

See [the project report](./report/Report.qmd) for more details.

## Pre-requisites

- [`uv`](https://astral.sh/uv/)
- Git LFS
- If running on macOS, you may need `libomp` installed via Homebrew.
  ```bash
  brew install libomp
  ```
    - If running on Apple Silicon (e.g., M1), you need to ensure Homebrew is ARM-based, and not x86_64. You can check this by running `brew --prefix`. If that returns `/opt/homebrew`, you're good to go; if not, you may need to install the correct version of Homebrew for XGBoost to work properly.
    - This resolves an error incorrectly stating that you are running 32-bit Python on a 64-bit OS.
- Initialize the virtual environment using `uv`:
  ```bash
  uv venv
  ```
    - Use this virtual environment as the Python kernel for any Jupyter notebooks.
    
- Quarto, for rendering the project report.
