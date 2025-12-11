# Contributing to ECG Analyzer

Thank you for your interest in contributing to the ECG Analyzer project! This project is intended for educational and research purposes.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone the project** to your local machine:
    ```bash
    git clone https://github.com/your-username/ecg_analyzer.git
    cd ecg_analyzer
    ```
3.  **Set up the environment**:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    
    pip install -r requirements.txt
    ```

## Development Guidelines

-   **Code Style**: Try to follow PEP 8 for Python code.
-   **Structure**: Keep backend logic in `backend/` and frontend in `templates/` and `static/`.
-   **Testing**: If you add new analytical features, please test them with provided sample data.

## Submitting Changes

1.  Create a new branch for your feature or fix:
    ```bash
    git checkout -b feature/amazing-feature
    ```
2.  Commit your changes:
    ```bash
    git commit -m "Add some amazing feature"
    ```
3.  Push to the branch:
    ```bash
    git push origin feature/amazing-feature
    ```
4.  Open a **Pull Request** on the main repository.

## Reporting Bugs

If you find a bug, please open an issue describing:
-   What you were trying to do.
-   What happened (error messages, screenshots).
-   What you expected to happen.

Thank you for contributing! ❤️
