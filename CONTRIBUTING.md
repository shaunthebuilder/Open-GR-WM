# Contributing to Open GR--WM

Thanks for contributing.

## Development Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Make your change and test locally.

## Contribution Flow
1. Fork the repository.
2. Create a branch from `main`.
3. Keep commits focused and descriptive.
4. Open a pull request with:
   - clear summary of changes
   - test evidence (screenshots/logs if UI-related)
   - notes on backward compatibility

## Code Guidelines
- Keep the app local-first (no external API dependency for core functionality).
- Preserve strict-grounded answer behavior in chat.
- Avoid committing runtime artifacts (`rag_store/`, caches, temp files).
- Update docs when behavior or UX changes.

## Reporting Issues
Use the issue templates for bug reports and feature requests.
