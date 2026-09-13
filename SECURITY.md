# Security policy

- Report vulnerabilities privately to the repository owner (see `git log` for the maintainer address); do not open public issues for security reports. Expect an acknowledgement within 3 working days.
- Supported: the latest tagged release and `main`.
- Controls and threat model: README "Security model". Operational response: README "Operations notes".
- Never commit `.env`, tokens, or licensed source PDFs; CI runs gitleaks, pip-audit, bandit, semgrep and Trivy on every change.
