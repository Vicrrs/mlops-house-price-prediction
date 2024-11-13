# tests/security/test_security.py
import subprocess
import json


def test_security_issues():
    result = subprocess.run(
        ["bandit", "-r", "src/", "-f", "json"], capture_output=True, text=True
    )
    issues = json.loads(result.stdout)
    vulnerabilities = [
        issue
        for issue in issues.get("results", [])
        if issue["issue_severity"] in ("MEDIUM", "HIGH")
    ]
    assert len(vulnerabilities) == 0, f"Vulnerabilidades encontradas: {vulnerabilities}"
