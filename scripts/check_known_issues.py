#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Verify KNOWN_ISSUES.md is well-formed and up-to-date.

Checks:
1. "Last Updated" date is present and formatted correctly
2. "Last Updated" date is reasonably recent relative to file modification
3. All resolved issues have dates
4. No duplicate issue titles
5. Required sections exist

Exit codes:
    0 - All checks pass (with possible warnings)
    1 - Errors found (file is malformed)

Usage:
    python scripts/check_known_issues.py
    python scripts/check_known_issues.py --strict  # Treat warnings as errors
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path


def find_project_root() -> Path:
    """Find the project root by looking for KNOWN_ISSUES.md."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "KNOWN_ISSUES.md").exists():
            return current
        current = current.parent
    # Fallback to current directory
    return Path.cwd()


def check_last_updated(content: str, file_mtime: datetime) -> tuple[list, list]:
    """Check the Last Updated date."""
    errors = []
    warnings = []
    
    match = re.search(r"\*\*Last Updated:\*\*\s*(\d{4}-\d{2}-\d{2})", content)
    if not match:
        errors.append("Missing or malformed 'Last Updated' date. "
                      "Expected format: **Last Updated:** YYYY-MM-DD")
        return errors, warnings
    
    try:
        last_updated = datetime.strptime(match.group(1), "%Y-%m-%d")
    except ValueError:
        errors.append(f"Invalid date format: {match.group(1)}. Expected YYYY-MM-DD")
        return errors, warnings
    
    # Check if file was modified more than 30 days after "Last Updated"
    if file_mtime - last_updated > timedelta(days=30):
        warnings.append(
            f"'Last Updated' ({match.group(1)}) may be stale. "
            f"File was last modified on {file_mtime.date()}. "
            f"Consider updating the date if content changed."
        )
    
    return errors, warnings


def check_resolved_issues(content: str) -> tuple[list, list]:
    """Check that resolved issues have dates."""
    errors = []
    warnings = []
    
    # Find the resolved issues section
    resolved_section = re.search(
        r"## Resolved Issues.*?(?=^## |\Z)", 
        content, 
        re.MULTILINE | re.DOTALL
    )
    
    if not resolved_section:
        # No resolved section is okay
        return errors, warnings
    
    resolved_text = resolved_section.group()
    
    # Find all resolved issue headings
    issues = re.findall(r"^### (.+)$", resolved_text, re.MULTILINE)
    
    for issue in issues:
        # Each resolved issue should have a date in format (YYYY-MM-DD)
        if not re.search(r"\(\d{4}-\d{2}-\d{2}\)", issue):
            warnings.append(
                f"Resolved issue missing date: '{issue}'. "
                f"Format should be: '### V1-Phase X: Title (YYYY-MM-DD)'"
            )
    
    return errors, warnings


def check_duplicate_titles(content: str) -> tuple[list, list]:
    """Check for duplicate issue titles."""
    errors = []
    warnings = []
    
    # Find all level-3 headings
    titles = re.findall(r"^### (.+)$", content, re.MULTILINE)
    
    seen = {}
    for title in titles:
        # Normalize title for comparison (strip dates)
        normalized = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)\s*", "", title).strip()
        if normalized in seen:
            errors.append(
                f"Duplicate issue title: '{title}' "
                f"(also appears as '{seen[normalized]}')"
            )
        else:
            seen[normalized] = title
    
    return errors, warnings


def check_required_sections(content: str) -> tuple[list, list]:
    """Check that key sections exist."""
    errors = []
    warnings = []
    
    # These sections should exist in KNOWN_ISSUES.md
    recommended_sections = [
        "Known Issues",          # Main title
        "Applies To",            # Version info
    ]
    
    for section in recommended_sections:
        if section not in content:
            warnings.append(f"Missing recommended content: '{section}'")
    
    return errors, warnings


def check_broken_links(content: str, project_root: Path) -> tuple[list, list]:
    """Check for obviously broken internal links."""
    errors = []
    warnings = []
    
    # Find markdown links to local files
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    
    for text, target in links:
        # Skip external links
        if target.startswith(('http://', 'https://', '#')):
            continue
        
        # Resolve relative to project root
        target_path = project_root / target.split('#')[0]
        
        if not target_path.exists():
            warnings.append(f"Potentially broken link: [{text}]({target})")
    
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Verify KNOWN_ISSUES.md is well-formed"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    args = parser.parse_args()
    
    project_root = find_project_root()
    known_issues = project_root / "KNOWN_ISSUES.md"
    
    if not known_issues.exists():
        print(f"ERROR: KNOWN_ISSUES.md not found in {project_root}")
        return 1
    
    content = known_issues.read_text()
    file_mtime = datetime.fromtimestamp(known_issues.stat().st_mtime)
    
    all_errors = []
    all_warnings = []
    
    # Run all checks
    checks = [
        ("Last Updated", check_last_updated, (content, file_mtime)),
        ("Resolved Issues", check_resolved_issues, (content,)),
        ("Duplicate Titles", check_duplicate_titles, (content,)),
        ("Required Sections", check_required_sections, (content,)),
        ("Broken Links", check_broken_links, (content, project_root)),
    ]
    
    for name, check_fn, check_args in checks:
        errors, warnings = check_fn(*check_args)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    # Report results
    if all_errors:
        print("ERRORS:")
        for e in all_errors:
            print(f"  ✗ {e}")
        print()
    
    if all_warnings:
        print("WARNINGS:")
        for w in all_warnings:
            print(f"  ⚠ {w}")
        print()
    
    if not all_errors and not all_warnings:
        print("✓ KNOWN_ISSUES.md: All checks passed")
        return 0
    
    if all_errors:
        print(f"✗ {len(all_errors)} error(s), {len(all_warnings)} warning(s)")
        return 1
    
    if all_warnings and args.strict:
        print(f"✗ {len(all_warnings)} warning(s) (strict mode)")
        return 1
    
    print(f"✓ {len(all_warnings)} warning(s), no errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
