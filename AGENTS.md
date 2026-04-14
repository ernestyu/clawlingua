# AGENTS.md

## General Rules

* Follow the instructions in SPEC.md if present
* Do not modify unrelated files
* Keep changes minimal and focused
* Preserve existing behavior unless explicitly asked to change

---

## Python Environment

* Always use `.venv` for Python execution
* Do not use system Python
* If `.venv` does not exist:

  * create it with `python3 -m venv .venv`
  * activate it
  * install dependencies from `requirements.txt`

---

## Testing

* Always write tests for new or modified logic
* Use `pytest`
* Cover edge cases and error handling
* Ensure all tests pass before finishing

---

## Refactoring

* Do not change functionality during refactoring
* Preserve public interfaces unless instructed otherwise
* Prefer small, incremental changes over large rewrites

---

## Execution Strategy

* For large tasks:

  * first explain the plan
  * then implement step by step
* Do not perform large multi-module changes in a single step unless explicitly requested

---
