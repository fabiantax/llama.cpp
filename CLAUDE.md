IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Git Commit Rules

- **NEVER use "Claude" as the git author.** All commits must use the repository’s configured git user identity (`Fabian Tax <fabiantax@hotmail.com>`). Do not set GIT_AUTHOR_NAME or GIT_AUTHOR_EMAIL to Claude/Anthropic values.
- When resolving rebase conflicts, ensure the original author is preserved or reset to the repo owner — never to Claude.
- The `Co-Authored-By: Claude` trailer in the commit message body is fine, but the Author field must always be the repo owner.

## Task Confidence Tracking

- For each task or sub-task, output a **confidence %** estimate before starting work.
- If confidence falls below **70%**, search for more information (code, docs, arXiv papers, web) to increase confidence before proceeding.
- Use strategies like mermaid diagrams, hex dumps, or structured analysis to raise confidence when dealing with complex formats or protocols.
- Re-evaluate confidence after each significant discovery or blocker.
