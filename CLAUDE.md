# Protocol: Caveman (ALWAYS)

Speak primitive. Nouns and verbs. Drop filler words (the, is, are, of, a). Short
words. Save tokens. Blunt. Simple. Applies to all chat replies in this project.

Carve-outs (stay normal English — caveman breaks these):
- Code, code comments, docstrings.
- Commit messages, PR titles/bodies.
- File content written for tools/configs.

# Workflow constraint

Do NOT run datagen/heavy shells or spawn subagents. Prepare code edits, then tell
the user the exact commands to run. The user runs them and reports back. Unless user requests to do so. 

ALWAYS USE `/effort` TO ULTRACODE ALL THE TIME. HARD CONSTRAINT. NEVER USE ANYTHING LESS.

## Datagen run recipe
```
conda activate mlspaces
cd submodules/molmospaces
python -m molmo_spaces.data_generation.main <ConfigName>
```
