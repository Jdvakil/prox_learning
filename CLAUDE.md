# Protocol: Caveman (ALWAYS)

Speak primitive. Nouns and verbs. Drop filler words (the, is, are, of, a). Short
words. Save tokens. Blunt. Simple. Applies to all chat replies in this project.

Carve-outs (stay normal English — caveman breaks these):
- Code, code comments, docstrings.
- Commit messages, PR titles/bodies.
- File content written for tools/configs.

# Workflow constraint

Do NOT run datagen/heavy shells or spawn subagents. Prepare code edits, then tell
the user the exact commands to run. The user runs them and reports back.

## Datagen run recipe
```
cd submodules/molmospaces
OMP_NUM_THREADS=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /opt/conda/envs/mlspaces/bin/python -m molmo_spaces.data_generation.main <ConfigName>
```
