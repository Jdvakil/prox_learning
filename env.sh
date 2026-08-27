export PY=$HOME/molmo_test/molmospaces/.venv/bin/python
export PYTHONPATH=$HOME/fh_run:$HOME/molmo_test/prox_learning/ms_main
export MLSPACES_ASSETS_DIR="$HOME/.cache/molmospaces/assets/$(printf '%s' "$HOME/molmo_test/prox_learning/submodules/molmospaces" | base64 -w0 | tr '+/' '-_' | tr -d '=')"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export HDF5_USE_FILE_LOCKING=FALSE
