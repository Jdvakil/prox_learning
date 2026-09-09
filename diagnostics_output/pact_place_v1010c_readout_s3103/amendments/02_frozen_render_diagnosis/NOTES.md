# Frozen replay rendering diagnosis

The supervisor exited 1 after observing the one frozen smoke worker exit 0. Training had not started. The rollout remains in the valid ledger and is never substituted into the original frozen50 denominator.

All non-RGB initial arrays matched exactly. Nine raw RGB channel values changed by one level (0.001366% of channels), within the predeclared max2 /0.1% tolerance. Only three channel values differed after resizing to the policy input. Independent inference with the original frozen policy and encoder reproduced both retained first action vectors bit for bit from their own saved inputs. Joint-state and encoded-proximity tensors were identical; image tensors were different. Repeated inference on each image was exact, and the reconstructed first-vector delta exactly matched the retained delta.

The full rollout action trace is not bit-identical. Maximum commanded-arm difference is0.003294rad; gripper commands are exact. Both rollouts failed the task and had identical collision-free-task-success status. The evidence explains the first-query divergence and confirms the model binding, without claiming exact full-trajectory reproducibility or choosing a replacement smoke.

The supervisor now requires the bound reconstruction evidence, source hashes, exact reconstructed first vectors and differences, identical non-RGB policy inputs, the original initial-state matching check, and unchanged gripper/outcome checks to accept this particular retained smoke. A four-case test accepts the real evidence and rejects altered prediction, proximity, or source-hash evidence. No upstream encoder bytes or scientific training/evaluation choices changed. The original48h deadline remains.
