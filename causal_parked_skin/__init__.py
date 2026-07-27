"""Offline learnability study for CAUSAL_PARKED_SKIN_REFERENCE_V1.

Reads the frozen parked-skin supervision dataset, trains a causal estimate of the parked
proximity field from deployable inputs only, and routes it through the frozen SafetyHead.
The dataset is never modified and the SafetyHead is never trained.
"""

REFERENCE_ID = "CAUSAL_PARKED_SKIN_REFERENCE_V1"
