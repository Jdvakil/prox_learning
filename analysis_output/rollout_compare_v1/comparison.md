# PLA vs baseline rollout comparison

- Paired episodes: **18**

- PLA success rate:      **0.0%**
- Baseline success rate: **0.0%**

## Bucket counts

- A_baseline_fail_pla_success: **0**
- B_baseline_success_pla_fail: **0**
- C_both_success: **0**
- D_both_fail: **18**

## Behavioural gap (positive = PLA better)

- Mean approach Δ (m, start→end) PLA:      +0.020
- Mean approach Δ (m, start→end) baseline: +0.037
- Mean gripper-open fraction PLA:      93.0%
- Mean gripper-open fraction baseline: 75.0%

## Per-episode

| house | traj | task | PLA success | base success | PLA Δapproach (m) | base Δapproach (m) | PLA grip_open% | base grip_open% |
|-------|------|------|-------------|--------------|-------------------|--------------------|----------------|-----------------|
| 11 | traj_0 | Pick up the brown pinecone keychain with charm and | . | . | +0.022 | -0.020 | 100% | 84% |
| 11 | traj_1 | Pick up the brown pinecone keychain with charm and | . | . | +0.048 | -0.023 | 94% | 100% |
| 12 | traj_0 | Pick up the sleek slim dark phone with camera and  | . | . | +0.024 | +0.038 | 86% | 54% |
| 12 | traj_1 | Pick up the personal credit line and place it in o | . | . | +0.040 | +0.069 | 83% | 65% |
| 13 | traj_0 | Pick up the transparent glass salt shaker and plac | . | . | +0.019 | +0.062 | 83% | 68% |
| 13 | traj_1 | Pick up the transparent glass pepper shaker and pl | . | . | -0.050 | -0.020 | 44% | 90% |
| 14 | traj_0 | Pick up the instrumentality and place it in or on  | . | . | +0.071 | +0.118 | 100% | 48% |
| 14 | traj_1 | Pick up the instrumentality and place it in or on  | . | . | -0.024 | +0.090 | 100% | 39% |
| 15 | traj_0 | Pick up the instrumentation and place it in or on  | . | . | +0.003 | -0.001 | 100% | 100% |
| 15 | traj_1 | Pick up the salt shaker and place it in or on the  | . | . | +0.002 | +0.125 | 100% | 56% |
| 16 | traj_0 | Pick up the metal knife and place it in or on the  | . | . | -0.002 | -0.028 | 100% | 100% |
| 16 | traj_1 | Pick up the green slender pointed tool and place i | . | . | -0.034 | -0.036 | 100% | 100% |
| 18 | traj_0 | Pick up the curved hunting knife and place it in o | . | . | +0.037 | +0.044 | 100% | 79% |
| 18 | traj_1 | Pick up the saltshaker and place it in or on the v | . | . | +0.011 | +0.015 | 100% | 100% |
| 19 | traj_0 | Pick up the golden brown crescent shaped croissant | . | . | +0.061 | +0.027 | 83% | 100% |
| 19 | traj_1 | Pick up the golden brown crescent shaped croissant | . | . | +0.063 | +0.114 | 100% | 62% |
| 20 | traj_0 | Pick up the slim blue remote control with buttons  | . | . | +0.026 | +0.084 | 100% | 64% |
| 20 | traj_1 | Pick up the blue rectangular alarm clock with knob | . | . | +0.038 | +0.007 | 100% | 41% |