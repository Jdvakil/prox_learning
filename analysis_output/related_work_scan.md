# Related Work Scan — PLA (Proximity-Learned ACT)

Scan date: 2026-05-12. Target venue: CoRL 2026. Headline claim: 29 body-mounted 8x8 SPAD-style depth arrays on a Franka arm, fused with RGB through an ACT-style policy, give better grasp approach in clutter than RGB-only baselines (because wrist-RGB occludes at close range during the final approach).

This scan was conducted via web search and abstract review. Where a paper is title-only and I could not access the full text, I have flagged it as "verify before citing". Nothing here is a direct read of contents I did not see.

---

## 1. Scoop Risk Assessment

**Risk level: LOW–MODERATE. No URGENT scoop identified.**

I did not find a paper that does PLA's exact experiment: many-link, body-distributed SPAD depth arrays fused with RGB through an ACT-style imitation policy, benchmarked on cluttered-approach grasping with a wrist-RGB occlusion narrative. The closest hits are:

- **ProxySKIN / CySkin (Maiolino group, Sensors 2024)** — uses VL53L5CX-class multi-zone ToF arrays distributed across robot body, but presents as an HRI / collision-avoidance system, not a learned manipulation policy.
- **TetraGrip (2024 project)** — VL53L5CX ToF sensors on a multi-suction gripper for reactive clutter manipulation, but ToFs are on the end-effector (not the arm body) and the controller is reactive heuristics, not an ACT-style learned policy.
- **TACT (RAL 2025)** — extends ACT with a tactile modality on a humanoid, exactly the architectural shape of PLA, but uses tactile (contact) measurements rather than pre-contact proximity, and the task is whole-body contact manipulation, not cluttered grasp approach.
- **Reactive Grasping with Optical Proximity Sensors (Hsiao/Saxena/Ng, ICRA 2009)** — the foundational PLA-shaped paper, but uses fingertip-mounted IR proximity, hand-engineered reactive controllers (not ACT), no clutter narrative.
- **Improving Grasp Performance with In-Hand Proximity (Patel & Correll, ISER 2016 / 2018)** — elastomer-embedded proximity on Baxter gripper, again gripper-mounted not arm-distributed, and pipeline-based not end-to-end learned.

The combination of (i) full-arm SPAD coverage, (ii) ACT-style chunked imitation policy, (iii) cluttered approach with wrist-RGB occlusion as the explicit failure mode, appears not to be claimed in any 2024–2026 venue I could surface. Verify directly in CoRL 2026 OpenReview once submissions are public and in RSS 2026 proceedings — those are the most likely scoop sources.

---

## 2. Direct Prior Art (high overlap)

Papers since 2023 that put body- or end-effector-mounted proximity sensing into a learned manipulation policy.

| # | Paper | What they do | What's different from PLA | Source | Citation key |
|---|---|---|---|---|---|
| 1 | **ProxySKIN: Multi-Modal Robotic Skin for HRI** | Distributed ToF arrays + capacitive tactile patches covering robot surface, providing 360° proxi-tactile feedback | HRI / collision-avoidance demo; no end-to-end learned manipulation policy; uses VL53L5CX-like multi-zone ToF | Cannata, Maiolino et al., *Sensors* 24(4):1334, 2024 | `[CannataEtAl2024-ProxySKIN, Sensors 2024]` |
| 2 | **TetraGrip: Sensor-Driven Multi-Suction Reactive Object Manipulation in Cluttered Scenes** | VL53L5CX ToF imagers on a multi-suction gripper, real-time clutter reactivity | Sensors on gripper not arm; reactive control rather than ACT-style policy; suction not parallel-jaw | tetragrip.github.io project page, 2024 | `[TetraGrip2024]` (verify before citing — title-only project page seen) |
| 3 | **Reactive Grasping Using Optical Proximity Sensors** | Fingertip-mounted IR proximity on Barrett hand; probabilistic state estimator and hierarchical reactive controller | No learned policy, no clutter setting, no ACT, fingertip-only | Hsiao, Nangeroni, Huber, Saxena, Ng, ICRA 2009 | `[HsiaoEtAl2009-ReactiveGrasping, ICRA 2009]` |
| 4 | **Improving Grasp Performance Using In-Hand Proximity and Dynamic Tactile Sensing** | Elastomer-embedded IR proximity arrays on Baxter gripper, pipeline grasp planning | Gripper-mounted only; hand-engineered grasp planner; pre-ACT | Patel & Correll, ISER 2016 / ISRR 2017 chapter | `[PatelCorrell2016-InHandProximity]` |
| 5 | **Multi-modal Prosthetic Fingertip Sensor (PCF)** | Combined IR proximity + barometric force fingertip; NN for contact estimation | Prosthetic-hand context; not a manipulation policy across an arm | Segil, Patel, Klingner, Weir, Correll, *Adv Mech Eng* 2019 | `[SegilEtAl2019-PCF]` |
| 6 | **Electric Field Pretouch for Robotic Grasping** | EF sensors on fingers/hand; closed-loop pre-grasp servoing | Pre-learning era; EF not ToF; fingertip-mounted | Wistort & Smith, ICRA 2008 / Smith group | `[WistortSmith2008-EFPretouch, ICRA 2008]` |
| 7 | **Pre-Touch Sensing for Sequential Manipulation** | Seashell / optical pretouch for shape acquisition before contact | Hand-engineered exploration, single-point pretouch, no learned policy | Jiang & Smith, ICRA 2017 and earlier | `[JiangSmith-PreTouch]` |
| 8 | **A Gripper for Object Search and Grasp Through Proximity Sensing** | Proximity-instrumented gripper finds and grasps occluded objects | Gripper-only sensor placement; not ACT; not clutter-trained imitation | Yang et al., IROS 2018 | `[Yang2018-ProxGripper, IROS 2018]` (verify) |
| 9 | **SkinGrip: Adaptive Soft Manipulator with Capacitive Sensing for Whole-Limb Bed Bathing** | Capacitive proximity array on a soft robot for bathing assistance | Soft arm, capacitive (not ToF), assistive task not grasping | arxiv 2405.02772, 2024 | `[SkinGrip2024]` (verify before citing — title-only) |
| 10 | **Bio-Inspired Grasping Controller for Sensorized 2-DoF Grippers** | Proximity-instrumented 2DoF grippers with bio-inspired reactive controller | No learned policy; toy-gripper scale | Lach et al., 2022 (Skadge/Lach) | `[Lach2022-BioInspiredGrasp]` |
| 11 | **Proximity Skin Sensor Using ToF Sensor for Human Collaborative Robot** | String-like ToF modules wrappable around robot links, HRI safety | Pure safety/HRI; no policy learning | Tsuji & Kohama, IEEE Sensors J 2019 | `[TsujiKohama2019-ProxSkinToF]` |

**Summary**: existing proximity-on-robot work splits cleanly into (a) HRI / collision-avoidance with full-arm coverage but no learned manipulation policy, and (b) fingertip / gripper proximity for reactive grasping but without arm-body coverage or modern policy learning. PLA sits in the empty cell: arm-distributed proximity + ACT-style learned policy + grasp-approach task.

---

## 3. Adjacent Prior Art (medium overlap)

### 3a. Vision + tactile fusion for manipulation (no proximity)

These are the architectural / data-engineering siblings — they fuse a contact-modality with vision in a learned policy. PLA's contribution differs by using **pre-contact** distance rather than contact tactile.

- **TACT — Tactile-Modality Extended ACT** — Humanoid whole-body contact manipulation, RAL 2025. *Most architecturally similar paper to PLA — same ACT-extension recipe but with tactile instead of proximity.* `[TACT2025-RAL]`
- **TactileAloha — Bimanual ACT with tactile** — arxiv 2025. `[TactileAloha2025]` (verify)
- **3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing** — 16x16 tactile arrays per finger, imitation learning. arxiv 2410.24091. `[3DViTac2024]`
- **Reactive Diffusion Policy (RDP) — Slow-Fast Visual-Tactile Policy** — RSS 2025, Franka. `[RDP2025-RSS]`
- **Multimodal Force-Matched IL with a See-Through Visuotactile Sensor (STS)** — T-RO 2025 / ICRA 2025. `[Ablett2025-STS-IL, T-RO 2025]`
- **Tactile-Conditioned Diffusion Policy (FARM)** — arxiv 2510.13324, 2025. `[FARM2025]`
- **Visuo-Tactile Transformers for Manipulation** — CoRL 2022, Chen et al. `[Chen2023-VTT, CoRL 2022]`
- **Sparsh-skin** — pre-trained encoder for magnetic skin (Meta). 2025. `[Sparsh-skin2025]`
- **DexSkin — High-Coverage Conformable Capacitive Skin for Contact-Rich Manipulation** — CoRL 2025, gripper-finger coverage. `[DexSkin2025-CoRL]`
- **AnySkin — Plug-and-Play Magnetic Skin** — arxiv 2409.08276. `[AnySkin2024]`
- **NeuralFeels (Suresh, Lambeta, Calandra et al.)** — visuotactile in-hand, *Science Robotics* 2024. `[Suresh2024-NeuralFeels, SciRob 2024]`
- **DIGIT tactile sensor (Lambeta, Calandra)** — RA-L 2020. Foundational tactile-sensor citation. `[Lambeta2020-DIGIT, RA-L]`

### 3b. Proximity sensors for collision avoidance (no learned policy)

These are the "sensor hardware" siblings — they motivate the arm-distributed-proximity-sensor argument but stop short of learned manipulation.

- **HEX-O-SKIN (Mittendorfer & Cheng)** — humanoid multimodal active skin modules, IEEE T-RO 2011. Foundational robot-skin citation. `[Mittendorfer2011-HexOSkin, T-RO]`
- **Proximity Perception in Human-Centered Robotics: A Survey** — Escaida Navarro et al., arxiv 2108.07206 / journal version. Best survey reference. `[EscaidaNavarro2022-ProximitySurvey]`
- **Capacitive Tactile Proximity Sensing** — Escaida Navarro et al., book chapter / IROS. `[EscaidaNavarro2014-CTPS]`
- **Whole-arm sensing skin with sonar proximity, neural reactive control** — older foundational reference (1990s, Cheng-style). Title-only — verify. `[WholeArmSkin-RL, older]`
- **Quasi whole-body laser-ranging sensor rings for safe pHRI** — 2021, ResearchGate ref. `[LaserRanging-pHRI-2021]` (verify)
- **Peripersonal Space Learning with Artificial Skin (iCub)** — Roncone, Hoffmann, Pattacini, Fadiga, Metta, *PLOS ONE* 2016. Classic biological-inspiration citation. `[Roncone2016-Peripersonal, PLOS ONE]`
- **Characterisation of the VL53L5CX for Indoor Robotics** — MDPI Sensors 2024. Sensor-validation citation. `[VL53L5CX-Char2024]`

### 3c. ACT and chunking policies (architectural relatives)

These are the policy-architecture citations PLA inherits from.

- **ACT — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ALOHA)** — Zhao, Kumar, Levine, Finn. RSS 2023. **The headline architectural cite.** `[Zhao2023-ACT, RSS 2023]`
- **Diffusion Policy** — Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song. RSS 2023 / IJRR 2024. `[Chi2023-DiffusionPolicy, RSS 2023]`
- **InterACT — Inter-dependency Aware ACT with Hierarchical Attention** — arxiv 2409.07914. `[InterACT2024]`
- **Bi-ACT — Bilateral Control-Based ACT** — AIM. `[BiACT]` (verify)
- **OpenVLA — Open-Source Vision-Language-Action Model** — Kim, Pertsch et al., arxiv 2406.09246. `[Kim2024-OpenVLA]`
- **RT-2 — Vision-Language-Action Model** — Brohan et al., 2023. `[Brohan2023-RT2]`
- **Universal Manipulation Interface (UMI)** — Chi et al., 2024. `[Chi2024-UMI]`
- **BEHAVIOR Robot Suite** — arxiv 2503.05652, 2025. `[BRS2025]`
- **Observer-Actor: Active Vision IL** — arxiv 2511.18140, 2025. `[ObserverActor2025]` (verify — addresses wrist-occlusion narrative, useful adversary)
- **Distracted Robot: How Visual Clutter Undermines Robotic Manipulation** — arxiv 2511.22780, 2025. **Very useful citation for the clutter-occlusion motivation.** `[DistractedRobot2025]` (verify)

### 3d. SPAD / ToF sensors in robotics

- **VL53L5CX / VL53L8CX datasheet (STMicroelectronics)** — hardware reference. `[STMicro-VL53L5CX]`
- **TMF8828 (ams-osram)** — alternative 8x8 dToF SPAD. `[ams-TMF8828]`
- **Characterisation of the VL53L5CX for Indoor Robotics** — Sensors 2024 (as above). `[VL53L5CX-Char2024]`
- **TacEx: GelSight Tactile Simulation in Isaac Sim** — visuo-tactile simulator, useful for the sim-side methods discussion. `[TacEx2024]`
- **NVIDIA Isaac Sim proximity-sensor primitive** — for the simulation methods section. `[IsaacSim-ProximitySensor]`
- **Fundamental Limits to Depth Imaging with Single-Photon Detector Arrays** — *Sci Rep* 2022. Sensor-physics citation if depth-noise is discussed. `[SPAD-DepthLimits-2022]`

---

## 4. Likely citation list for the paper (sorted by paragraph)

These are BibTeX-key suggestions, grouped by where they probably land in the manuscript. ~20 entries.

**Intro / motivation paragraph (clutter, wrist-occlusion, pre-contact sensing):**
1. `Zhao2023-ACT` — sets the policy-learning baseline narrative
2. `Chi2023-DiffusionPolicy` — alternative imitation-learning architecture
3. `DistractedRobot2025` (verify) — clutter undermines vision policies
4. `EscaidaNavarro2022-ProximitySurvey` — survey citation that grounds "proximity sensing is a real modality"
5. `HsiaoEtAl2009-ReactiveGrasping` — pre-touch / proximity in grasping origin story

**Related work — sensing modalities:**
6. `Mittendorfer2011-HexOSkin` — multimodal robot skin foundational
7. `CannataEtAl2024-ProxySKIN` — closest hardware analogue
8. `TsujiKohama2019-ProxSkinToF` — ToF skin for HRI
9. `PatelCorrell2016-InHandProximity` — in-hand proximity foundational
10. `WistortSmith2008-EFPretouch` — pre-touch tradition (different modality)
11. `JiangSmith-PreTouch` — pre-touch in sequential manipulation
12. `Roncone2016-Peripersonal` — biological inspiration / peripersonal space

**Related work — vision-tactile / multimodal policies:**
13. `TACT2025-RAL` — direct architectural sibling (ACT + tactile)
14. `RDP2025-RSS` — alternative slow-fast architecture, contact-rich
15. `Ablett2025-STS-IL` — IL + see-through visuotactile
16. `3DViTac2024` — fine-grained imitation with 16x16 tactile
17. `DexSkin2025-CoRL` — most-recent skin-based policy learning (CoRL 2025)

**Related work — policy architectures:**
18. `Zhao2023-ACT` (re-cited)
19. `Kim2024-OpenVLA` — VLA comparison
20. `Chi2024-UMI` — wrist-camera teleop framework, contrast point

**Methods — hardware and simulation:**
21. `STMicro-VL53L5CX` — sensor datasheet
22. `VL53L5CX-Char2024` — sensor-characterisation evidence
23. `IsaacSim-ProximitySensor` — simulation pipeline
24. `TacEx2024` — sim-side framework (if applicable)

**Experiments / discussion:**
25. `TetraGrip2024` (verify) — clutter-grasping comparison with ToF
26. `ObserverActor2025` (verify) — wrist-occlusion alternative solution (active vision)

---

## 5. Gaps / Framing Opportunities

**Gap 1: Body-distributed pre-contact sensing has never been driven by a chunked, end-to-end imitation policy.** The proximity-perception literature (ProxySKIN, HEX-O-SKIN, Escaida Navarro's surveys) has built the hardware and produced reactive / safety controllers, but stops short of letting a modern transformer policy *learn* what to do with the proximity readings. The imitation-learning literature (ACT, Diffusion Policy, TACT, RDP) has built the policy machinery and demonstrated multimodal fusion, but has almost exclusively used contact tactile (DIGIT, GelSight, AnySkin, DexSkin) — not pre-contact distance. PLA's contribution is to fuse these two streams: the "pre-contact distance" sensing modality, fused at the level of an ACT-style policy, on a full-arm sensor layout. The architectural recipe is closest to TACT (ACT + auxiliary modality), but the sensing modality and task framing are different and complementary.

**Gap 2: The wrist-RGB occlusion failure mode of cluttered grasping is widely acknowledged but not solved by adding more vision.** Existing responses to wrist-occlusion include multi-view active vision (Observer-Actor), 3D Gaussian Splatting view selection, view-fusion networks, and wrist-view generation (WristWorld). All of these stay within the visual modality. PLA's framing — "occlusion at close range is fundamentally a vision-modality limitation, and pre-contact distance sensing is the right complement" — is, as far as this scan can tell, not the framing of any recent paper. This is a clean contribution sentence: *"Where prior work addresses close-range wrist-RGB occlusion with additional cameras or view-synthesis, we argue that the missing information is not visual at all — it is geometric, and is most cheaply obtained from body-distributed pre-contact sensors."*

**Suggested framing for the contribution paragraph**: (i) we are the first to fuse arm-distributed multi-zone SPAD ToF arrays with RGB inside an ACT-style chunked imitation policy; (ii) we identify wrist-RGB close-range occlusion as the specific failure mode that proximity addresses, and we show that PLA closes this gap on cluttered grasps without changing the visual front-end; (iii) we release the sensor-instrumented Franka dataset and the proximity-conditioned ACT codebase. Defensive citations to TACT (RAL 2025) and DexSkin (CoRL 2025) are essential — these are the papers a reviewer will reach for as "did you compare?" challenges.

---

## Notes on confidence

- High-confidence (read or strongly corroborated abstracts): ACT, Diffusion Policy, ProxySKIN, HEX-O-SKIN, Reactive Grasping (Hsiao 2009), Patel & Correll, Escaida Navarro survey, TACT, RDP, STS-IL, 3D-ViTac, DexSkin, AnySkin, OpenVLA, UMI, FingerVision, DIGIT.
- Title-only / verify before citing: TetraGrip, SkinGrip, BiACT, Distracted Robot, Observer-Actor, "Yang 2018 proximity gripper", the older "Whole-arm sensing skin RL" reference.
- Not found in this scan but plausible to exist: a CoRL/RSS 2025 paper specifically on "ACT + proximity" — re-check CoRL 2026 OpenReview when it opens, and run targeted searches on Google Scholar with sensor-vendor part numbers (VL53L5CX, VL53L8CX, TMF8828) plus "imitation learning".

If supervisor wants, the next-pass tasks would be: (a) actually pull the TACT 2025 paper PDF and check whether it compares against any proximity baseline, (b) email-search for any 2026 RSS or ICRA submissions with "proximity" + "imitation" / "ACT" in the title, (c) verify the TetraGrip publication venue and whether it has a learned-policy ablation.
