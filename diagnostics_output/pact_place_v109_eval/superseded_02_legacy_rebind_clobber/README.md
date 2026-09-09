# Superseded V10.9 infrastructure smoke, attempt 02

Retained, not deleted. `infrastructure_healthy: false`, 0 of 8 rollouts
completed, full evaluation correctly refused, no evaluation instance touched.

Attempt 01's defect was fixed — every smoke row resolved this time. The next
one surfaced immediately:

    ValueError: unregistered task_sampler_class 'PactPlaceCorridorV106Sampler'

`eval_pact_place_row.main()` assigns its **own** module globals onto
`eval_pact_collision_row` as its last act before delegating. The V10.9 wrapper
had patched `legacy.*` directly and then called `place.main()`, which promptly
overwrote all of it — restoring `place`'s V2-era sampler allow-list, and, more
seriously, `PactPlaceEvalConfig`, whose `scene_xml_paths` is the **V2 place
scene**. The visible error was the sampler allow-list; the silent one would
have been the scene.

It could not actually have produced a wrong-scene result: the V10.6 sampler
verifies the loaded scene against the row's `pact_v106_scene_sha256` and
refuses a mismatch. The run would have failed rather than lied. But it failed
for the wrong reason, and the guard should not be the only thing standing.

Fix: `install_v109_bindings()` patches the **`place`** module's globals, so
`place.main()` installs the V10.9 objects itself. `bindings_are_v109()` reports
what `legacy` ends up holding, and
`test_v109_bindings_survive_delegation_to_the_legacy_evaluator` reproduces the
delegation block and asserts all six survive.
