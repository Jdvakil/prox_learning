# Local UID-cache fallback for machines without /weka

`molmo_spaces/utils/synset_utils.py` caches the pickupable-UID scan at
`VALID_PICKUPABLE_OBJA_UIDS_PATH`, which points into the cluster's `/weka`
mount. Off-cluster that path never exists, so **every process that imports the
datagen configs re-scans ~130k annotations with per-file grasp checks — 10 to
30 minutes cold, on every start.** (The scan also never writes the cache back.)

Fix: append this to the end of `molmo_spaces/utils/synset_utils.py` (it must
run after the original assignment, so end-of-file is correct):

```python
# Local override: the upstream cache path targets the cluster's /weka mount,
# absent on workstations, so the ~100k-file grasp scan re-ran on every import.
import os as _os  # noqa: E402

VALID_PICKUPABLE_OBJA_UIDS_PATH = _os.environ.get(
    "MLSPACES_UID_CACHE",
    _os.path.expanduser("~/.cache/molmospaces/valid_pickupable_obja_uids.txt"),
)
```

Then generate the cache once (15–30 min, one time):

```python
from pathlib import Path
from molmo_spaces.utils.synset_utils import (
    VALID_PICKUPABLE_OBJA_UIDS_PATH as P, get_valid_pickupable_obja_uids)
uids = get_valid_pickupable_obja_uids()
Path(P).parent.mkdir(parents=True, exist_ok=True)
Path(P).write_text("\n".join(uids) + "\n")
```

Every subsequent import is then instant. On cluster machines with `/weka` this
patch is unnecessary (though harmless).
