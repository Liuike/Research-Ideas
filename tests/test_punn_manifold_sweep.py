from concurrent.futures import ThreadPoolExecutor
import os
import uuid

import pytest

from optimizer_resurrection.punn_manifold_sweep import _exclusive_local_sweep


@pytest.mark.skipif(os.name != "nt", reason="local Windows controller")
def test_controller_mutex_rejects_concurrent_owner_and_releases_afterward():
    identity = "test-" + uuid.uuid4().hex

    def contend():
        with pytest.raises(RuntimeError, match="already owns"):
            with _exclusive_local_sweep(identity):
                pass

    with _exclusive_local_sweep(identity):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(contend).result(timeout=10)
    with _exclusive_local_sweep(identity):
        pass
