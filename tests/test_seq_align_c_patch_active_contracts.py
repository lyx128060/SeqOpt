import pytest
from seq_align.assembler import verify_c_patch
from seq_align.components.c_kernels import (
    ACTIVE_C_KERNEL_CONTRACTS,
    LEGACY_C_KERNEL_CONTRACTS,
    dp_kernel,
    dp_kernel_neon,
)


def test_active_contract_registry_uses_neon_and_keeps_sse_for_replay() -> None:
    assert {
        "pipeline_flow",
        "jackhmmer_loop",
        "dp_kernel_neon",
        "ssv_filter_neon",
        "null2_correction_neon",
        "sequence_io",
    } == set(ACTIVE_C_KERNEL_CONTRACTS)
    assert {"dp_kernel", "ssv_filter", "null2_correction"} <= set(
        LEGACY_C_KERNEL_CONTRACTS
    )


@pytest.mark.requires_hmmer_source
def test_neon_probe_diff_is_accepted_only_by_neon_contract() -> None:
    diff = """--- a/src/impl_neon/fwdback.c
+++ b/src/impl_neon/fwdback.c
@@ -503,3 +503,3 @@ forward_engine(int do_full, const ESL_DSQ *dsq, int L, const P7_OPROFILE *om, P7_OMX *ox, float *opt_sc)
   if (opt_sc != NULL) *opt_sc = ox->totscale + log(xC * om->xf[p7O_C][p7O_MOVE]);
-  return eslOK;
+  return (eslOK);
 }
"""

    assert verify_c_patch(dp_kernel_neon, diff) == []
    legacy_errors = verify_c_patch(dp_kernel, diff)
    assert any("allows only" in error for error in legacy_errors)
