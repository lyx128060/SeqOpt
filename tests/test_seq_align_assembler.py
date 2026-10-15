from __future__ import annotations

from seq_align.assembler import _all_function_ranges, extract_actual_diff_footprint


def test_all_function_ranges_only_reports_file_scope_definitions() -> None:
    lines = """
static int
real_function(const double value)
{
  if (isfinite(value))
  {
    return 1;
  }
  switch ((int) value)
  {
    case 0: return 0;
    default: break;
  }
  return 2;
}

int prototype_only(int value);
static int (*callback)(int value);
""".splitlines()

    ranges = _all_function_ranges(lines)

    assert list(ranges) == ["real_function"]
    assert not ({"if", "switch", "isfinite"} & set(ranges))


def test_all_function_ranges_handles_multiline_signature_and_control_headers() -> None:
    lines = """
static float *
p7_example(
    const float *values,
    int count)
{
  for (int idx = 0;
       idx < count;
       idx++)
  {
    while (values[idx] > 0.0f)
    {
      break;
    }
  }
  return 0;
}
""".splitlines()

    ranges = _all_function_ranges(lines)

    assert list(ranges) == ["p7_example"]
    start, end = ranges["p7_example"]
    assert start == 2
    assert end == len(lines)


def test_actual_footprint_attributes_control_body_to_enclosing_function(tmp_path) -> None:
    source = """int
p7_example(int value)
{
  if (isfinite(value))
  {
    return value;
  }
  return 0;
}
"""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "example.c").write_text(source, encoding="utf-8")
    diff = """diff --git a/src/example.c b/src/example.c
--- a/src/example.c
+++ b/src/example.c
@@ -3,7 +3,7 @@ p7_example(int value)
 {
   if (isfinite(value))
   {
-    return value;
+    return value + 1;
   }
   return 0;
 }
"""

    footprint = extract_actual_diff_footprint(
        diff,
        "example.c",
        hmmer_src_root=tmp_path,
    )

    assert footprint.extraction_status == "ok"
    assert footprint.actual_functions == ["p7_example"]
