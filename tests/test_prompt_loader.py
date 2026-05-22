import tempfile
import unittest
from pathlib import Path

from app.prompt_loader import load_system_prompt


class LoadSystemPromptTests(unittest.TestCase):
    def test_loads_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "system_prompt.md"
            path.write_text("# Role\n\nTest prompt body.", encoding="utf-8")
            result = load_system_prompt(str(path))
            self.assertEqual(result, "# Role\n\nTest prompt body.")

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.md"
            with self.assertRaises(RuntimeError) as ctx:
                load_system_prompt(str(path))
            self.assertIn("not found", str(ctx.exception).lower())

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.md"
            path.write_text("   \n  ", encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                load_system_prompt(str(path))
            self.assertIn("empty", str(ctx.exception).lower())

    def test_invalid_utf8_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_bytes(b"\xff\xfe not utf8")
            with self.assertRaises(RuntimeError) as ctx:
                load_system_prompt(str(path))
            self.assertIn("utf-8", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
