"""Check publication routes, complete snapshots, and failure reporting."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from visualization.pages import prepare_pages_site, publish_pages_site


class VisualizationPagesTests(unittest.TestCase):
    """Exercise publication with local fixtures and no external deployment."""

    def setUp(self) -> None:
        self.workspace = TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        (self.root / "pyproject.toml").touch()
        (self.root / "conf").mkdir()
        self.config = self.root / "conf" / "visualization_pages.toml"

    def configure(self, *paths: str) -> None:
        """Write explicit source selections for a dedicated test project."""
        entries = ", ".join(f'"{path}"' for path in paths)
        self.config.write_text(
            'project_name = "test-visualizations"\n'
            'base_url = "https://test-visualizations.pages.dev"\n'
            'production_branch = "production"\n'
            f'html_files = [{entries}]\n', encoding="utf-8",
        )

    def export(self, relative: str, content: str = "<html>test</html>") -> Path:
        """Create a small standalone HTML export."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_routes_preserve_hierarchy_and_each_snapshot_includes_every_export(self) -> None:
        first = "notebooks/developements/poincare_section/Study A & B/view.html"
        second = "notebooks/developements/other_study/plot.html"
        source = self.export(first)
        self.export(second)
        source.with_name("private.csv").write_text("excluded", encoding="utf-8")
        self.configure(first, second)

        initial = prepare_pages_site(self.config)
        source.write_text("<html>updated</html>", encoding="utf-8")
        updated = prepare_pages_site(self.config)

        relative = Path("developements/poincare_section/Study A & B/index.html")
        self.assertEqual((initial.directory / relative).read_text(), "<html>test</html>")
        self.assertEqual((updated.directory / relative).read_text(), "<html>updated</html>")
        self.assertTrue((updated.directory / "developements/other_study/index.html").is_file())
        self.assertEqual(len(list(updated.directory.rglob("*.html"))), 4)
        self.assertFalse(list(updated.directory.rglob("*.csv")))
        self.assertIn(
            "https://test-visualizations.pages.dev/developements/poincare_section/Study%20A%20%26%20B/",
            updated.urls,
        )
        self.assertIn("Study A &amp; B", (updated.directory / "index.html").read_text())

    def test_rejects_out_of_scope_sources_and_directory_collisions(self) -> None:
        self.configure("notebooks/experiments/untouched/view.html")
        with self.assertRaisesRegex(ValueError, "inside notebooks/developements"):
            prepare_pages_site(self.config)
        first = "notebooks/developements/study/first.html"
        second = "notebooks/developements/study/second.html"
        self.export(first)
        self.export(second)
        self.configure(first, second)
        with self.assertRaisesRegex(ValueError, "one HTML export per directory"):
            prepare_pages_site(self.config)
        self.assertFalse((self.root / "build").exists())

    def test_rejects_oversized_export_before_building(self) -> None:
        relative = "notebooks/developements/study/large.html"
        source = self.export(relative)
        with source.open("ab") as stream:
            stream.truncate(25 * 1024 * 1024 + 1)
        self.configure(relative)
        with self.assertRaisesRegex(ValueError, "25 MiB"):
            prepare_pages_site(self.config)
        self.assertFalse((self.root / "build").exists())

    def test_deploy_targets_configured_branch_and_reports_upload_failure(self) -> None:
        relative = "notebooks/developements/study/view.html"
        self.export(relative)
        self.configure(relative)
        with (
            patch("visualization.pages.shutil.which", return_value="/bin/wrangler"),
            patch("visualization.pages.subprocess.run") as run,
        ):
            site = publish_pages_site(self.config)
            run.assert_called_once_with(
                ["/bin/wrangler", "pages", "deploy", str(site.directory),
                 "--project-name", "test-visualizations", "--branch", "production"],
                cwd=site.directory, check=True,
            )
            run.side_effect = subprocess.CalledProcessError(1, "wrangler")
            with self.assertRaises(subprocess.CalledProcessError):
                publish_pages_site(self.config)


if __name__ == "__main__":
    unittest.main()
