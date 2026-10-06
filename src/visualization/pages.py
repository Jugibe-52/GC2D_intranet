"""Prepare and publish standalone notebook HTML files with Cloudflare Pages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path
from collections.abc import Mapping
from typing import Any
import shutil
import subprocess
import tempfile
import tomllib
from urllib.parse import quote, urlsplit

from diagnostics.paths import find_project_root


__all__ = ["PagesSite", "prepare_pages_site", "publish_pages_site"]

_MAX_FILE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class PagesSite:
    """A complete static snapshot and its prospective production URLs.

    ``directory`` contains only selected HTML exports and a generated catalog.
    Preparing a snapshot does not make ``urls`` available on the Internet.
    """

    directory: Path
    routes: tuple[str, ...]
    urls: tuple[str, ...]
    project_name: str
    production_branch: str


def _validated_pages_config(config: Mapping[str, Any]) -> tuple[str, str, str, list[Any]]:
    """Normalize publication settings and check the configured export list."""
    project_name = str(config.get("project_name", "")).strip()
    production_branch = str(config.get("production_branch", "main")).strip()
    base_url = str(config.get("base_url", "")).strip().rstrip("/")
    if not production_branch:
        raise ValueError("production_branch must not be empty.")
    if base_url:
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.path
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("base_url must be an HTTPS site origin without a path.")

    exports = config.get("html_files")
    if not isinstance(exports, list) or not exports:
        raise ValueError("html_files must list at least one standalone HTML export.")
    return project_name, production_branch, base_url, exports


def _validated_export_sources(root: Path, exports: list[Any]) -> dict[Path, Path]:
    """Resolve permitted HTML sources and unique routes before creating a snapshot."""
    notebooks = root / "notebooks"
    development_root = notebooks / "developements"
    sources: dict[Path, Path] = {}
    for entry in exports:
        if not isinstance(entry, str) or Path(entry).is_absolute():
            raise ValueError("html_files entries must be paths relative to the project root.")
        source = (root / entry).resolve()
        if not source.is_relative_to(development_root):
            raise ValueError("HTML exports must be inside notebooks/developements/.")
        if source.suffix.lower() != ".html" or not source.is_file():
            raise ValueError(f"Standalone HTML export not found: {source}")
        if source.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError(f"HTML export exceeds the Pages limit of 25 MiB: {source}")
        route = source.relative_to(notebooks).parent
        if route in sources:
            raise ValueError(f"Select only one HTML export per directory: {route}")
        sources[route] = source
    return sources


def prepare_pages_site(config_path: str | Path | None = None) -> PagesSite:
    """Build all configured exports, preserving their paths below notebooks/.

    Each selected HTML becomes ``<experiment-directory>/index.html``. Select
    one standalone HTML per directory. Paths in the configuration are relative
    to the project root; only development exports are supported. All sources
    are checked before a new snapshot is created under ignored ``build/``.
    """
    config_file = (
        find_project_root(Path.cwd()) / "conf" / "visualization_pages.toml"
        if config_path is None
        else Path(config_path).expanduser().resolve()
    )
    root = find_project_root(config_file)
    with config_file.open("rb") as stream:
        config = tomllib.load(stream)

    project_name, production_branch, base_url, exports = _validated_pages_config(config)

    sources = _validated_export_sources(root, exports)

    # Rebuild the complete catalog on every deployment: a Pages deployment
    # replaces the site's snapshot, so uploading just one page loses others.
    build_root = root / "build" / "cloudflare-pages"
    build_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="site-", dir=build_root))
    routes: list[str] = []
    links: list[str] = []
    try:
        for relative, source in sorted(sources.items()):
            target = directory / relative / "index.html"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            route = "/" + quote(relative.as_posix(), safe="/") + "/"
            routes.append(route)
            # Relative links also work when the prepared catalog is opened offline.
            links.append(
                f'<li><a href="{escape(route.lstrip("/"), quote=True)}index.html">'
                f'{escape(relative.as_posix())}</a></li>'
            )
        (directory / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>GC2D visualizations</title>'
            '<style>body{font:18px system-ui;max-width:72rem;margin:3rem auto;'
            'padding:0 1rem}li{margin:1rem 0;overflow-wrap:anywhere}</style>'
            '<h1>GC2D visualizations</h1><ul>' + "".join(links) + '</ul></html>',
            encoding="utf-8",
        )
        (directory / "404.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Not found</title><h1>Visualization not found</h1></html>',
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(directory)
        raise

    return PagesSite(
        directory=directory,
        routes=tuple(routes),
        urls=tuple(base_url + route for route in routes) if base_url else (),
        project_name=project_name,
        production_branch=production_branch,
    )


def publish_pages_site(config_path: str | Path | None = None) -> PagesSite:
    """Upload every configured visualization using an authenticated Wrangler.

    Configure the existing Pages project and its actual production origin once.
    This is an explicit network operation; failed uploads raise an exception.
    Each call rebuilds the full configured set using the latest HTML exports.
    """
    executable = shutil.which("wrangler")
    if executable is None:
        raise RuntimeError("Install Wrangler and run 'wrangler login' before publishing.")
    site = prepare_pages_site(config_path)
    if not site.project_name or not site.urls:
        raise ValueError("Set project_name and base_url in visualization_pages.toml before publishing.")
    subprocess.run(
        [executable, "pages", "deploy", str(site.directory),
         "--project-name", site.project_name, "--branch", site.production_branch],
        cwd=site.directory,
        check=True,
    )
    return site


def _main() -> None:
    """Prepare locally by default; publish only with the explicit deploy flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Pages TOML configuration path.")
    parser.add_argument("--deploy", action="store_true", help="Publish the full configured site.")
    arguments = parser.parse_args()
    site = (publish_pages_site if arguments.deploy else prepare_pages_site)(arguments.config)
    print(f"Prepared site: {site.directory}")
    label = "Published URL" if arguments.deploy else "Planned URL"
    for url in site.urls:
        print(f"{label}: {url}")
    if not site.urls:
        for route in site.routes:
            print(f"Prepared route: {route}")
        print("Set project_name and base_url in the configuration to enable publishing.")


if __name__ == "__main__":
    _main()
