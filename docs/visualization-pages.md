# Publishing notebook visualizations with Cloudflare Pages

`visualization.pages` publishes existing standalone HTML exports. The URL path
preserves the experiment directory relative to `notebooks/`. For example:

```text
notebooks/developements/poincare_section/Poincare_comparacion_cuatro_ejecuciones/poincare_four_runs.html
    -> /developements/poincare_section/Poincare_comparacion_cuatro_ejecuciones/
```

The uploaded copy is named `index.html`. The notebook and its original HTML
keep their names. Each deployment includes a generated home page linking to
all selected visualizations. Browser interaction continues to use the embedded
data; publishing does not execute notebooks or numerical integrations.

## One-time setup

1. Install Node.js and the Cloudflare CLI: `npm install -g wrangler`.
2. Authenticate with `wrangler login`.
3. Create a dedicated Pages project for this catalog with
   `wrangler pages project create`, selecting `main` as its production branch.
4. Set `project_name`, `base_url`, and the actual production branch in
   `conf/visualization_pages.toml`. Copy the assigned HTTPS origin from
   Cloudflare; do not assume the requested name is the assigned domain.

Keep credentials in Wrangler's login session or its documented environment
variables; do not put them in notebooks or this configuration file.

## Prepare and publish

From the project environment and repository root:

```bash
python -m visualization.pages
python -m visualization.pages --deploy
```

The first command creates a local snapshot and prints prospective routes/URLs.
The second uploads the complete catalog and prints the configured production
URLs only after Wrangler succeeds. A failed upload raises an error. Prepared
snapshots live under ignored `build/cloudflare-pages/site-*`; they can be
deleted when no longer needed. The generated `index.html` can also be opened
locally to inspect the catalog before deploying.

Use a dedicated Pages project: each deployment replaces the full site snapshot.
Add another standalone HTML path to `html_files` to publish a new experiment.
The directory hierarchy is inferred automatically, including spelling and case.
Re-exporting and publishing updates the same URL. Changing a source directory
changes its URL. Removing an entry removes it from the next production snapshot.
Select one HTML per directory to avoid ambiguous directory URLs.

Only explicitly listed HTML files are copied. The helper supports
`notebooks/developements/` and rejects exports above the Pages limit of 25 MiB.
Use standalone exports with embedded data and scripts; sibling images, scripts,
and data files are not copied automatically. The Poincare comparison export
already satisfies this requirement.

## Publish from a notebook

After generating the HTML, use a dedicated publication cell:

```python
from visualization.pages import publish_pages_site

site = publish_pages_site()
for url in site.urls:
    print(url)
```

The helper discovers the project configuration from the current directory.
An explicit `config_path` is also supported. Running this cell uploads every
configured export, using each file's latest saved contents. To publish whenever
an export is generated, place the same call immediately after the export call.
Run publications sequentially to avoid older snapshots replacing newer ones.

Published HTML includes the displayed data. A normal Pages deployment is public;
private access requires separately configured Cloudflare Access protection.

References: [Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/),
[file limits](https://developers.cloudflare.com/pages/platform/limits/), and
[HTML route handling](https://developers.cloudflare.com/pages/configuration/serving-pages/).
