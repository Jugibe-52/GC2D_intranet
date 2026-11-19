# Affected parts

The shared static viewer in `src/visualization/main_campaign.py` gains image export and an HTML-only refresh path so existing verified assets need not be regenerated. Browser checks exercise copying, zoom, selection and clipboard failure. Study documentation and project memory describe the new control.

Deployment targets only the reference campaign's existing Cloudflare Pages project. Compare all non-HTML site files before and after the refresh and verify the published HTML. No numerical integration, archive migration or thesis PDF change is involved.
