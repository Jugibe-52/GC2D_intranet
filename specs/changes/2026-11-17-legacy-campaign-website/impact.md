# Scope and verification

The only hosted change is a new independent Cloudflare Pages project and deployment. Read the historical site to audit filenames, byte sizes, SHA256 hashes and scientific metadata, then compare the deployed HTML, metadata and every available binary asset with those saved bytes. Check the legacy address separately from the corrected production and pilot addresses.

There are no numerical-code, notebook, trajectory-integration or Backblaze-write changes. Keep the historical source directory, existing receipts and stored results intact. No Git index changes, commits or pushes are included; unrelated staged work must remain untouched.

After verification, add the legacy address and the presentation exception to [the campaign documentation](../../../docs/studies/main-reflected-campaign.md), its Spanish copy and a short existing-topic entry in `MEMORY.md` within its 300-line limit. Record observed deployment and verification evidence in this change's execution results. Full numerical tests are unnecessary because the saved simulation and viewer code are not modified.
