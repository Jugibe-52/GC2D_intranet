# HTML viewer delivery

Deliver HTML viewers using a clickable Markdown link to an active local HTTP
URL, such as `[Open the viewer](http://127.0.0.1:<port>/<experiment_name>.html)`.
Use a descriptive experiment or notebook name in the URL. Do not make the user
copy and paste a filesystem path to open the viewer.

For notebook viewers, use
`visualization.poincare_probe.display_notebook_viewer_link` and display the
returned URL in the notebook. Verify that the URL responds and keep its local
server available when delivering the link. Reuse saved results; do not rerun
the numerical calculation just to provide or refresh a viewer link.
