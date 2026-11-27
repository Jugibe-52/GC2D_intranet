# Affected parts

The canonical section 4.2 visualization notebook gains a public-data loading path and loses its requirement for a local scientific checkout in Colab. Its saved figure, section README and [study documentation](../../../docs/studies/method-timings.md) must describe the final behavior accurately. Preserve the existing statistics and original timing protocol.

The remote changes are a separate public B2 bucket containing only the selected timing data and publication manifest, and one notebook in the chosen Drive folder. The personal publication registry records that single selection and its confirmed identity; the maintained registry stays outside this repository. Necessary connection settings and their personal configuration record remain outside public artifacts.

Private archives, calculation notebooks, the other thesis notebooks, `notebooks/developements/`, simulation code and the thesis sources are outside this change. Checks cover byte hashes, recorded statistics, notebook execution without private dependencies, saved outputs, anonymous B2 downloads and actual Drive publication. Local success alone does not establish either remote publication or execution inside Colab.
