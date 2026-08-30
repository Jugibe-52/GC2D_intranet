"""Private implementation notes for the shared-time ABBA state extension.

The public API exposes this extension through
``state_extension="shared_time"`` on the five canonical ABBA methods. The
runtime implementation lives beside the common implicit and midpoint drivers;
this module intentionally defines no additional numerical-method class.
"""

__all__: list[str] = []
