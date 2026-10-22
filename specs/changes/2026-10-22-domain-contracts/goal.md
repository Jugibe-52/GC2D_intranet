# Contracts organized by domain

Existing interfaces are scattered between implementation modules and a flat
contracts package. Put existing pure contracts in one predictable hierarchy
that mirrors the responsibilities and subdirectories of `src`.

Preserve numerical behavior, contract names, validation, and supported public
facades. This is a relocation, not a redesign of data or algorithms.
