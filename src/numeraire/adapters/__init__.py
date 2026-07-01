"""numeraire.adapters — thin wrappers making reference libraries conform to core protocols.

Glue, not spine — each adapter imports its reference library only at module top level, so installing
core alone never requires it. Reference-library methods (e.g. IPCA via ``ipca``) ship in extension
packages such as ``numeraire-zoo``, which declare those heavy dependencies themselves.
"""
