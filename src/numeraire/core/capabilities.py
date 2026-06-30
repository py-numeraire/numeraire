"""Capability names — the open extractor set (SPEC §2.4).

Capabilities, not mandatory methods: a ``Model`` declares which of these it supports via
``Model.capabilities()`` and evaluators dispatch on them. This set is v0 and expected to
crystallize once the third real adapter (IPCA / VoC / a cross-sectional SDF method) lands.
Keep it a flat registry of string constants, not an enum — extensions may add their own.
"""

TO_WEIGHTS = "to_weights"
"""Produces a stream of portfolio weights (e.g. tangency / SDF / timing positions)."""

TO_FORECAST = "to_forecast"
"""Produces a conditional return forecast for the next horizon (predictive regressions)."""

TO_PRICING = "to_pricing"
"""Produces pricing output: factor loadings/parameters and pricing errors (alphas)."""

TO_DENSITY = "to_density"
"""(Future) produces a conditional return density."""

TO_SURFACE = "to_surface"
"""(Future) produces an option/implied surface."""

BUNDLED: frozenset[str] = frozenset({TO_WEIGHTS, TO_FORECAST, TO_PRICING, TO_DENSITY, TO_SURFACE})
"""Capability names shipped with core. Extensions may register additional names freely."""
