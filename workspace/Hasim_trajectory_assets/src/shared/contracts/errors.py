"""Errors raised by scenario assets and package integration."""


class ScenarioError(ValueError):
    """Base class for invalid scenario integration data."""


class CatalogError(ScenarioError):
    """An asset catalog is invalid or cannot satisfy a query."""


class AssetError(ScenarioError):
    """An asset payload is malformed or cannot be bound."""


class GroundError(ScenarioError):
    """Ground geometry is missing, malformed or too sparse for placement."""


class PackageError(ScenarioError):
    """A ScenarioPackage violates its schema."""
