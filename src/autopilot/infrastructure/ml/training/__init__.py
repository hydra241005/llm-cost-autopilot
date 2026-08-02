"""Dataset loading, training, and evaluation for the complexity classifier.

Import-heavy and scikit-learn dependent, so it is deliberately kept out of the
request path: the API imports :mod:`autopilot.infrastructure.ml.classifier`,
never this package.
"""
