# Changelog

Notable changes of pflacco. Releases before 1.2.3 are described on the
[GitHub releases page](https://github.com/Reiyan/pflacco/releases).

## 1.2.3 (unreleased)

A fix release. No public argument is removed or renamed and the dependency
floors are those of 1.2.2, so an existing 1.2.2 environment works unchanged.
Some feature values do change, because they were wrong before, see
[Changed values](#changed-values).

### Fixed

- `calculate_ela_curvate`: `ela_curv.grad_norm.*` reported the gradient scale
  instead of the gradient norm, and the three derivative columns were mixed up
  before aggregation, so every `ela_curv.*` feature was affected. Reported by
  Shuhei Tanaka.
- #42 `create_initial_sample` with `sample_type = 'sobol'` ignored the `seed`
  argument and always used seed 50.
- #38 `calculate_information_content` raised a `KeyError` when identical
  decision vectors carried different objective values.
- #27 The cell mapping prerequisite checked `3 * dim^blocks <= n` instead of
  `3 * prod(blocks) <= n`. Valid block counts were rejected in low dimensions
  and grids with fewer than 3 observations per cell were accepted in higher
  ones.
- `calculate_length_scales_features`: with a seed set, the Levy random walk
  repeated the same step instead of walking. Pairs of identical points could
  also inject uninitialised values into the ratios and occasionally make the
  feature set raise.
- Compatibility with current dependencies: the cell mapping features failed
  with numpy 2 and current pandas, and the classical ELA features could not be
  imported with current scipy.

### Added

- #43 Optional objective normalization, after Prager and Trautmann (2023).
  `calculate_ela_meta`, `calculate_pca`, `calculate_information_content`,
  `calculate_limo`, `calculate_fitness_distance_correlation`,
  `calculate_ela_conv`, `calculate_ela_curvate` and `calculate_ela_local` take
  `normalize: bool = False`. Three `ela_local` features remain scale
  dependent, see its docstring.
- #45 `create_initial_sample` takes `categorical_values` for mixed continuous,
  integer and categorical search spaces (`sample_type = 'random'` only).
- #41 `CITATION.cff` for GitHub's "Cite this repository".

### Changed

- The upper bounds of the dependencies are removed, the floors stay as in
  1.2.2. `pyDOE` is capped at `<0.4`, because later releases ignore
  `np.random.seed`.
- #27 The block count warning is reworded to name the number of observations
  required. When pflacco chooses the block count itself (`blocks = None`), a
  violation now warns instead of raising.

### Changed values

Values computed with 1.2.2 are not comparable to 1.2.3 for:

- every `ela_curv.*` feature,
- `fla_metrics.sobol_indices.*` for every seed other than 50,
- the length scale features when a seed is set.
