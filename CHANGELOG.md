# Changelog

## Unreleased

### Added

- Explicit `--tied-kv-pools N` measurement for two or more disjoint cache
  components with equal token capacity. Component byte costs are summed; memory
  is not deduplicated and the common token capacity is not multiplied by N.
- Ledger validation for `kv_pool_layout = "tied"` and a finite positive
  `kv_pool_bytes_per_token` array whose sum matches `kv_bytes_per_token`.
  Component count is independent of the number of serving roles.
- Synthetic regression coverage for one, two, three, five, and ten components,
  measurement-to-planner round trips, and invalid or inconsistent reports.

### Compatibility and limits

- Existing single-pool ledgers and the default measurement path remain
  compatible. `--tied-kv-pools 1` preserves that path. The
  `tied_target_draft` layout spelling remains accepted for exactly two
  components; new multi-component measurements emit `tied`.
- Measurement requires exactly N unambiguous component reports from one
  startup/accounting domain. Count mismatches, unequal capacities, malformed
  fields, and anonymous duplicate reports are refused with empty stdout.
- Fixed Mamba state remains supported. Proportional Mamba accounting with tied
  KV remains unsupported because its component-specific scaling base is unknown.
- Synthetic tests verify accounting, not backend support. A scalar runtime
  capacity report alone does not prove that all components obey the cap. The
  live admission check is unchanged; backend-specific qualification is still
  required before enrollment. See the [measurement limits](docs/measure-model-budget.md#multiple-kv-pools-composite-kv-allocation).
