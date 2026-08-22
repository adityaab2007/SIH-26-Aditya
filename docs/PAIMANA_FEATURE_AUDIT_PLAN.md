# PAIMANA Feature Audit Plan

## Checks

- Feature availability and missingness report
- Leakage checks for outcome columns
- Validation that lifecycle trend features require snapshot history
- Verification that agency and sector history are built only from prior years

## Acceptance criteria

- No target columns are used as model inputs
- Missing lifecycle data is not replaced with synthetic values
- Temporal validation remains isolated by year
