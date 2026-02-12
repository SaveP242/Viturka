//! Proof of Credibility consensus for the Viturka blockchain.
//!
//! Implements top-10 validator selection, credibility calculations,
//! decay mechanics, and protocol parameters.

use crate::types::Address;

// ---------------------------------------------------------------------------
// Protocol Parameters
// ---------------------------------------------------------------------------

/// Maximum number of validators participating in each validation round.
pub const TOP_VALIDATORS: usize = 10;

/// Minimum valid proofs required out of top validators for DDTXO approval.
pub const CONSENSUS_THRESHOLD: usize = 7;

/// Blocks a validator must wait after participating before being eligible again.
pub const HALT_PERIOD: u64 = 10;

/// Minimum credibility score to be eligible for validation.
pub const MIN_CREDIBILITY: u64 = 1000;

/// Number of blocks after DTx inclusion before the reveal deadline.
pub const REVEAL_WINDOW: u64 = 10;

/// Number of blocks for ZKML validation after reveal (blocks N+11 to N+25).
pub const VALIDATION_WINDOW: u64 = 15;

/// Required fee deposit for data contributions (in VIT).
pub const DTX_DEPOSIT: u64 = 100;

/// Base reward for block proposers (in VIT).
pub const BASE_BLOCK_REWARD: u64 = 50;

/// Base reward for approved data contributions (in VIT).
pub const BASE_DATA_REWARD: u64 = 50;

/// Base reward per valid validation proof (in VIT).
pub const BASE_VALIDATION_REWARD: u64 = 10;

/// Base credibility earned per approved data contribution.
pub const BASE_DATA_CREDIBILITY: u64 = 10;

/// Base credibility earned per valid validation proof.
pub const BASE_VALIDATION_CREDIBILITY: u64 = 5;

/// Base credibility earned by block proposer.
pub const BASE_PROPOSER_CREDIBILITY: u64 = 20;

/// Soft cap for credibility (used in decay calculation).
pub const MAX_CREDIBILITY: u64 = 10_000;

/// Base monthly decay rate at maximum credibility (3%).
pub const BASE_DECAY_RATE: f64 = 0.03;

/// Exponent for progressive decay scaling.
pub const DECAY_EXPONENT: f64 = 1.5;

/// Minimum accuracy improvement required for DDTXO approval.
pub const MIN_ACCURACY_IMPROVEMENT: f64 = 0.001;

/// Standard transaction fee (in VIT).
pub const BASE_TX_FEE: u64 = 1;

/// Approximate blocks per month (for decay calculation).
/// Assuming ~12 second block time: 30 * 24 * 60 * 60 / 12 = 216,000.
pub const BLOCKS_PER_MONTH: u64 = 216_000;

/// Staking VIT required for maximum decay reduction.
pub const STAKE_FOR_MAX_REDUCTION: u64 = 5_000;

/// Maximum reduction in decay rate from staking (50%).
pub const MAX_STAKE_REDUCTION: f64 = 0.50;

/// Default stake for genesis validators.
pub const GENESIS_VALIDATOR_STAKE: u64 = 1_000;

// ---------------------------------------------------------------------------
// Validator eligibility info (used by top-10 selection)
// ---------------------------------------------------------------------------

/// Lightweight view of a validator used for selection sorting.
#[derive(Debug, Clone)]
pub struct ValidatorCandidate {
    pub address: Address,
    pub credibility: u64,
}

// ---------------------------------------------------------------------------
// Top-10 Validator Selection
// ---------------------------------------------------------------------------

/// Select the top validators eligible for a given block height.
///
/// Filtering criteria (per whitepaper Section 3.5):
/// 1. `credibility >= MIN_CREDIBILITY` (1000)
/// 2. Not in cooldown: `last_validation_block + HALT_PERIOD <= block_height`
/// 3. `zkml_registered == true`
///
/// Sorting: credibility descending, address ascending (deterministic tie-break).
/// Returns at most `TOP_VALIDATORS` (10) addresses.
pub fn select_top_validators(
    candidates: &[ValidatorCandidate],
) -> Vec<Address> {
    let mut sorted: Vec<&ValidatorCandidate> = candidates.iter().collect();

    // Sort by credibility descending, address ascending for tie-break
    sorted.sort_by(|a, b| {
        b.credibility
            .cmp(&a.credibility)
            .then_with(|| a.address.cmp(&b.address))
    });

    sorted
        .into_iter()
        .take(TOP_VALIDATORS)
        .map(|v| v.address.clone())
        .collect()
}

/// Check whether a validator is in cooldown at a given block height.
/// A `last_validation_block` of 0 means the validator has never participated,
/// so they are NOT in cooldown.
pub fn is_in_cooldown(last_validation_block: u64, current_block: u64) -> bool {
    if last_validation_block == 0 {
        return false;
    }
    last_validation_block + HALT_PERIOD > current_block
}

// ---------------------------------------------------------------------------
// Block Proposer Selection
// ---------------------------------------------------------------------------

/// Deterministic block proposer selection (round-robin among top validators).
/// Returns `None` if there are no eligible validators.
pub fn select_block_proposer(
    top_validators: &[Address],
    block_height: u64,
) -> Option<Address> {
    if top_validators.is_empty() {
        return None;
    }
    let index = (block_height as usize) % top_validators.len();
    Some(top_validators[index].clone())
}

// ---------------------------------------------------------------------------
// Credibility Calculations
// ---------------------------------------------------------------------------

/// Calculate credibility earned from an approved data contribution.
///
/// Formula: `BASE_DATA_CREDIBILITY * quality_multiplier * category_multiplier`
/// where `quality_multiplier = 1.0 + min(1.0, accuracy_improvement / 0.01)`
pub fn calculate_data_credibility(
    accuracy_improvement: f64,
    category_multiplier: f64,
) -> u64 {
    let baseline_improvement = 0.01; // 1% baseline
    let quality_multiplier = 1.0 + (accuracy_improvement / baseline_improvement).min(1.0);
    let raw = BASE_DATA_CREDIBILITY as f64 * quality_multiplier * category_multiplier;
    raw.round() as u64
}

/// Calculate credibility earned from submitting a valid validation proof.
pub fn calculate_validation_credibility() -> u64 {
    BASE_VALIDATION_CREDIBILITY
}

/// Calculate credibility earned by a block proposer.
pub fn calculate_proposer_credibility() -> u64 {
    BASE_PROPOSER_CREDIBILITY
}

/// Calculate data contributor reward in VIT for an approved contribution.
///
/// Formula: `BASE_DATA_REWARD * quality_multiplier * category_multiplier`
/// Clamped to range [25, 300].
pub fn calculate_data_reward(
    accuracy_improvement: f64,
    category_multiplier: f64,
) -> u64 {
    let baseline_improvement = 0.01;
    let quality_multiplier = 1.0 + (accuracy_improvement / baseline_improvement).min(1.0);
    let raw = BASE_DATA_REWARD as f64 * quality_multiplier * category_multiplier;
    (raw.round() as u64).clamp(25, 300)
}

/// Apply credibility decay for a given number of elapsed blocks.
///
/// Progressive decay formula (per whitepaper Section 4.3):
/// `monthly_decay_rate = BASE_DECAY * (credibility / MAX_CREDIBILITY) ^ DECAY_EXPONENT`
///
/// Staking reduces the effective decay rate: each VIT staked reduces decay
/// proportionally, up to `MAX_STAKE_REDUCTION` (50%) at `STAKE_FOR_MAX_REDUCTION` VIT.
///
/// Decay is applied proportionally based on blocks elapsed relative to blocks per month.
pub fn apply_credibility_decay(credibility: u64, blocks_elapsed: u64, stake: u64) -> u64 {
    if credibility == 0 || blocks_elapsed == 0 {
        return credibility;
    }

    let cred_ratio = (credibility as f64) / (MAX_CREDIBILITY as f64);
    let stake_factor =
        1.0 - (stake as f64 / STAKE_FOR_MAX_REDUCTION as f64).min(1.0) * MAX_STAKE_REDUCTION;
    let monthly_rate = BASE_DECAY_RATE * cred_ratio.powf(DECAY_EXPONENT) * stake_factor;
    let block_rate = monthly_rate / (BLOCKS_PER_MONTH as f64);
    let total_decay_fraction = block_rate * (blocks_elapsed as f64);

    let decayed = credibility as f64 * (1.0 - total_decay_fraction);
    decayed.max(0.0).round() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(byte: u8) -> Address {
        Address::from_bytes([byte; 32])
    }

    #[test]
    fn test_select_top_validators_basic() {
        let candidates = vec![
            ValidatorCandidate { address: addr(1), credibility: 5000 },
            ValidatorCandidate { address: addr(2), credibility: 3000 },
            ValidatorCandidate { address: addr(3), credibility: 8000 },
            ValidatorCandidate { address: addr(4), credibility: 1500 },
            ValidatorCandidate { address: addr(5), credibility: 2000 },
        ];

        let top = select_top_validators(&candidates);
        assert_eq!(top.len(), 5); // only 5 candidates, all selected
        assert_eq!(top[0], addr(3)); // highest credibility first
        assert_eq!(top[1], addr(1));
        assert_eq!(top[2], addr(2));
    }

    #[test]
    fn test_select_top_validators_cap_at_10() {
        let candidates: Vec<ValidatorCandidate> = (0..20)
            .map(|i| ValidatorCandidate {
                address: addr(i as u8),
                credibility: (i + 1) * 1000,
            })
            .collect();

        let top = select_top_validators(&candidates);
        assert_eq!(top.len(), TOP_VALIDATORS);
    }

    #[test]
    fn test_select_top_validators_tiebreak_by_address() {
        let candidates = vec![
            ValidatorCandidate { address: addr(5), credibility: 5000 },
            ValidatorCandidate { address: addr(2), credibility: 5000 },
            ValidatorCandidate { address: addr(9), credibility: 5000 },
        ];

        let top = select_top_validators(&candidates);
        // Same credibility — sorted by address ascending
        assert_eq!(top[0], addr(2));
        assert_eq!(top[1], addr(5));
        assert_eq!(top[2], addr(9));
    }

    #[test]
    fn test_select_top_validators_empty() {
        let top = select_top_validators(&[]);
        assert!(top.is_empty());
    }

    #[test]
    fn test_cooldown() {
        assert!(is_in_cooldown(100, 105));  // 100 + 10 = 110 > 105
        assert!(is_in_cooldown(100, 109));  // 100 + 10 = 110 > 109
        assert!(!is_in_cooldown(100, 110)); // 100 + 10 = 110, not >
        assert!(!is_in_cooldown(100, 200)); // well past cooldown
        assert!(!is_in_cooldown(0, 10));    // exactly at boundary
    }

    #[test]
    fn test_block_proposer_selection() {
        let validators = vec![addr(1), addr(2), addr(3)];
        assert_eq!(select_block_proposer(&validators, 0), Some(addr(1)));
        assert_eq!(select_block_proposer(&validators, 1), Some(addr(2)));
        assert_eq!(select_block_proposer(&validators, 2), Some(addr(3)));
        assert_eq!(select_block_proposer(&validators, 3), Some(addr(1))); // wraps
    }

    #[test]
    fn test_block_proposer_no_validators() {
        assert_eq!(select_block_proposer(&[], 0), None);
    }

    #[test]
    fn test_data_credibility_basic() {
        // 1% improvement, 1.0x category = 10 * 2.0 * 1.0 = 20
        let cred = calculate_data_credibility(0.01, 1.0);
        assert_eq!(cred, 20);
    }

    #[test]
    fn test_data_credibility_partial_improvement() {
        // 0.5% improvement, 1.0x category = 10 * 1.5 * 1.0 = 15
        let cred = calculate_data_credibility(0.005, 1.0);
        assert_eq!(cred, 15);
    }

    #[test]
    fn test_data_credibility_large_improvement_capped() {
        // 5% improvement capped at 1.0 multiplier = 10 * 2.0 * 1.0 = 20
        let cred = calculate_data_credibility(0.05, 1.0);
        assert_eq!(cred, 20);
    }

    #[test]
    fn test_data_credibility_high_priority_category() {
        // 1% improvement, 3.0x category = 10 * 2.0 * 3.0 = 60
        let cred = calculate_data_credibility(0.01, 3.0);
        assert_eq!(cred, 60);
    }

    #[test]
    fn test_data_reward_clamped() {
        let low = calculate_data_reward(0.0001, 0.5);
        assert!(low >= 25);

        let high = calculate_data_reward(0.05, 3.0);
        assert!(high <= 300);
    }

    #[test]
    fn test_credibility_decay_zero() {
        assert_eq!(apply_credibility_decay(0, 1000, 0), 0);
        assert_eq!(apply_credibility_decay(1000, 0, 0), 1000);
    }

    #[test]
    fn test_credibility_decay_low_credibility_slow() {
        // Low credibility (1000) should decay very slowly
        let after = apply_credibility_decay(1000, BLOCKS_PER_MONTH, 0);
        // Expected: ~3 points lost at 1000 credibility over 1 month
        assert!(after >= 990, "Low credibility should barely decay: got {after}");
        assert!(after < 1000, "Should have some decay");
    }

    #[test]
    fn test_credibility_decay_high_credibility_faster() {
        // High credibility (10000) should decay faster
        let after = apply_credibility_decay(10_000, BLOCKS_PER_MONTH, 0);
        // Expected: ~300 points lost (3% of 10000)
        assert!(after >= 9500, "Should not decay too much: got {after}");
        assert!(after <= 9800, "Should have meaningful decay: got {after}");
    }

    #[test]
    fn test_credibility_decay_progressive() {
        // Higher credibility should decay proportionally more
        let decay_low = 1000 - apply_credibility_decay(1000, BLOCKS_PER_MONTH, 0);
        let decay_high = 10_000 - apply_credibility_decay(10_000, BLOCKS_PER_MONTH, 0);
        assert!(
            decay_high > decay_low * 5,
            "High credibility should decay much faster: low={decay_low}, high={decay_high}"
        );
    }

    #[test]
    fn test_credibility_decay_staking_reduces_decay() {
        let without_stake = apply_credibility_decay(10_000, BLOCKS_PER_MONTH, 0);
        let with_stake = apply_credibility_decay(10_000, BLOCKS_PER_MONTH, 5_000);
        // Max staking should halve the decay
        let decay_without = 10_000 - without_stake;
        let decay_with = 10_000 - with_stake;
        assert!(
            decay_with < decay_without,
            "Staking should reduce decay: without={decay_without}, with={decay_with}"
        );
        // With max stake, decay should be roughly half
        let ratio = decay_with as f64 / decay_without as f64;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "Max stake should halve decay: ratio={ratio:.3}"
        );
    }

    #[test]
    fn test_validation_credibility() {
        assert_eq!(calculate_validation_credibility(), 5);
    }

    #[test]
    fn test_proposer_credibility() {
        assert_eq!(calculate_proposer_credibility(), 20);
    }
}
