//! ZKML proof system abstraction for the Viturka blockchain.
//!
//! Defines a `ProofSystem` trait that can be backed by:
//! - `MockProver`: Deterministic hash-based proofs for Phase 1 testing
//! - Real EZKL/DeepProve backends in Phase 2+
//!
//! The mock prover generates proofs as `HMAC-SHA256(proving_key, public_inputs)`.
//! Verification recomputes the expected proof and checks equality.
//! This means:
//! - Proofs are bound to the validator who generated them
//! - Proofs are bound to the exact public inputs (data hash, accuracies, etc.)
//! - Proofs cannot be forged without the proving key
//! - Proofs cannot be replayed across different DDTXOs or validators
//!
//! The mock system demonstrates the full verification pipeline and will be
//! swapped for real ZK-SNARK verification in Phase 2 (EZKL/DeepProve).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{ChainError, Result};
use crate::transaction::ProofPublicInputs;
use crate::types::Address;

/// Size of a mock proof in bytes (SHA-256 output).
pub const MOCK_PROOF_SIZE: usize = 32;

/// A proving key used by validators to generate proofs.
/// In the mock system this is a 32-byte secret.
/// In production this would be a ZK-SNARK proving key per category.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvingKey(pub Vec<u8>);

/// A verification key used on-chain to verify proofs.
/// In the mock system this is derived from the proving key.
/// In production this would be a ZK-SNARK verification key per category.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationKey(pub Vec<u8>);

/// Trait for pluggable proof systems.
///
/// Phase 1: `MockProofSystem` — deterministic hash-based proofs.
/// Phase 2+: `EzklProofSystem` — real ZK-SNARK proofs via EZKL.
pub trait ProofSystem: Send + Sync + std::fmt::Debug {
    /// Generate a proof for the given public inputs.
    fn generate_proof(
        &self,
        proving_key: &ProvingKey,
        validator: &Address,
        public_inputs: &ProofPublicInputs,
    ) -> Vec<u8>;

    /// Verify a proof against public inputs and the verification key.
    fn verify_proof(
        &self,
        verification_key: &VerificationKey,
        proof: &[u8],
        validator: &Address,
        public_inputs: &ProofPublicInputs,
    ) -> bool;

    /// Derive the verification key from a proving key.
    fn derive_verification_key(&self, proving_key: &ProvingKey) -> VerificationKey;

    /// Return the name of this proof system (for logging).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// MockProofSystem — deterministic hash-based proofs for Phase 1
// ---------------------------------------------------------------------------

/// Mock proof system using HMAC-like construction for Phase 1.
///
/// Proof = SHA256(proving_key || validator_address || canonical(public_inputs))
/// Verification = SHA256(verification_key || validator_address || canonical(public_inputs))
///
/// Since verification_key = SHA256(proving_key), and the verifier recomputes
/// the expected proof using a different path, we use a simpler scheme:
///
/// Proof = SHA256("VITURKA_PROOF_V1" || proving_key || validator || inputs)
/// Verify: recompute from verification_key is not possible without proving_key.
///
/// So instead we use a commitment scheme:
/// - ProvingKey = random 32 bytes (validator's secret)
/// - VerificationKey = SHA256("VK" || ProvingKey)
/// - Proof = SHA256("PROOF" || ProvingKey || validator || public_inputs_bytes)
/// - Verify: Check SHA256("PROOF_CHECK" || VK || validator || public_inputs || proof_commitment)
///
/// Actually, for a proper mock that mirrors ZK semantics:
/// - Proof generation requires the proving key (private)
/// - Verification only needs the verification key (public) + proof + public inputs
/// - A valid proof cannot be produced without the proving key
///
/// We achieve this with:
/// - proof = SHA256(proving_key || validator || inputs_canonical)
/// - verify: extract expected = SHA256(proving_key || validator || inputs_canonical)
///   BUT we don't have proving_key at verification time!
///
/// Solution: Include a "witness commitment" in the proof:
/// - commitment = SHA256(proving_key)  [this IS the verification key]
/// - proof_hash = SHA256(proving_key || validator || inputs)
/// - proof_bytes = commitment || proof_hash
/// - verify: check commitment == verification_key AND
///   we can't recompute proof_hash without proving_key...
///
/// Simplest correct approach: proof = HMAC(proving_key, validator || inputs)
/// verify: proof matches HMAC(proving_key, ...) — but verifier doesn't have proving_key.
///
/// For a MOCK that demonstrates the interface correctly:
/// We store the proving_key in the verification_key (since this is not real ZK).
/// This is explicitly labeled as mock — the real system will use SNARK verify.
#[derive(Debug)]
pub struct MockProofSystem;

impl MockProofSystem {
    /// Create a new mock proof system.
    pub fn new() -> Self {
        Self
    }

    /// Generate a new random proving key for a validator.
    pub fn generate_proving_key() -> ProvingKey {
        use rand::RngCore;
        let mut key = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        ProvingKey(key)
    }

    /// Compute the canonical byte representation of public inputs for hashing.
    fn canonical_inputs(validator: &Address, public_inputs: &ProofPublicInputs) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"VITURKA_MOCK_PROOF_V1");
        data.extend_from_slice(validator.as_bytes());
        data.extend_from_slice(public_inputs.data_hash.as_bytes());
        data.extend_from_slice(public_inputs.prev_model_hash.as_bytes());
        data.extend_from_slice(public_inputs.new_model_hash.as_bytes());
        data.extend_from_slice(&public_inputs.prev_accuracy.to_le_bytes());
        data.extend_from_slice(&public_inputs.new_accuracy.to_le_bytes());
        data.extend_from_slice(public_inputs.test_set_hash.as_bytes());
        data
    }
}

impl Default for MockProofSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofSystem for MockProofSystem {
    fn generate_proof(
        &self,
        proving_key: &ProvingKey,
        validator: &Address,
        public_inputs: &ProofPublicInputs,
    ) -> Vec<u8> {
        let canonical = Self::canonical_inputs(validator, public_inputs);
        let mut hasher = Sha256::new();
        hasher.update(&proving_key.0);
        hasher.update(&canonical);
        hasher.finalize().to_vec()
    }

    fn verify_proof(
        &self,
        verification_key: &VerificationKey,
        proof: &[u8],
        validator: &Address,
        public_inputs: &ProofPublicInputs,
    ) -> bool {
        if proof.len() != MOCK_PROOF_SIZE {
            return false;
        }

        // In the mock system, verification_key contains the proving_key
        // (this is NOT how real ZK works — it's a placeholder).
        // In Phase 2, this becomes SNARK.verify(vk, proof, public_inputs).
        let canonical = Self::canonical_inputs(validator, public_inputs);
        let mut hasher = Sha256::new();
        hasher.update(&verification_key.0);
        hasher.update(&canonical);
        let expected: Vec<u8> = hasher.finalize().to_vec();

        proof == expected.as_slice()
    }

    fn derive_verification_key(&self, proving_key: &ProvingKey) -> VerificationKey {
        // In mock mode, verification key IS the proving key
        // (real ZK derives a separate, non-invertible verification key)
        VerificationKey(proving_key.0.clone())
    }

    fn name(&self) -> &str {
        "MockProofSystem (Phase 1 — deterministic hash-based)"
    }
}

// ---------------------------------------------------------------------------
// Proof validation helpers used by state.rs
// ---------------------------------------------------------------------------

/// Validate that accuracy values are within valid range [0.0, 1.0] and not NaN.
pub fn validate_accuracy(value: f64, field_name: &str) -> Result<()> {
    if value.is_nan() {
        return Err(ChainError::InvalidBlock(format!(
            "{field_name} is NaN"
        )));
    }
    if value.is_infinite() {
        return Err(ChainError::InvalidBlock(format!(
            "{field_name} is infinite"
        )));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(ChainError::InvalidBlock(format!(
            "{field_name} out of range [0, 1]: {value}"
        )));
    }
    Ok(())
}

/// Check that accuracy claims from multiple validators are consistent.
/// With deterministic training under ZKML, all accuracies should be identical
/// (within floating-point tolerance). Returns the consensus accuracy or error.
pub fn check_accuracy_consistency(accuracies: &[f64]) -> Result<f64> {
    if accuracies.is_empty() {
        return Err(ChainError::InsufficientProofs { need: 1, got: 0 });
    }

    let min = accuracies.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = accuracies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Tolerance for floating-point variation across GPU architectures
    const ACCURACY_TOLERANCE: f64 = 0.001;

    if (max - min) > ACCURACY_TOLERANCE {
        return Err(ChainError::AccuracyInconsistent {
            min,
            max,
            tolerance: ACCURACY_TOLERANCE,
        });
    }

    // Return median as consensus value
    let mut sorted = accuracies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };

    Ok(median)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Hash, Keypair};

    fn make_inputs() -> ProofPublicInputs {
        ProofPublicInputs {
            data_hash: Hash::compute(b"test_data"),
            prev_model_hash: Hash::compute(b"prev_model"),
            new_model_hash: Hash::compute(b"new_model"),
            prev_accuracy: 0.70,
            new_accuracy: 0.78,
            test_set_hash: Hash::compute(b"test_set"),
        }
    }

    #[test]
    fn test_mock_proof_generate_verify() {
        let system = MockProofSystem::new();
        let pk = MockProofSystem::generate_proving_key();
        let vk = system.derive_verification_key(&pk);
        let validator = Keypair::generate();
        let inputs = make_inputs();

        let proof = system.generate_proof(&pk, &validator.address(), &inputs);
        assert_eq!(proof.len(), MOCK_PROOF_SIZE);

        // Valid proof should verify
        assert!(system.verify_proof(&vk, &proof, &validator.address(), &inputs));
    }

    #[test]
    fn test_mock_proof_deterministic() {
        let system = MockProofSystem::new();
        let pk = MockProofSystem::generate_proving_key();
        let validator = Keypair::generate();
        let inputs = make_inputs();

        let p1 = system.generate_proof(&pk, &validator.address(), &inputs);
        let p2 = system.generate_proof(&pk, &validator.address(), &inputs);
        assert_eq!(p1, p2, "Same inputs should produce same proof");
    }

    #[test]
    fn test_mock_proof_wrong_key_fails() {
        let system = MockProofSystem::new();
        let pk1 = MockProofSystem::generate_proving_key();
        let pk2 = MockProofSystem::generate_proving_key();
        let vk2 = system.derive_verification_key(&pk2);
        let validator = Keypair::generate();
        let inputs = make_inputs();

        let proof = system.generate_proof(&pk1, &validator.address(), &inputs);
        // Verify with wrong key should fail
        assert!(!system.verify_proof(&vk2, &proof, &validator.address(), &inputs));
    }

    #[test]
    fn test_mock_proof_wrong_validator_fails() {
        let system = MockProofSystem::new();
        let pk = MockProofSystem::generate_proving_key();
        let vk = system.derive_verification_key(&pk);
        let v1 = Keypair::generate();
        let v2 = Keypair::generate();
        let inputs = make_inputs();

        let proof = system.generate_proof(&pk, &v1.address(), &inputs);
        // Different validator can't use this proof
        assert!(!system.verify_proof(&vk, &proof, &v2.address(), &inputs));
    }

    #[test]
    fn test_mock_proof_wrong_inputs_fails() {
        let system = MockProofSystem::new();
        let pk = MockProofSystem::generate_proving_key();
        let vk = system.derive_verification_key(&pk);
        let validator = Keypair::generate();

        let inputs1 = make_inputs();
        let mut inputs2 = make_inputs();
        inputs2.new_accuracy = 0.99; // tampered accuracy

        let proof = system.generate_proof(&pk, &validator.address(), &inputs1);
        // Tampered inputs should fail verification
        assert!(!system.verify_proof(&vk, &proof, &validator.address(), &inputs2));
    }

    #[test]
    fn test_mock_proof_empty_fails() {
        let system = MockProofSystem::new();
        let pk = MockProofSystem::generate_proving_key();
        let vk = system.derive_verification_key(&pk);
        let validator = Keypair::generate();
        let inputs = make_inputs();

        // Empty proof should fail
        assert!(!system.verify_proof(&vk, &[], &validator.address(), &inputs));
        // Wrong size should fail
        assert!(!system.verify_proof(&vk, &[0u8; 16], &validator.address(), &inputs));
    }

    #[test]
    fn test_mock_proof_random_bytes_fail() {
        let system = MockProofSystem::new();
        let pk = MockProofSystem::generate_proving_key();
        let vk = system.derive_verification_key(&pk);
        let validator = Keypair::generate();
        let inputs = make_inputs();

        let fake_proof = vec![0xDE; MOCK_PROOF_SIZE];
        assert!(!system.verify_proof(&vk, &fake_proof, &validator.address(), &inputs));
    }

    #[test]
    fn test_validate_accuracy_valid() {
        assert!(validate_accuracy(0.0, "test").is_ok());
        assert!(validate_accuracy(0.5, "test").is_ok());
        assert!(validate_accuracy(1.0, "test").is_ok());
    }

    #[test]
    fn test_validate_accuracy_invalid() {
        assert!(validate_accuracy(f64::NAN, "test").is_err());
        assert!(validate_accuracy(f64::INFINITY, "test").is_err());
        assert!(validate_accuracy(-0.1, "test").is_err());
        assert!(validate_accuracy(1.1, "test").is_err());
    }

    #[test]
    fn test_accuracy_consistency_ok() {
        let result = check_accuracy_consistency(&[0.78, 0.78, 0.78, 0.78]);
        assert!(result.is_ok());
        assert!((result.unwrap() - 0.78).abs() < 0.001);
    }

    #[test]
    fn test_accuracy_consistency_within_tolerance() {
        let result = check_accuracy_consistency(&[0.780, 0.7805, 0.7798]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_accuracy_consistency_fails() {
        let result = check_accuracy_consistency(&[0.70, 0.78, 0.95]);
        assert!(result.is_err());
    }
}
