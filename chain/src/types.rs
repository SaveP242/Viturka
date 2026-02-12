//! Core types used throughout the Viturka blockchain.

use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

// ---------------------------------------------------------------------------
// Address — 32-byte Ed25519 public key wrapper
// ---------------------------------------------------------------------------

/// A network address derived from an Ed25519 public key.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address(pub [u8; 32]);

impl Address {
    /// Create an address from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Create an address from a verifying (public) key.
    pub fn from_public_key(key: &VerifyingKey) -> Self {
        Self(key.to_bytes())
    }

    /// Return the underlying bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Return the hex-encoded representation.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address({}..{})", &self.to_hex()[..8], &self.to_hex()[56..])
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl PartialOrd for Address {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Address {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Hash — 32-byte SHA-256 digest wrapper
// ---------------------------------------------------------------------------

/// A SHA-256 hash digest.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    /// The zero hash (used as previous_hash for genesis block).
    pub fn zero() -> Self {
        Self([0u8; 32])
    }

    /// Create a hash from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Compute SHA-256 of arbitrary data.
    pub fn compute(data: &[u8]) -> Self {
        let digest = Sha256::digest(data);
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&digest);
        Self(bytes)
    }

    /// Return the hex-encoded representation.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Return the underlying bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash({}..{})", &self.to_hex()[..8], &self.to_hex()[56..])
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// ---------------------------------------------------------------------------
// Signature — 64-byte Ed25519 signature wrapper
// ---------------------------------------------------------------------------

/// An Ed25519 signature. Uses Vec<u8> internally for serde compatibility.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature(pub Vec<u8>);

impl Signature {
    /// Create a signature from raw bytes.
    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        Self(bytes.to_vec())
    }

    /// Return a placeholder empty signature (for unsigned transaction construction).
    pub fn empty() -> Self {
        Self(vec![0u8; 64])
    }

    /// Return the underlying bytes as a slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Signature({}..)", &hex::encode(&self.0[..8]))
    }
}

// ---------------------------------------------------------------------------
// Keypair helper — wraps ed25519-dalek for convenience
// ---------------------------------------------------------------------------

/// A signing keypair for creating transactions.
pub struct Keypair {
    pub signing_key: SigningKey,
    pub verifying_key: VerifyingKey,
}

impl Keypair {
    /// Generate a new random keypair.
    pub fn generate() -> Self {
        let mut rng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Return the address derived from this keypair.
    pub fn address(&self) -> Address {
        Address::from_public_key(&self.verifying_key)
    }

    /// Sign arbitrary data.
    pub fn sign(&self, data: &[u8]) -> Signature {
        use ed25519_dalek::Signer;
        let sig = self.signing_key.sign(data);
        Signature(sig.to_bytes().to_vec())
    }

    /// Verify a signature against this keypair's public key.
    pub fn verify(&self, data: &[u8], signature: &Signature) -> bool {
        verify_signature(&self.address(), data, signature)
    }
}

/// Verify a signature against an address (public key).
pub fn verify_signature(address: &Address, data: &[u8], signature: &Signature) -> bool {
    let Ok(verifying_key) = VerifyingKey::from_bytes(address.as_bytes()) else {
        return false;
    };
    if signature.0.len() != 64 {
        return false;
    }
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(&signature.0);
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    use ed25519_dalek::Verifier;
    verifying_key.verify(data, &sig).is_ok()
}

// ---------------------------------------------------------------------------
// Simple wrapper types
// ---------------------------------------------------------------------------

/// Category identifier for a model category (e.g. "defi_credit_v1").
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CategoryId(pub String);

impl fmt::Display for CategoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// IPFS content identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CID(pub String);

impl fmt::Display for CID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Hashable trait
// ---------------------------------------------------------------------------

/// Trait for types that can produce a canonical byte representation for hashing.
pub trait Hashable {
    /// Serialize to canonical bytes for hashing.
    fn to_hash_bytes(&self) -> Vec<u8>;

    /// Compute the SHA-256 hash of the canonical bytes.
    fn hash(&self) -> Hash {
        Hash::compute(&self.to_hash_bytes())
    }
}

// ---------------------------------------------------------------------------
// Merkle root computation
// ---------------------------------------------------------------------------

/// Compute a merkle root from a list of hashes.
/// Uses recursive SHA-256 pair hashing. Returns zero hash for empty list.
pub fn compute_merkle_root(hashes: &[Hash]) -> Hash {
    if hashes.is_empty() {
        return Hash::zero();
    }
    if hashes.len() == 1 {
        return hashes[0].clone();
    }

    let mut current_level: Vec<Hash> = hashes.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        for chunk in current_level.chunks(2) {
            let combined = if chunk.len() == 2 {
                let mut data = Vec::with_capacity(64);
                data.extend_from_slice(chunk[0].as_bytes());
                data.extend_from_slice(chunk[1].as_bytes());
                Hash::compute(&data)
            } else {
                // Odd element: hash with itself
                let mut data = Vec::with_capacity(64);
                data.extend_from_slice(chunk[0].as_bytes());
                data.extend_from_slice(chunk[0].as_bytes());
                Hash::compute(&data)
            };
            next_level.push(combined);
        }
        current_level = next_level;
    }

    current_level.into_iter().next().unwrap_or_else(Hash::zero)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_display_roundtrip() {
        let kp = Keypair::generate();
        let addr = kp.address();
        let hex_str = addr.to_hex();
        assert_eq!(hex_str.len(), 64);

        let bytes = hex::decode(&hex_str).expect("valid hex");
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        assert_eq!(Address::from_bytes(arr), addr);
    }

    #[test]
    fn test_hash_compute_deterministic() {
        let data = b"viturka";
        let h1 = Hash::compute(data);
        let h2 = Hash::compute(data);
        assert_eq!(h1, h2);

        let h3 = Hash::compute(b"different");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_sign_verify() {
        let kp = Keypair::generate();
        let msg = b"test message";
        let sig = kp.sign(msg);
        assert!(kp.verify(msg, &sig));
        assert!(!kp.verify(b"wrong message", &sig));
    }

    #[test]
    fn test_verify_wrong_key() {
        let kp1 = Keypair::generate();
        let kp2 = Keypair::generate();
        let msg = b"test";
        let sig = kp1.sign(msg);
        assert!(!verify_signature(&kp2.address(), msg, &sig));
    }

    #[test]
    fn test_merkle_root_empty() {
        assert_eq!(compute_merkle_root(&[]), Hash::zero());
    }

    #[test]
    fn test_merkle_root_single() {
        let h = Hash::compute(b"leaf");
        assert_eq!(compute_merkle_root(&[h.clone()]), h);
    }

    #[test]
    fn test_merkle_root_deterministic() {
        let leaves: Vec<Hash> = (0..5).map(|i| Hash::compute(&[i])).collect();
        let r1 = compute_merkle_root(&leaves);
        let r2 = compute_merkle_root(&leaves);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_merkle_root_order_matters() {
        let h1 = Hash::compute(b"a");
        let h2 = Hash::compute(b"b");
        let r1 = compute_merkle_root(&[h1.clone(), h2.clone()]);
        let r2 = compute_merkle_root(&[h2, h1]);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_address_ordering() {
        let a1 = Address::from_bytes([0u8; 32]);
        let a2 = Address::from_bytes([1u8; 32]);
        assert!(a1 < a2);
    }
}
