//! Transaction types for the Viturka blockchain.
//!
//! Four transaction types per the whitepaper:
//! - `Tx`: Standard VIT token transfers
//! - `DTx`: Data contribution submissions
//! - `RTx`: Decryption key reveal for committed data
//! - `PTx`: ZKML proof submission from validators

use serde::{Deserialize, Serialize};

use crate::types::{Address, CategoryId, Hash, Hashable, Keypair, Signature, CID};

// ---------------------------------------------------------------------------
// Tx — Standard token transfer
// ---------------------------------------------------------------------------

/// Standard VIT token transfer between two addresses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tx {
    pub sender: Address,
    pub recipient: Address,
    pub amount: u64,
    pub fee: u64,
    pub nonce: u64,
    pub timestamp: u64,
    pub signature: Signature,
}

impl Tx {
    /// Create a new unsigned transfer transaction.
    pub fn new(
        sender: Address,
        recipient: Address,
        amount: u64,
        fee: u64,
        nonce: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            sender,
            recipient,
            amount,
            fee,
            nonce,
            timestamp,
            signature: Signature::empty(),
        }
    }

    /// Return the signable bytes (everything except the signature field).
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.sender.as_bytes());
        bytes.extend_from_slice(self.recipient.as_bytes());
        bytes.extend_from_slice(&self.amount.to_le_bytes());
        bytes.extend_from_slice(&self.fee.to_le_bytes());
        bytes.extend_from_slice(&self.nonce.to_le_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes
    }

    /// Sign this transaction with a keypair.
    pub fn sign(&mut self, keypair: &Keypair) {
        self.signature = keypair.sign(&self.signable_bytes());
    }

    /// Verify the signature is valid for the sender address.
    pub fn verify_signature(&self) -> bool {
        crate::types::verify_signature(&self.sender, &self.signable_bytes(), &self.signature)
    }
}

impl Hashable for Tx {
    fn to_hash_bytes(&self) -> Vec<u8> {
        let mut bytes = self.signable_bytes();
        bytes.extend_from_slice(self.signature.as_bytes());
        bytes
    }
}

// ---------------------------------------------------------------------------
// DTx — Data contribution transaction
// ---------------------------------------------------------------------------

/// Data contribution transaction. Contributors submit encrypted training data
/// along with a hash commitment and fee deposit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DTx {
    pub contributor: Address,
    pub category_id: CategoryId,
    /// SHA-256 hash of the plaintext data.
    pub data_hash: Hash,
    /// IPFS CID of the encrypted data blob.
    pub encrypted_cid: CID,
    /// SHA-256 hash of the decryption key (commitment).
    pub encryption_key_hash: Hash,
    /// Number of training samples in the contribution.
    pub sample_count: u32,
    /// Locked fee deposit in VIT (must be DTX_DEPOSIT = 100).
    pub fee_deposit: u64,
    pub timestamp: u64,
    pub signature: Signature,
}

impl DTx {
    /// Create a new unsigned data transaction.
    pub fn new(
        contributor: Address,
        category_id: CategoryId,
        data_hash: Hash,
        encrypted_cid: CID,
        encryption_key_hash: Hash,
        sample_count: u32,
        fee_deposit: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            contributor,
            category_id,
            data_hash,
            encrypted_cid,
            encryption_key_hash,
            sample_count,
            fee_deposit,
            timestamp,
            signature: Signature::empty(),
        }
    }

    /// Return the signable bytes.
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.contributor.as_bytes());
        bytes.extend_from_slice(self.category_id.0.as_bytes());
        bytes.extend_from_slice(self.data_hash.as_bytes());
        bytes.extend_from_slice(self.encrypted_cid.0.as_bytes());
        bytes.extend_from_slice(self.encryption_key_hash.as_bytes());
        bytes.extend_from_slice(&self.sample_count.to_le_bytes());
        bytes.extend_from_slice(&self.fee_deposit.to_le_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes
    }

    /// Sign this transaction with a keypair.
    pub fn sign(&mut self, keypair: &Keypair) {
        self.signature = keypair.sign(&self.signable_bytes());
    }

    /// Verify the signature is valid for the contributor address.
    pub fn verify_signature(&self) -> bool {
        crate::types::verify_signature(
            &self.contributor,
            &self.signable_bytes(),
            &self.signature,
        )
    }
}

impl Hashable for DTx {
    fn to_hash_bytes(&self) -> Vec<u8> {
        let mut bytes = self.signable_bytes();
        bytes.extend_from_slice(self.signature.as_bytes());
        bytes
    }
}

// ---------------------------------------------------------------------------
// RTx — Reveal transaction
// ---------------------------------------------------------------------------

/// Reveal transaction. The contributor reveals the decryption key for their
/// previously committed data, allowing validators to decrypt and validate it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RTx {
    /// Hash reference to the DDTXO being revealed.
    pub ddtxo_reference: Hash,
    /// The actual AES-256 decryption key (32 bytes).
    pub decryption_key: Vec<u8>,
    /// Address of the original contributor.
    pub contributor: Address,
    pub signature: Signature,
}

impl RTx {
    /// Create a new unsigned reveal transaction.
    pub fn new(ddtxo_reference: Hash, decryption_key: Vec<u8>, contributor: Address) -> Self {
        Self {
            ddtxo_reference,
            decryption_key,
            contributor,
            signature: Signature::empty(),
        }
    }

    /// Return the signable bytes.
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.ddtxo_reference.as_bytes());
        bytes.extend_from_slice(&self.decryption_key);
        bytes.extend_from_slice(self.contributor.as_bytes());
        bytes
    }

    /// Sign this transaction with a keypair.
    pub fn sign(&mut self, keypair: &Keypair) {
        self.signature = keypair.sign(&self.signable_bytes());
    }

    /// Verify the signature is valid for the contributor address.
    pub fn verify_signature(&self) -> bool {
        crate::types::verify_signature(
            &self.contributor,
            &self.signable_bytes(),
            &self.signature,
        )
    }
}

impl Hashable for RTx {
    fn to_hash_bytes(&self) -> Vec<u8> {
        let mut bytes = self.signable_bytes();
        bytes.extend_from_slice(self.signature.as_bytes());
        bytes
    }
}

// ---------------------------------------------------------------------------
// ProofPublicInputs — Public inputs embedded in a PTx
// ---------------------------------------------------------------------------

/// Public inputs for a ZKML proof. These are verified on-chain against
/// the known chain state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofPublicInputs {
    /// Must match the DDTXO's data_hash.
    pub data_hash: Hash,
    /// Hash of the model checkpoint before training.
    pub prev_model_hash: Hash,
    /// Hash of the model checkpoint after training.
    pub new_model_hash: Hash,
    /// Model accuracy before training on the test set.
    pub prev_accuracy: f64,
    /// Model accuracy after training on the test set.
    pub new_accuracy: f64,
    /// Hash of the test set used for evaluation.
    pub test_set_hash: Hash,
}

impl ProofPublicInputs {
    /// Serialize to bytes for inclusion in signing/hashing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.data_hash.as_bytes());
        bytes.extend_from_slice(self.prev_model_hash.as_bytes());
        bytes.extend_from_slice(self.new_model_hash.as_bytes());
        bytes.extend_from_slice(&self.prev_accuracy.to_le_bytes());
        bytes.extend_from_slice(&self.new_accuracy.to_le_bytes());
        bytes.extend_from_slice(self.test_set_hash.as_bytes());
        bytes
    }
}

// ---------------------------------------------------------------------------
// PTx — Proof transaction
// ---------------------------------------------------------------------------

/// Proof transaction. Validators submit ZK proofs attesting to correct
/// training execution and resulting accuracy. In Phase 1, the proof field
/// is a stub (empty bytes); real ZKML proofs are added in Phase 2.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PTx {
    pub validator: Address,
    /// Hash reference to the DDTXO being validated.
    pub ddtxo_reference: Hash,
    /// ZK-SNARK proof bytes. Stub (empty) in Phase 1.
    pub proof: Vec<u8>,
    /// Publicly verifiable inputs to the proof.
    pub public_inputs: ProofPublicInputs,
    /// IPFS CID of the new model weights after training.
    pub new_weights_cid: CID,
    pub timestamp: u64,
    pub signature: Signature,
}

impl PTx {
    /// Create a new unsigned proof transaction.
    pub fn new(
        validator: Address,
        ddtxo_reference: Hash,
        proof: Vec<u8>,
        public_inputs: ProofPublicInputs,
        new_weights_cid: CID,
        timestamp: u64,
    ) -> Self {
        Self {
            validator,
            ddtxo_reference,
            proof,
            public_inputs,
            new_weights_cid,
            timestamp,
            signature: Signature::empty(),
        }
    }

    /// Return the signable bytes.
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.validator.as_bytes());
        bytes.extend_from_slice(self.ddtxo_reference.as_bytes());
        bytes.extend_from_slice(&self.proof);
        bytes.extend_from_slice(&self.public_inputs.to_bytes());
        bytes.extend_from_slice(self.new_weights_cid.0.as_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes
    }

    /// Sign this transaction with a keypair.
    pub fn sign(&mut self, keypair: &Keypair) {
        self.signature = keypair.sign(&self.signable_bytes());
    }

    /// Verify the signature is valid for the validator address.
    pub fn verify_signature(&self) -> bool {
        crate::types::verify_signature(&self.validator, &self.signable_bytes(), &self.signature)
    }
}

impl Hashable for PTx {
    fn to_hash_bytes(&self) -> Vec<u8> {
        let mut bytes = self.signable_bytes();
        bytes.extend_from_slice(self.signature.as_bytes());
        bytes
    }
}

// ---------------------------------------------------------------------------
// Transaction — Unified enum
// ---------------------------------------------------------------------------

/// Unified transaction enum wrapping all four transaction types.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Transaction {
    Transfer(Tx),
    DataSubmission(DTx),
    Reveal(RTx),
    Proof(PTx),
}

impl Transaction {
    /// Verify the signature of any transaction variant.
    pub fn verify_signature(&self) -> bool {
        match self {
            Transaction::Transfer(tx) => tx.verify_signature(),
            Transaction::DataSubmission(dtx) => dtx.verify_signature(),
            Transaction::Reveal(rtx) => rtx.verify_signature(),
            Transaction::Proof(ptx) => ptx.verify_signature(),
        }
    }
}

impl Hashable for Transaction {
    fn to_hash_bytes(&self) -> Vec<u8> {
        match self {
            Transaction::Transfer(tx) => tx.to_hash_bytes(),
            Transaction::DataSubmission(dtx) => dtx.to_hash_bytes(),
            Transaction::Reveal(rtx) => rtx.to_hash_bytes(),
            Transaction::Proof(ptx) => ptx.to_hash_bytes(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_keypair() -> Keypair {
        Keypair::generate()
    }

    fn now() -> u64 {
        1_700_000_000_000
    }

    #[test]
    fn test_tx_sign_verify() {
        let sender = make_keypair();
        let recipient = make_keypair();

        let mut tx = Tx::new(sender.address(), recipient.address(), 100, 1, 0, now());
        assert!(!tx.verify_signature()); // unsigned

        tx.sign(&sender);
        assert!(tx.verify_signature());
    }

    #[test]
    fn test_tx_wrong_signer() {
        let sender = make_keypair();
        let recipient = make_keypair();
        let imposter = make_keypair();

        let mut tx = Tx::new(sender.address(), recipient.address(), 100, 1, 0, now());
        tx.sign(&imposter); // signed by wrong key
        assert!(!tx.verify_signature());
    }

    #[test]
    fn test_tx_hash_deterministic() {
        let sender = make_keypair();
        let recipient = make_keypair();

        let mut tx = Tx::new(sender.address(), recipient.address(), 100, 1, 0, now());
        tx.sign(&sender);

        let h1 = tx.hash();
        let h2 = tx.hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_dtx_sign_verify() {
        let contributor = make_keypair();
        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"data"),
            CID("QmTest123".into()),
            Hash::compute(b"key"),
            1000,
            100,
            now(),
        );
        dtx.sign(&contributor);
        assert!(dtx.verify_signature());
    }

    #[test]
    fn test_rtx_sign_verify() {
        let contributor = make_keypair();
        let mut rtx = RTx::new(
            Hash::compute(b"ddtxo_ref"),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            contributor.address(),
        );
        rtx.sign(&contributor);
        assert!(rtx.verify_signature());
    }

    #[test]
    fn test_ptx_sign_verify() {
        let validator = make_keypair();
        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"data"),
            prev_model_hash: Hash::compute(b"prev_model"),
            new_model_hash: Hash::compute(b"new_model"),
            prev_accuracy: 0.85,
            new_accuracy: 0.87,
            test_set_hash: Hash::compute(b"test_set"),
        };
        let mut ptx = PTx::new(
            validator.address(),
            Hash::compute(b"ddtxo"),
            vec![], // stub proof
            inputs,
            CID("QmNewWeights".into()),
            now(),
        );
        ptx.sign(&validator);
        assert!(ptx.verify_signature());
    }

    #[test]
    fn test_transaction_enum_verify() {
        let kp = make_keypair();
        let kp2 = make_keypair();
        let mut tx = Tx::new(kp.address(), kp2.address(), 50, 1, 0, now());
        tx.sign(&kp);

        let unified = Transaction::Transfer(tx);
        assert!(unified.verify_signature());
    }

    #[test]
    fn test_tx_serialization_roundtrip() {
        let sender = make_keypair();
        let recipient = make_keypair();
        let mut tx = Tx::new(sender.address(), recipient.address(), 100, 1, 0, now());
        tx.sign(&sender);

        let json = serde_json::to_string(&tx).expect("serialize");
        let tx2: Tx = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(tx.hash(), tx2.hash());
    }
}
