//! Error types for the Viturka blockchain.

use thiserror::Error;

/// Top-level error type for chain operations.
#[derive(Debug, Error)]
pub enum ChainError {
    #[error("insufficient balance: have {have}, need {need}")]
    InsufficientBalance { have: u64, need: u64 },

    #[error("invalid signature")]
    InvalidSignature,

    #[error("signature verification failed: {0}")]
    SignatureError(String),

    #[error("DDTXO not found: {0}")]
    DDTXONotFound(String),

    #[error("invalid DDTXO state transition: {from} -> {to}")]
    InvalidStateTransition { from: String, to: String },

    #[error("reveal deadline passed at block {deadline}, current block {current}")]
    RevealDeadlinePassed { deadline: u64, current: u64 },

    #[error("validation deadline not reached: deadline {deadline}, current {current}")]
    ValidationDeadlineNotReached { deadline: u64, current: u64 },

    #[error("key hash mismatch — contributor banned")]
    KeyHashMismatch,

    #[error("validator not eligible: {reason}")]
    ValidatorNotEligible { reason: String },

    #[error("address is banned: {0}")]
    AddressBanned(String),

    #[error("invalid DTx deposit: expected {expected}, got {got}")]
    InvalidDeposit { expected: u64, got: u64 },

    #[error("duplicate transaction: {0}")]
    DuplicateTransaction(String),

    #[error("invalid nonce: expected {expected}, got {got}")]
    InvalidNonce { expected: u64, got: u64 },

    #[error("account not found: {0}")]
    AccountNotFound(String),

    #[error("not the original contributor")]
    NotContributor,

    #[error("reveal window not open yet: opens at block {opens_at}, current {current}")]
    RevealWindowNotOpen { opens_at: u64, current: u64 },

    #[error("invalid block: {0}")]
    InvalidBlock(String),

    #[error("genesis block already exists")]
    GenesisExists,

    #[error("previous block hash mismatch")]
    PreviousHashMismatch,

    #[error("invalid block height: expected {expected}, got {got}")]
    InvalidBlockHeight { expected: u64, got: u64 },

    #[error("no eligible validators")]
    NoEligibleValidators,

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("category not found: {0}")]
    CategoryNotFound(String),

    #[error("accuracy did not improve: previous {previous}, new {new}")]
    AccuracyNotImproved { previous: f64, new: f64 },

    #[error("insufficient valid proofs: need {need}, got {got}")]
    InsufficientProofs { need: usize, got: usize },

    #[error("proof verification failed for validator {validator}")]
    ProofVerificationFailed { validator: String },

    #[error("accuracy inconsistent across validators: min={min:.4}, max={max:.4}, tolerance={tolerance}")]
    AccuracyInconsistent { min: f64, max: f64, tolerance: f64 },

    #[error("validator does not support category: {category}")]
    UnsupportedCategory { category: String },

    #[error("invalid stake amount: {0}")]
    InvalidStake(String),
}

/// Result type alias using ChainError.
pub type Result<T> = std::result::Result<T, ChainError>;
