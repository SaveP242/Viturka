//! In-memory chain state management for the Viturka blockchain.
//!
//! Tracks accounts, validators, DDTXOs, model checkpoints, and blocks.
//! Provides methods to apply each transaction type and finalize DDTXOs.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::block::{Block, BlockBody};
use crate::consensus::{self, ValidatorCandidate};
use crate::error::{ChainError, Result};
use crate::proof::{check_accuracy_consistency, validate_accuracy, ProofSystem, VerificationKey};
use crate::transaction::{DTx, PTx, RTx, Tx};
use crate::types::{Address, CategoryId, Hash, Hashable, CID};

// ---------------------------------------------------------------------------
// DDTXO — Distributed Data Transaction Output
// ---------------------------------------------------------------------------

/// Status of a DDTXO through its lifecycle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DDTXOStatus {
    /// DTx included, waiting for reveal.
    Pending,
    /// Decryption key revealed, waiting for validation.
    Revealed,
    /// Validation proofs being collected.
    Validating,
    /// Approved — data improved model accuracy.
    Approved,
    /// Rejected — data did not improve accuracy, or insufficient proofs.
    Rejected,
    /// Expired — contributor did not reveal in time.
    Expired,
    /// Invalid — key hash mismatch (contributor banned).
    Invalid,
}

impl std::fmt::Display for DDTXOStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DDTXOStatus::Pending => write!(f, "PENDING"),
            DDTXOStatus::Revealed => write!(f, "REVEALED"),
            DDTXOStatus::Validating => write!(f, "VALIDATING"),
            DDTXOStatus::Approved => write!(f, "APPROVED"),
            DDTXOStatus::Rejected => write!(f, "REJECTED"),
            DDTXOStatus::Expired => write!(f, "EXPIRED"),
            DDTXOStatus::Invalid => write!(f, "INVALID"),
        }
    }
}

/// A recorded validation proof from a validator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationProof {
    pub validator: Address,
    pub new_accuracy: f64,
    pub new_model_hash: Hash,
    pub new_weights_cid: CID,
}

/// A pending data contribution tracked on-chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DDTXO {
    pub dtx_hash: Hash,
    pub data_hash: Hash,
    pub encrypted_cid: CID,
    pub encryption_key_hash: Hash,
    pub contributor: Address,
    pub category_id: CategoryId,
    pub fee_locked: u64,
    pub block_created: u64,
    /// Block at which the contributor must reveal (block_created + REVEAL_WINDOW).
    pub reveal_deadline: u64,
    /// Block at which validation finalizes (block_created + REVEAL_WINDOW + VALIDATION_WINDOW).
    pub validation_deadline: u64,
    pub status: DDTXOStatus,
    /// Set after a valid RTx reveal.
    pub decryption_key: Option<Vec<u8>>,
    /// Valid proofs submitted by validators.
    pub validation_proofs: Vec<ValidationProof>,
    /// Final consensus accuracy (set on approval).
    pub final_accuracy: Option<f64>,
}

// ---------------------------------------------------------------------------
// Account
// ---------------------------------------------------------------------------

/// An account on the Viturka network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Account {
    pub address: Address,
    pub balance: u64,
    pub nonce: u64,
    pub credibility: u64,
    pub is_banned: bool,
}

impl Account {
    /// Create a new account with zero balance and no credibility.
    pub fn new(address: Address) -> Self {
        Self {
            address,
            balance: 0,
            nonce: 0,
            credibility: 0,
            is_banned: false,
        }
    }

    /// Create a new account with a starting balance (for genesis).
    pub fn with_balance(address: Address, balance: u64) -> Self {
        Self {
            address,
            balance,
            nonce: 0,
            credibility: 0,
            is_banned: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ValidatorInfo
// ---------------------------------------------------------------------------

/// On-chain state for a registered validator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub address: Address,
    pub credibility: u64,
    /// Block height of last validation participation (0 = never).
    pub last_validation_block: u64,
    pub zkml_registered: bool,
    pub supported_categories: Vec<CategoryId>,
    pub total_validations: u64,
    pub stake: u64,
}

// ---------------------------------------------------------------------------
// ModelCheckpoint
// ---------------------------------------------------------------------------

/// Tracks the current model state for a category.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub category_id: CategoryId,
    pub model_hash: Hash,
    pub weights_cid: CID,
    pub accuracy: f64,
    pub last_updated_block: u64,
}

// ---------------------------------------------------------------------------
// ChainState
// ---------------------------------------------------------------------------

/// In-memory state of the entire Viturka blockchain.
#[derive(Clone)]
pub struct ChainState {
    pub accounts: HashMap<Address, Account>,
    pub validators: HashMap<Address, ValidatorInfo>,
    pub ddtxos: HashMap<Hash, DDTXO>,
    pub blocks: Vec<Block>,
    pub current_height: u64,
    pub model_checkpoints: HashMap<CategoryId, ModelCheckpoint>,
    pub banned_addresses: HashSet<Address>,
    /// Pool of pending transactions not yet included in a block.
    pub pending_tx: Vec<Tx>,
    pub pending_dtx: Vec<DTx>,
    pub pending_rtx: Vec<RTx>,
    pub pending_ptx: Vec<PTx>,
    /// Pluggable proof system for ZKML verification.
    pub proof_system: Arc<dyn ProofSystem>,
    /// Verification keys for each registered validator.
    pub verification_keys: HashMap<Address, VerificationKey>,
}

impl std::fmt::Debug for ChainState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainState")
            .field("current_height", &self.current_height)
            .field("accounts", &self.accounts.len())
            .field("validators", &self.validators.len())
            .field("ddtxos", &self.ddtxos.len())
            .field("blocks", &self.blocks.len())
            .field("proof_system", &self.proof_system.name())
            .finish()
    }
}

impl ChainState {
    /// Create a new empty chain state with the given proof system.
    pub fn new(proof_system: Arc<dyn ProofSystem>) -> Self {
        Self {
            accounts: HashMap::new(),
            validators: HashMap::new(),
            ddtxos: HashMap::new(),
            blocks: Vec::new(),
            current_height: 0,
            model_checkpoints: HashMap::new(),
            banned_addresses: HashSet::new(),
            pending_tx: Vec::new(),
            pending_dtx: Vec::new(),
            pending_rtx: Vec::new(),
            pending_ptx: Vec::new(),
            proof_system,
            verification_keys: HashMap::new(),
        }
    }

    /// Initialize chain state with a genesis block, pre-funded accounts,
    /// pre-registered validators, and a pluggable proof system.
    pub fn initialize_genesis(
        genesis_accounts: Vec<(Address, u64)>,
        genesis_validators: Vec<(Address, u64)>,
        initial_categories: Vec<(CategoryId, f64)>,
        proof_system: Arc<dyn ProofSystem>,
        verification_keys: HashMap<Address, VerificationKey>,
    ) -> Self {
        let mut state = Self::new(proof_system);
        state.verification_keys = verification_keys;

        // Create accounts with initial balances
        for (addr, balance) in &genesis_accounts {
            state
                .accounts
                .insert(addr.clone(), Account::with_balance(addr.clone(), *balance));
        }

        // Register validators with initial credibility
        for (addr, credibility) in &genesis_validators {
            // Ensure the validator also has an account
            state
                .accounts
                .entry(addr.clone())
                .or_insert_with(|| Account::with_balance(addr.clone(), 0));

            // Set account credibility to match
            if let Some(acct) = state.accounts.get_mut(addr) {
                acct.credibility = *credibility;
            }

            state.validators.insert(
                addr.clone(),
                ValidatorInfo {
                    address: addr.clone(),
                    credibility: *credibility,
                    last_validation_block: 0,
                    zkml_registered: true,
                    supported_categories: initial_categories
                        .iter()
                        .map(|(cat, _)| cat.clone())
                        .collect(),
                    total_validations: 0,
                    stake: consensus::GENESIS_VALIDATOR_STAKE,
                },
            );
        }

        // Initialize model checkpoints for each category
        for (cat_id, initial_accuracy) in &initial_categories {
            state.model_checkpoints.insert(
                cat_id.clone(),
                ModelCheckpoint {
                    category_id: cat_id.clone(),
                    model_hash: Hash::compute(cat_id.0.as_bytes()),
                    weights_cid: CID(format!("genesis_{}", cat_id.0)),
                    accuracy: *initial_accuracy,
                    last_updated_block: 0,
                },
            );
        }

        // Create the genesis block
        let proposer = genesis_validators
            .first()
            .map(|(addr, _)| addr.clone())
            .unwrap_or_else(|| genesis_accounts[0].0.clone());

        let genesis = Block::genesis(proposer, current_timestamp());
        state.blocks.push(genesis);

        state
    }

    // -----------------------------------------------------------------------
    // Account helpers
    // -----------------------------------------------------------------------

    /// Get or create an account for the given address.
    pub fn get_or_create_account(&mut self, address: &Address) -> &mut Account {
        self.accounts
            .entry(address.clone())
            .or_insert_with(|| Account::new(address.clone()))
    }

    /// Get the balance of an address (0 if account doesn't exist).
    pub fn get_balance(&self, address: &Address) -> u64 {
        self.accounts.get(address).map_or(0, |a| a.balance)
    }

    /// Get the credibility of an address (0 if not found).
    pub fn get_credibility(&self, address: &Address) -> u64 {
        self.validators
            .get(address)
            .map_or(0, |v| v.credibility)
    }

    // -----------------------------------------------------------------------
    // Top-10 validator selection
    // -----------------------------------------------------------------------

    /// Get the list of eligible validator candidates at the given block height.
    pub fn get_eligible_validators(&self, block_height: u64) -> Vec<ValidatorCandidate> {
        self.validators
            .values()
            .filter(|v| {
                v.credibility >= consensus::MIN_CREDIBILITY
                    && !consensus::is_in_cooldown(v.last_validation_block, block_height)
                    && v.zkml_registered
            })
            .map(|v| ValidatorCandidate {
                address: v.address.clone(),
                credibility: v.credibility,
            })
            .collect()
    }

    /// Get the top validators for the given block height.
    pub fn get_top_validators(&self, block_height: u64) -> Vec<Address> {
        let candidates = self.get_eligible_validators(block_height);
        consensus::select_top_validators(&candidates)
    }

    // -----------------------------------------------------------------------
    // Apply transactions
    // -----------------------------------------------------------------------

    /// Apply a standard transfer transaction.
    pub fn apply_tx(&mut self, tx: &Tx) -> Result<()> {
        if self.banned_addresses.contains(&tx.sender) {
            return Err(ChainError::AddressBanned(tx.sender.to_hex()));
        }

        if !tx.verify_signature() {
            return Err(ChainError::InvalidSignature);
        }

        let sender_balance = self.get_balance(&tx.sender);
        let total_cost = tx
            .amount
            .checked_add(tx.fee)
            .ok_or(ChainError::InsufficientBalance {
                have: sender_balance,
                need: u64::MAX,
            })?;

        if sender_balance < total_cost {
            return Err(ChainError::InsufficientBalance {
                have: sender_balance,
                need: total_cost,
            });
        }

        // Check nonce
        let expected_nonce = self
            .accounts
            .get(&tx.sender)
            .map_or(0, |a| a.nonce);
        if tx.nonce != expected_nonce {
            return Err(ChainError::InvalidNonce {
                expected: expected_nonce,
                got: tx.nonce,
            });
        }

        // Debit sender
        let sender = self.get_or_create_account(&tx.sender);
        sender.balance -= total_cost;
        sender.nonce += 1;

        // Credit recipient
        let recipient = self.get_or_create_account(&tx.recipient);
        recipient.balance += tx.amount;

        Ok(())
    }

    /// Apply a data contribution transaction, creating a DDTXO.
    pub fn apply_dtx(&mut self, dtx: &DTx, current_block: u64) -> Result<()> {
        if self.banned_addresses.contains(&dtx.contributor) {
            return Err(ChainError::AddressBanned(dtx.contributor.to_hex()));
        }

        if !dtx.verify_signature() {
            return Err(ChainError::InvalidSignature);
        }

        if dtx.fee_deposit != consensus::DTX_DEPOSIT {
            return Err(ChainError::InvalidDeposit {
                expected: consensus::DTX_DEPOSIT,
                got: dtx.fee_deposit,
            });
        }

        // Verify category exists
        if !self.model_checkpoints.contains_key(&dtx.category_id) {
            return Err(ChainError::CategoryNotFound(dtx.category_id.0.clone()));
        }

        let balance = self.get_balance(&dtx.contributor);
        if balance < dtx.fee_deposit {
            return Err(ChainError::InsufficientBalance {
                have: balance,
                need: dtx.fee_deposit,
            });
        }

        // Lock fee
        let contributor = self.get_or_create_account(&dtx.contributor);
        contributor.balance -= dtx.fee_deposit;

        // Create DDTXO
        let dtx_hash = dtx.hash();
        let ddtxo = DDTXO {
            dtx_hash: dtx_hash.clone(),
            data_hash: dtx.data_hash.clone(),
            encrypted_cid: dtx.encrypted_cid.clone(),
            encryption_key_hash: dtx.encryption_key_hash.clone(),
            contributor: dtx.contributor.clone(),
            category_id: dtx.category_id.clone(),
            fee_locked: dtx.fee_deposit,
            block_created: current_block,
            reveal_deadline: current_block + consensus::REVEAL_WINDOW,
            validation_deadline: current_block
                + consensus::REVEAL_WINDOW
                + consensus::VALIDATION_WINDOW,
            status: DDTXOStatus::Pending,
            decryption_key: None,
            validation_proofs: Vec::new(),
            final_accuracy: None,
        };

        self.ddtxos.insert(dtx_hash, ddtxo);
        Ok(())
    }

    /// Apply a reveal transaction, transitioning DDTXO from PENDING to REVEALED.
    pub fn apply_rtx(&mut self, rtx: &RTx, current_block: u64) -> Result<()> {
        if !rtx.verify_signature() {
            return Err(ChainError::InvalidSignature);
        }

        let ddtxo = self
            .ddtxos
            .get_mut(&rtx.ddtxo_reference)
            .ok_or_else(|| ChainError::DDTXONotFound(rtx.ddtxo_reference.to_hex()))?;

        // Must be the original contributor
        if rtx.contributor != ddtxo.contributor {
            return Err(ChainError::NotContributor);
        }

        // Must be in PENDING status
        if ddtxo.status != DDTXOStatus::Pending {
            return Err(ChainError::InvalidStateTransition {
                from: ddtxo.status.to_string(),
                to: "REVEALED".into(),
            });
        }

        // Check reveal deadline
        if current_block > ddtxo.reveal_deadline {
            return Err(ChainError::RevealDeadlinePassed {
                deadline: ddtxo.reveal_deadline,
                current: current_block,
            });
        }

        // Verify key hash commitment
        let key_hash = Hash::compute(&rtx.decryption_key);
        if key_hash != ddtxo.encryption_key_hash {
            // Fraud! Key hash mismatch — ban the contributor
            let contributor = ddtxo.contributor.clone();
            let fee = ddtxo.fee_locked;
            ddtxo.status = DDTXOStatus::Invalid;

            self.banned_addresses.insert(contributor.clone());

            if let Some(acct) = self.accounts.get_mut(&contributor) {
                acct.is_banned = true;
            }

            // Distribute fee to validators
            self.distribute_fee_to_validators(fee);

            return Err(ChainError::KeyHashMismatch);
        }

        // Valid reveal
        ddtxo.decryption_key = Some(rtx.decryption_key.clone());
        ddtxo.status = DDTXOStatus::Revealed;

        Ok(())
    }

    /// Apply a proof transaction from a validator.
    ///
    /// Validates: signature, top-10 eligibility, accuracy range, category support,
    /// DDTXO state, public inputs consistency, duplicate check, and ZKML proof.
    pub fn apply_ptx(&mut self, ptx: &PTx, current_block: u64) -> Result<()> {
        if !ptx.verify_signature() {
            return Err(ChainError::InvalidSignature);
        }

        let top_validators = self.get_top_validators(current_block);
        if !top_validators.contains(&ptx.validator) {
            return Err(ChainError::ValidatorNotEligible {
                reason: "not in top-10 validator set".into(),
            });
        }

        // Validate accuracy values are within valid range [0.0, 1.0]
        validate_accuracy(ptx.public_inputs.prev_accuracy, "prev_accuracy")?;
        validate_accuracy(ptx.public_inputs.new_accuracy, "new_accuracy")?;

        // All validation requiring immutable borrows
        {
            let ddtxo = self
                .ddtxos
                .get(&ptx.ddtxo_reference)
                .ok_or_else(|| ChainError::DDTXONotFound(ptx.ddtxo_reference.to_hex()))?;

            // Enforce category support — validator must support the DDTXO's category
            if let Some(validator_info) = self.validators.get(&ptx.validator) {
                if !validator_info.supported_categories.contains(&ddtxo.category_id) {
                    return Err(ChainError::UnsupportedCategory {
                        category: ddtxo.category_id.0.clone(),
                    });
                }
            }

            // Must be REVEALED or VALIDATING
            if ddtxo.status != DDTXOStatus::Revealed && ddtxo.status != DDTXOStatus::Validating {
                return Err(ChainError::InvalidStateTransition {
                    from: ddtxo.status.to_string(),
                    to: "VALIDATING".into(),
                });
            }

            // Verify public inputs match DDTXO
            if ptx.public_inputs.data_hash != ddtxo.data_hash {
                return Err(ChainError::InvalidBlock(
                    "PTx data_hash does not match DDTXO".into(),
                ));
            }

            // Verify previous model hash matches current checkpoint
            if let Some(checkpoint) = self.model_checkpoints.get(&ddtxo.category_id) {
                if ptx.public_inputs.prev_model_hash != checkpoint.model_hash {
                    return Err(ChainError::InvalidBlock(
                        "PTx prev_model_hash does not match current checkpoint".into(),
                    ));
                }
            }

            // Check that this validator hasn't already submitted for this DDTXO
            let already_submitted = ddtxo
                .validation_proofs
                .iter()
                .any(|p| p.validator == ptx.validator);
            if already_submitted {
                return Err(ChainError::DuplicateTransaction(
                    "Validator already submitted proof for this DDTXO".into(),
                ));
            }

            // Verify the ZKML proof using the pluggable proof system
            let vk = self
                .verification_keys
                .get(&ptx.validator)
                .ok_or_else(|| ChainError::ValidatorNotEligible {
                    reason: "no verification key registered".into(),
                })?;

            if !self
                .proof_system
                .verify_proof(vk, &ptx.proof, &ptx.validator, &ptx.public_inputs)
            {
                return Err(ChainError::ProofVerificationFailed {
                    validator: ptx.validator.to_hex(),
                });
            }
        }

        // Record the proof (mutable borrow)
        let ddtxo = self
            .ddtxos
            .get_mut(&ptx.ddtxo_reference)
            .expect("checked above");
        ddtxo.validation_proofs.push(ValidationProof {
            validator: ptx.validator.clone(),
            new_accuracy: ptx.public_inputs.new_accuracy,
            new_model_hash: ptx.public_inputs.new_model_hash.clone(),
            new_weights_cid: ptx.new_weights_cid.clone(),
        });

        // Transition to VALIDATING if first proof
        if ddtxo.status == DDTXOStatus::Revealed {
            ddtxo.status = DDTXOStatus::Validating;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // DDTXO Finalization
    // -----------------------------------------------------------------------

    /// Process all DDTXOs that have reached their deadlines at the current block height.
    /// - Expire DDTXOs past reveal deadline without reveal.
    /// - Finalize DDTXOs past validation deadline.
    pub fn process_ddtxo_deadlines(&mut self, current_block: u64) {
        // Collect DDTXOs to process (can't borrow mutably while iterating)
        let to_expire: Vec<Hash> = self
            .ddtxos
            .iter()
            .filter(|(_, d)| {
                d.status == DDTXOStatus::Pending && current_block > d.reveal_deadline
            })
            .map(|(h, _)| h.clone())
            .collect();

        let to_finalize: Vec<Hash> = self
            .ddtxos
            .iter()
            .filter(|(_, d)| {
                (d.status == DDTXOStatus::Revealed || d.status == DDTXOStatus::Validating)
                    && current_block >= d.validation_deadline
            })
            .map(|(h, _)| h.clone())
            .collect();

        // Expire unrevealeds
        for hash in to_expire {
            if let Some(ddtxo) = self.ddtxos.get_mut(&hash) {
                ddtxo.status = DDTXOStatus::Expired;
                let fee = ddtxo.fee_locked;
                self.distribute_fee_to_validators(fee);
            }
        }

        // Finalize validated
        for hash in to_finalize {
            self.finalize_ddtxo(&hash);
        }
    }

    /// Finalize a single DDTXO: check proofs, approve or reject, distribute rewards.
    fn finalize_ddtxo(&mut self, ddtxo_hash: &Hash) {
        let ddtxo = match self.ddtxos.get(ddtxo_hash) {
            Some(d) => d.clone(),
            None => return,
        };

        let valid_proofs = &ddtxo.validation_proofs;

        if valid_proofs.len() < consensus::CONSENSUS_THRESHOLD {
            // Not enough proofs — reject
            if let Some(d) = self.ddtxos.get_mut(ddtxo_hash) {
                d.status = DDTXOStatus::Rejected;
            }
            self.distribute_fee_to_validators(ddtxo.fee_locked);
            return;
        }

        // Check accuracy consistency across validators (ZKML ensures determinism)
        let accuracies: Vec<f64> = valid_proofs.iter().map(|p| p.new_accuracy).collect();
        let consensus_accuracy = match check_accuracy_consistency(&accuracies) {
            Ok(acc) => acc,
            Err(_) => {
                // Accuracy claims inconsistent across validators — reject
                if let Some(d) = self.ddtxos.get_mut(ddtxo_hash) {
                    d.status = DDTXOStatus::Rejected;
                }
                self.distribute_fee_to_validators(ddtxo.fee_locked);
                return;
            }
        };

        // Check if accuracy improved
        let prev_accuracy = self
            .model_checkpoints
            .get(&ddtxo.category_id)
            .map_or(0.0, |cp| cp.accuracy);

        let improvement = consensus_accuracy - prev_accuracy;

        if improvement < consensus::MIN_ACCURACY_IMPROVEMENT {
            // Data didn't improve the model
            if let Some(d) = self.ddtxos.get_mut(ddtxo_hash) {
                d.status = DDTXOStatus::Rejected;
                d.final_accuracy = Some(consensus_accuracy);
            }
            self.distribute_fee_to_validators(ddtxo.fee_locked);
            return;
        }

        // APPROVED!
        if let Some(d) = self.ddtxos.get_mut(ddtxo_hash) {
            d.status = DDTXOStatus::Approved;
            d.final_accuracy = Some(consensus_accuracy);
        }

        // Refund contributor fee
        if let Some(acct) = self.accounts.get_mut(&ddtxo.contributor) {
            acct.balance += ddtxo.fee_locked;
        }

        // Pay contributor reward and credibility
        let reward = consensus::calculate_data_reward(improvement, 1.0);
        let cred = consensus::calculate_data_credibility(improvement, 1.0);

        if let Some(acct) = self.accounts.get_mut(&ddtxo.contributor) {
            acct.balance += reward;
            acct.credibility += cred;
        }

        // Update model checkpoint using first valid proof's model
        if let Some(first_proof) = valid_proofs.first() {
            if let Some(checkpoint) = self.model_checkpoints.get_mut(&ddtxo.category_id) {
                checkpoint.model_hash = first_proof.new_model_hash.clone();
                checkpoint.weights_cid = first_proof.new_weights_cid.clone();
                checkpoint.accuracy = consensus_accuracy;
                checkpoint.last_updated_block = self.current_height;
            }
        }

        // Reward validators with valid proofs and apply cooldown
        for proof in valid_proofs {
            if let Some(acct) = self.accounts.get_mut(&proof.validator) {
                acct.balance += consensus::BASE_VALIDATION_REWARD;
            }

            if let Some(validator) = self.validators.get_mut(&proof.validator) {
                validator.credibility += consensus::calculate_validation_credibility();
                validator.last_validation_block = self.current_height;
                validator.total_validations += 1;
            }
        }
    }

    /// Distribute a fee evenly among current validators.
    fn distribute_fee_to_validators(&mut self, fee: u64) {
        let validator_addrs: Vec<Address> = self.validators.keys().cloned().collect();
        if validator_addrs.is_empty() {
            return;
        }
        let share = fee / validator_addrs.len() as u64;
        let remainder = fee % validator_addrs.len() as u64;

        for (i, addr) in validator_addrs.iter().enumerate() {
            let extra = if (i as u64) < remainder { 1 } else { 0 };
            if let Some(acct) = self.accounts.get_mut(addr) {
                acct.balance += share + extra;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Block processing
    // -----------------------------------------------------------------------

    /// Apply a complete block to the chain state.
    pub fn apply_block(&mut self, block: &Block) -> Result<()> {
        let height = block.header.block_height;

        // Validate block height
        let expected_height = if self.blocks.is_empty() { 0 } else { self.current_height + 1 };
        if height != expected_height {
            return Err(ChainError::InvalidBlockHeight {
                expected: expected_height,
                got: height,
            });
        }

        // Validate previous hash (skip for genesis)
        if height > 0 {
            let prev_hash = self.blocks.last().map(|b| b.block_hash()).unwrap_or_else(Hash::zero);
            if block.header.previous_hash != prev_hash {
                return Err(ChainError::PreviousHashMismatch);
            }
        }

        // Verify merkle roots
        if !block.verify_merkle_roots() {
            return Err(ChainError::InvalidBlock("merkle root mismatch".into()));
        }

        // Apply all transactions
        for tx in &block.body.transactions {
            self.apply_tx(tx)?;
        }

        for dtx in &block.body.data_transactions {
            self.apply_dtx(dtx, height)?;
        }

        for rtx in &block.body.reveal_transactions {
            self.apply_rtx(rtx, height)?;
        }

        for ptx in &block.body.proof_transactions {
            self.apply_ptx(ptx, height)?;
        }

        // Process DDTXO deadlines
        self.process_ddtxo_deadlines(height);

        // Apply credibility decay to all validators (1 block elapsed)
        let validator_addrs: Vec<Address> = self.validators.keys().cloned().collect();
        for addr in &validator_addrs {
            if let Some(v) = self.validators.get_mut(addr) {
                v.credibility =
                    consensus::apply_credibility_decay(v.credibility, 1, v.stake);
            }
            // Sync account credibility with validator credibility
            if let Some(v_cred) = self.validators.get(addr).map(|v| v.credibility) {
                if let Some(acct) = self.accounts.get_mut(addr) {
                    acct.credibility = v_cred;
                }
            }
        }

        // Reward block proposer
        if let Some(acct) = self.accounts.get_mut(&block.header.proposer) {
            acct.balance += consensus::BASE_BLOCK_REWARD;
            acct.credibility += consensus::calculate_proposer_credibility();
        }
        if let Some(validator) = self.validators.get_mut(&block.header.proposer) {
            validator.credibility += consensus::calculate_proposer_credibility();
        }

        // Store block and advance height
        self.current_height = height;
        self.blocks.push(block.clone());

        Ok(())
    }

    /// Create and apply an empty block advanced by the next proposer.
    /// Useful for simulating block progression in Phase 1.
    pub fn mine_empty_block(&mut self) -> Result<Block> {
        let height = self.current_height + 1;
        let prev_hash = self
            .blocks
            .last()
            .map(|b| b.block_hash())
            .unwrap_or_else(Hash::zero);

        let top = self.get_top_validators(height);
        let proposer = consensus::select_block_proposer(&top, height)
            .or_else(|| self.validators.keys().next().cloned())
            .or_else(|| self.accounts.keys().next().cloned())
            .ok_or(ChainError::NoEligibleValidators)?;

        let cred = self.get_credibility(&proposer);

        let block = Block::new(
            prev_hash,
            height,
            current_timestamp(),
            proposer,
            cred,
            BlockBody::empty(),
            Hash::zero(),
            Hash::zero(),
        );

        self.apply_block(&block)?;
        Ok(block)
    }

    /// Create and apply a block containing the given transactions.
    pub fn mine_block_with(
        &mut self,
        transactions: Vec<Tx>,
        data_transactions: Vec<DTx>,
        reveal_transactions: Vec<RTx>,
        proof_transactions: Vec<PTx>,
    ) -> Result<Block> {
        let height = self.current_height + 1;
        let prev_hash = self
            .blocks
            .last()
            .map(|b| b.block_hash())
            .unwrap_or_else(Hash::zero);

        let top = self.get_top_validators(height);
        let proposer = consensus::select_block_proposer(&top, height)
            .or_else(|| self.validators.keys().next().cloned())
            .or_else(|| self.accounts.keys().next().cloned())
            .ok_or(ChainError::NoEligibleValidators)?;

        let cred = self.get_credibility(&proposer);

        let body = BlockBody {
            transactions,
            data_transactions,
            reveal_transactions,
            proof_transactions,
        };

        let block = Block::new(
            prev_hash,
            height,
            current_timestamp(),
            proposer,
            cred,
            body,
            Hash::zero(),
            Hash::zero(),
        );

        self.apply_block(&block)?;
        Ok(block)
    }

    // -----------------------------------------------------------------------
    // Summary / Display helpers
    // -----------------------------------------------------------------------

    /// Return a summary of the chain state for display.
    pub fn summary(&self) -> ChainSummary {
        ChainSummary {
            block_height: self.current_height,
            total_accounts: self.accounts.len(),
            total_validators: self.validators.len(),
            total_ddtxos: self.ddtxos.len(),
            pending_ddtxos: self
                .ddtxos
                .values()
                .filter(|d| d.status == DDTXOStatus::Pending)
                .count(),
            approved_ddtxos: self
                .ddtxos
                .values()
                .filter(|d| d.status == DDTXOStatus::Approved)
                .count(),
            total_supply: self.accounts.values().map(|a| a.balance).sum(),
            categories: self.model_checkpoints.keys().map(|k| k.0.clone()).collect(),
        }
    }
}

/// Summary of chain state for CLI display.
#[derive(Debug, Serialize)]
pub struct ChainSummary {
    pub block_height: u64,
    pub total_accounts: usize,
    pub total_validators: usize,
    pub total_ddtxos: usize,
    pub pending_ddtxos: usize,
    pub approved_ddtxos: usize,
    pub total_supply: u64,
    pub categories: Vec<String>,
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Return a simple current timestamp in milliseconds.
pub fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::{MockProofSystem, ProvingKey};
    use crate::transaction::{DTx, PTx, ProofPublicInputs, RTx, Tx};
    use crate::types::{CategoryId, Keypair, CID};

    /// Setup helper that creates a test chain with MockProofSystem.
    /// Returns (state, all_keys, proving_keys) where all_keys[0] is the contributor.
    fn setup_test_state() -> (ChainState, Vec<Keypair>, HashMap<Address, ProvingKey>) {
        let contributor = Keypair::generate();
        let mut validators: Vec<Keypair> = (0..12).map(|_| Keypair::generate()).collect();

        let proof_system = Arc::new(MockProofSystem::new());

        let mut proving_keys = HashMap::new();
        let mut verification_keys = HashMap::new();
        for v in &validators {
            let pk = MockProofSystem::generate_proving_key();
            let vk = proof_system.derive_verification_key(&pk);
            proving_keys.insert(v.address(), pk);
            verification_keys.insert(v.address(), vk);
        }

        let genesis_accounts = vec![(contributor.address(), 10_000)];
        let genesis_validators: Vec<(Address, u64)> =
            validators.iter().map(|v| (v.address(), 2000)).collect();
        let categories = vec![(CategoryId("defi_credit_v1".into()), 0.70)];

        let state = ChainState::initialize_genesis(
            genesis_accounts,
            genesis_validators,
            categories,
            proof_system,
            verification_keys,
        );

        let mut all_keys = vec![contributor];
        all_keys.append(&mut validators);
        (state, all_keys, proving_keys)
    }

    /// Generate a valid proof for a validator using the MockProofSystem.
    fn generate_proof(
        proof_system: &dyn ProofSystem,
        proving_keys: &HashMap<Address, ProvingKey>,
        validator: &Address,
        inputs: &ProofPublicInputs,
    ) -> Vec<u8> {
        let pk = proving_keys.get(validator).expect("proving key must exist");
        proof_system.generate_proof(pk, validator, inputs)
    }

    #[test]
    fn test_genesis_state() {
        let (state, keys, _pks) = setup_test_state();
        assert_eq!(state.current_height, 0);
        assert_eq!(state.blocks.len(), 1);
        assert_eq!(state.get_balance(&keys[0].address()), 10_000);
        assert_eq!(state.validators.len(), 12);
        // Validators should have stake set
        for v in state.validators.values() {
            assert_eq!(v.stake, consensus::GENESIS_VALIDATOR_STAKE);
        }
    }

    #[test]
    fn test_transfer() {
        let (mut state, keys, _pks) = setup_test_state();
        let sender = &keys[0];
        let recipient = &keys[1];

        let mut tx =
            Tx::new(sender.address(), recipient.address(), 500, 1, 0, current_timestamp());
        tx.sign(sender);

        state.apply_tx(&tx).expect("transfer should succeed");
        assert_eq!(state.get_balance(&sender.address()), 10_000 - 501);
        assert_eq!(state.get_balance(&recipient.address()), 500);
    }

    #[test]
    fn test_transfer_insufficient_balance() {
        let (mut state, keys, _pks) = setup_test_state();
        let sender = &keys[0];
        let recipient = &keys[1];

        let mut tx = Tx::new(
            sender.address(),
            recipient.address(),
            20_000,
            1,
            0,
            current_timestamp(),
        );
        tx.sign(sender);

        let result = state.apply_tx(&tx);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtx_creates_ddtxo() {
        let (mut state, keys, _pks) = setup_test_state();
        let contributor = &keys[0];

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"training_data"),
            CID("QmEncrypted123".into()),
            Hash::compute(b"secret_key"),
            1000,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);

        let dtx_hash = dtx.hash();
        state.apply_dtx(&dtx, 1).expect("DTx should succeed");

        assert_eq!(state.get_balance(&contributor.address()), 10_000 - 100);
        let ddtxo = state.ddtxos.get(&dtx_hash).expect("DDTXO should exist");
        assert_eq!(ddtxo.status, DDTXOStatus::Pending);
        assert_eq!(ddtxo.reveal_deadline, 11);
        assert_eq!(ddtxo.validation_deadline, 26);
    }

    #[test]
    fn test_reveal_valid_key() {
        let (mut state, keys, _pks) = setup_test_state();
        let contributor = &keys[0];
        let secret_key = b"my_secret_encryption_key_32bytes!";

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"data"),
            CID("QmEncrypted".into()),
            Hash::compute(secret_key),
            100,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();
        state.apply_dtx(&dtx, 1).expect("DTx OK");

        let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
        rtx.sign(contributor);
        state.apply_rtx(&rtx, 5).expect("RTx should succeed");

        let ddtxo = state.ddtxos.get(&dtx_hash).expect("DDTXO exists");
        assert_eq!(ddtxo.status, DDTXOStatus::Revealed);
        assert!(ddtxo.decryption_key.is_some());
    }

    #[test]
    fn test_reveal_wrong_key_bans_contributor() {
        let (mut state, keys, _pks) = setup_test_state();
        let contributor = &keys[0];
        let secret_key = b"correct_key_here_32_bytes_long!!";

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"data"),
            CID("QmEncrypted".into()),
            Hash::compute(secret_key),
            100,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();
        state.apply_dtx(&dtx, 1).expect("DTx OK");

        let mut rtx = RTx::new(dtx_hash.clone(), b"wrong_key".to_vec(), contributor.address());
        rtx.sign(contributor);
        let result = state.apply_rtx(&rtx, 5);
        assert!(result.is_err());

        let ddtxo = state.ddtxos.get(&dtx_hash).expect("DDTXO exists");
        assert_eq!(ddtxo.status, DDTXOStatus::Invalid);
        assert!(state.banned_addresses.contains(&contributor.address()));
    }

    #[test]
    fn test_ddtxo_expiration() {
        let (mut state, keys, _pks) = setup_test_state();
        let contributor = &keys[0];

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"data"),
            CID("QmTest".into()),
            Hash::compute(b"key"),
            100,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();
        state.apply_dtx(&dtx, 1).expect("DTx OK");

        state.process_ddtxo_deadlines(12);

        let ddtxo = state.ddtxos.get(&dtx_hash).expect("DDTXO exists");
        assert_eq!(ddtxo.status, DDTXOStatus::Expired);
    }

    #[test]
    fn test_top_validators() {
        let (state, _keys, _pks) = setup_test_state();
        let top = state.get_top_validators(1);
        assert_eq!(top.len(), 10); // 12 validators, max 10
    }

    #[test]
    fn test_mine_empty_block() {
        let (mut state, _keys, _pks) = setup_test_state();
        let block = state.mine_empty_block().expect("mine should succeed");
        assert_eq!(block.header.block_height, 1);
        assert_eq!(state.current_height, 1);
        assert_eq!(state.blocks.len(), 2);
    }

    #[test]
    fn test_full_approval_lifecycle() {
        let (mut state, keys, proving_keys) = setup_test_state();
        let contributor = &keys[0];
        let secret_key = b"encryption_key_32bytes_exactly!!";

        // 1. Submit DTx at block 1
        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"training_data"),
            CID("QmEncrypted".into()),
            Hash::compute(secret_key),
            500,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();

        state
            .mine_block_with(vec![], vec![dtx], vec![], vec![])
            .expect("block with DTx");

        assert_eq!(state.get_balance(&contributor.address()), 10_000 - 100);

        // 2. Mine empty blocks to reach reveal window
        for _ in 0..9 {
            state.mine_empty_block().expect("mine empty");
        }

        // 3. Reveal at block 11
        let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
        rtx.sign(contributor);
        state
            .mine_block_with(vec![], vec![], vec![rtx], vec![])
            .expect("block with RTx");

        let ddtxo = state.ddtxos.get(&dtx_hash).expect("exists");
        assert_eq!(ddtxo.status, DDTXOStatus::Revealed);

        // 4. Submit 7 PTx with real proofs from eligible validators
        let checkpoint = state
            .model_checkpoints
            .get(&CategoryId("defi_credit_v1".into()))
            .expect("category exists")
            .clone();

        let next_height = state.current_height + 1;
        let top10 = state.get_top_validators(next_height);
        let eligible_keys: Vec<&Keypair> = keys
            .iter()
            .filter(|k| top10.contains(&k.address()))
            .take(7)
            .collect();
        assert!(eligible_keys.len() >= 7, "Need at least 7 eligible validators");

        let mut proof_txs = Vec::new();
        for (i, validator) in eligible_keys.iter().enumerate() {
            let inputs = ProofPublicInputs {
                data_hash: Hash::compute(b"training_data"),
                prev_model_hash: checkpoint.model_hash.clone(),
                new_model_hash: Hash::compute(format!("new_model_{i}").as_bytes()),
                prev_accuracy: 0.70,
                new_accuracy: 0.75,
                test_set_hash: Hash::compute(b"test_set"),
            };
            let proof_bytes = generate_proof(
                state.proof_system.as_ref(),
                &proving_keys,
                &validator.address(),
                &inputs,
            );
            let mut ptx = PTx::new(
                validator.address(),
                dtx_hash.clone(),
                proof_bytes,
                inputs,
                CID(format!("QmNewWeights_{i}")),
                current_timestamp(),
            );
            ptx.sign(validator);
            proof_txs.push(ptx);
        }

        state
            .mine_block_with(vec![], vec![], vec![], proof_txs)
            .expect("block with PTx");

        // 5. Mine to finalization
        while state.current_height < 26 {
            state.mine_empty_block().expect("mine");
        }

        // 6. Check DDTXO is APPROVED
        let ddtxo = state.ddtxos.get(&dtx_hash).expect("exists");
        assert_eq!(ddtxo.status, DDTXOStatus::Approved);
        assert!(ddtxo.final_accuracy.is_some());

        let contributor_balance = state.get_balance(&contributor.address());
        assert!(
            contributor_balance > 10_000,
            "Contributor should profit: {contributor_balance}"
        );
    }

    #[test]
    fn test_rejection_no_improvement() {
        let (mut state, keys, proving_keys) = setup_test_state();
        let contributor = &keys[0];
        let secret_key = b"encryption_key_32bytes_exactly!!";

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"bad_data"),
            CID("QmEncrypted".into()),
            Hash::compute(secret_key),
            100,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();

        state
            .mine_block_with(vec![], vec![dtx], vec![], vec![])
            .expect("ok");
        for _ in 0..9 {
            state.mine_empty_block().expect("mine");
        }

        let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
        rtx.sign(contributor);
        state
            .mine_block_with(vec![], vec![], vec![rtx], vec![])
            .expect("ok");

        let checkpoint = state
            .model_checkpoints
            .get(&CategoryId("defi_credit_v1".into()))
            .unwrap()
            .clone();

        let next_height = state.current_height + 1;
        let top10 = state.get_top_validators(next_height);
        let eligible_keys: Vec<&Keypair> = keys
            .iter()
            .filter(|k| top10.contains(&k.address()))
            .take(7)
            .collect();

        let mut proof_txs = Vec::new();
        for validator in &eligible_keys {
            let inputs = ProofPublicInputs {
                data_hash: Hash::compute(b"bad_data"),
                prev_model_hash: checkpoint.model_hash.clone(),
                new_model_hash: Hash::compute(b"same_model"),
                prev_accuracy: 0.70,
                new_accuracy: 0.70, // no improvement
                test_set_hash: Hash::compute(b"test_set"),
            };
            let proof_bytes = generate_proof(
                state.proof_system.as_ref(),
                &proving_keys,
                &validator.address(),
                &inputs,
            );
            let mut ptx = PTx::new(
                validator.address(),
                dtx_hash.clone(),
                proof_bytes,
                inputs,
                CID("QmSame".into()),
                current_timestamp(),
            );
            ptx.sign(validator);
            proof_txs.push(ptx);
        }

        state
            .mine_block_with(vec![], vec![], vec![], proof_txs)
            .expect("ok");
        while state.current_height < 26 {
            state.mine_empty_block().expect("mine");
        }

        let ddtxo = state.ddtxos.get(&dtx_hash).expect("exists");
        assert_eq!(ddtxo.status, DDTXOStatus::Rejected);

        let balance = state.get_balance(&contributor.address());
        assert!(balance < 10_000, "Fee should be forfeited: balance={balance}");
    }

    #[test]
    fn test_proof_verification_rejects_bad_proof() {
        let (mut state, keys, _pks) = setup_test_state();
        let contributor = &keys[0];
        let secret_key = b"encryption_key_32bytes_exactly!!";

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"data"),
            CID("QmTest".into()),
            Hash::compute(secret_key),
            100,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();

        state
            .mine_block_with(vec![], vec![dtx], vec![], vec![])
            .expect("ok");
        for _ in 0..9 {
            state.mine_empty_block().expect("mine");
        }

        let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
        rtx.sign(contributor);
        state
            .mine_block_with(vec![], vec![], vec![rtx], vec![])
            .expect("ok");

        let checkpoint = state
            .model_checkpoints
            .get(&CategoryId("defi_credit_v1".into()))
            .unwrap()
            .clone();

        // Try to submit a PTx with FAKE proof bytes
        let next_height = state.current_height + 1;
        let top10 = state.get_top_validators(next_height);
        let validator = keys.iter().find(|k| top10.contains(&k.address())).unwrap();

        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"data"),
            prev_model_hash: checkpoint.model_hash.clone(),
            new_model_hash: Hash::compute(b"new"),
            prev_accuracy: 0.70,
            new_accuracy: 0.80,
            test_set_hash: Hash::compute(b"test_set"),
        };
        let mut ptx = PTx::new(
            validator.address(),
            dtx_hash.clone(),
            vec![0xDE, 0xAD, 0xBE, 0xEF], // fake proof
            inputs,
            CID("QmFake".into()),
            current_timestamp(),
        );
        ptx.sign(validator);

        let result = state.mine_block_with(vec![], vec![], vec![], vec![ptx]);
        assert!(result.is_err(), "Fake proof should be rejected");
    }

    #[test]
    fn test_accuracy_validation_rejects_nan() {
        let (mut state, keys, proving_keys) = setup_test_state();
        let contributor = &keys[0];
        let secret_key = b"encryption_key_32bytes_exactly!!";

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("defi_credit_v1".into()),
            Hash::compute(b"data"),
            CID("QmTest".into()),
            Hash::compute(secret_key),
            100,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();

        state
            .mine_block_with(vec![], vec![dtx], vec![], vec![])
            .expect("ok");
        for _ in 0..9 {
            state.mine_empty_block().expect("mine");
        }

        let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
        rtx.sign(contributor);
        state
            .mine_block_with(vec![], vec![], vec![rtx], vec![])
            .expect("ok");

        let checkpoint = state
            .model_checkpoints
            .get(&CategoryId("defi_credit_v1".into()))
            .unwrap()
            .clone();

        let next_height = state.current_height + 1;
        let top10 = state.get_top_validators(next_height);
        let validator = keys.iter().find(|k| top10.contains(&k.address())).unwrap();

        // Submit PTx with NaN accuracy
        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"data"),
            prev_model_hash: checkpoint.model_hash.clone(),
            new_model_hash: Hash::compute(b"new"),
            prev_accuracy: 0.70,
            new_accuracy: f64::NAN,
            test_set_hash: Hash::compute(b"test_set"),
        };
        let proof_bytes = generate_proof(
            state.proof_system.as_ref(),
            &proving_keys,
            &validator.address(),
            &inputs,
        );
        let mut ptx = PTx::new(
            validator.address(),
            dtx_hash.clone(),
            proof_bytes,
            inputs,
            CID("QmNaN".into()),
            current_timestamp(),
        );
        ptx.sign(validator);

        let result = state.mine_block_with(vec![], vec![], vec![], vec![ptx]);
        assert!(result.is_err(), "NaN accuracy should be rejected");
    }

    #[test]
    fn test_unsupported_category_rejected() {
        let (mut state, keys, proving_keys) = setup_test_state();
        let contributor = &keys[0];
        let secret_key = b"encryption_key_32bytes_exactly!!";

        // Add a new category that no validator supports
        state.model_checkpoints.insert(
            CategoryId("bio_medical_v1".into()),
            ModelCheckpoint {
                category_id: CategoryId("bio_medical_v1".into()),
                model_hash: Hash::compute(b"bio_model"),
                weights_cid: CID("genesis_bio".into()),
                accuracy: 0.50,
                last_updated_block: 0,
            },
        );

        let mut dtx = DTx::new(
            contributor.address(),
            CategoryId("bio_medical_v1".into()),
            Hash::compute(b"bio_data"),
            CID("QmBio".into()),
            Hash::compute(secret_key),
            100,
            100,
            current_timestamp(),
        );
        dtx.sign(contributor);
        let dtx_hash = dtx.hash();

        state
            .mine_block_with(vec![], vec![dtx], vec![], vec![])
            .expect("ok");
        for _ in 0..9 {
            state.mine_empty_block().expect("mine");
        }

        let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
        rtx.sign(contributor);
        state
            .mine_block_with(vec![], vec![], vec![rtx], vec![])
            .expect("ok");

        let checkpoint = state
            .model_checkpoints
            .get(&CategoryId("bio_medical_v1".into()))
            .unwrap()
            .clone();

        let next_height = state.current_height + 1;
        let top10 = state.get_top_validators(next_height);
        let validator = keys.iter().find(|k| top10.contains(&k.address())).unwrap();

        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"bio_data"),
            prev_model_hash: checkpoint.model_hash.clone(),
            new_model_hash: Hash::compute(b"new_bio"),
            prev_accuracy: 0.50,
            new_accuracy: 0.60,
            test_set_hash: Hash::compute(b"test_set"),
        };
        let proof_bytes = generate_proof(
            state.proof_system.as_ref(),
            &proving_keys,
            &validator.address(),
            &inputs,
        );
        let mut ptx = PTx::new(
            validator.address(),
            dtx_hash.clone(),
            proof_bytes,
            inputs,
            CID("QmBioWeights".into()),
            current_timestamp(),
        );
        ptx.sign(validator);

        let result = state.mine_block_with(vec![], vec![], vec![], vec![ptx]);
        assert!(result.is_err(), "Unsupported category should be rejected");
    }

    #[test]
    fn test_credibility_decay_applied_per_block() {
        let (mut state, _keys, _pks) = setup_test_state();

        // Record initial credibility
        let initial_cred: u64 = state.validators.values().map(|v| v.credibility).sum();

        // Mine some blocks
        for _ in 0..10 {
            state.mine_empty_block().expect("mine");
        }

        // Credibility should be slightly reduced (or same due to rounding) + proposer rewards
        // At credibility 2000 with 1 block decay, the decay rounds to 0
        // But proposer gains +20 per block. So total should increase.
        let final_cred: u64 = state.validators.values().map(|v| v.credibility).sum();
        assert!(
            final_cred >= initial_cred,
            "Credibility should not decrease with proposer rewards: initial={initial_cred}, final={final_cred}"
        );
    }
}
