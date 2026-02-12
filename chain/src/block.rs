//! Block structure for the Viturka blockchain.
//!
//! Each block contains a header with metadata and merkle roots,
//! plus a body containing all four transaction types.

use serde::{Deserialize, Serialize};

use crate::transaction::{DTx, PTx, RTx, Tx};
use crate::types::{compute_merkle_root, Address, Hash, Hashable};

// ---------------------------------------------------------------------------
// BlockHeader
// ---------------------------------------------------------------------------

/// Header for a Viturka block. Contains block metadata, merkle roots
/// of transaction lists, and state commitments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Protocol version (1 for Phase 1).
    pub version: u32,
    /// Hash of the previous block header (zero for genesis).
    pub previous_hash: Hash,
    /// Block height (0 for genesis).
    pub block_height: u64,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    /// Address of the block proposer.
    pub proposer: Address,
    /// Credibility score of the proposer at time of proposal.
    pub proposer_credibility: u64,
    /// Merkle root of standard transfer transactions.
    pub merkle_root_tx: Hash,
    /// Merkle root of data contribution transactions.
    pub merkle_root_dtx: Hash,
    /// Merkle root of proof transactions.
    pub merkle_root_ptx: Hash,
    /// Root hash of the chain state after this block.
    pub state_root: Hash,
    /// Hash of the current validator set.
    pub validator_set_hash: Hash,
}

impl Hashable for BlockHeader {
    fn to_hash_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(self.previous_hash.as_bytes());
        bytes.extend_from_slice(&self.block_height.to_le_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(self.proposer.as_bytes());
        bytes.extend_from_slice(&self.proposer_credibility.to_le_bytes());
        bytes.extend_from_slice(self.merkle_root_tx.as_bytes());
        bytes.extend_from_slice(self.merkle_root_dtx.as_bytes());
        bytes.extend_from_slice(self.merkle_root_ptx.as_bytes());
        bytes.extend_from_slice(self.state_root.as_bytes());
        bytes.extend_from_slice(self.validator_set_hash.as_bytes());
        bytes
    }
}

// ---------------------------------------------------------------------------
// BlockBody
// ---------------------------------------------------------------------------

/// Body of a Viturka block containing all transaction types.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockBody {
    /// Standard VIT transfer transactions.
    pub transactions: Vec<Tx>,
    /// Data contribution transactions.
    pub data_transactions: Vec<DTx>,
    /// Decryption key reveal transactions.
    pub reveal_transactions: Vec<RTx>,
    /// ZKML proof transactions.
    pub proof_transactions: Vec<PTx>,
}

impl BlockBody {
    /// Create an empty block body.
    pub fn empty() -> Self {
        Self {
            transactions: Vec::new(),
            data_transactions: Vec::new(),
            reveal_transactions: Vec::new(),
            proof_transactions: Vec::new(),
        }
    }

    /// Compute the merkle root of standard transactions.
    pub fn compute_merkle_root_tx(&self) -> Hash {
        let hashes: Vec<Hash> = self.transactions.iter().map(|tx| tx.hash()).collect();
        compute_merkle_root(&hashes)
    }

    /// Compute the merkle root of data transactions.
    pub fn compute_merkle_root_dtx(&self) -> Hash {
        let hashes: Vec<Hash> = self.data_transactions.iter().map(|dtx| dtx.hash()).collect();
        compute_merkle_root(&hashes)
    }

    /// Compute the merkle root of proof transactions.
    pub fn compute_merkle_root_ptx(&self) -> Hash {
        let hashes: Vec<Hash> = self.proof_transactions.iter().map(|ptx| ptx.hash()).collect();
        compute_merkle_root(&hashes)
    }

    /// Return the total number of transactions across all types.
    pub fn total_transactions(&self) -> usize {
        self.transactions.len()
            + self.data_transactions.len()
            + self.reveal_transactions.len()
            + self.proof_transactions.len()
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// A complete Viturka block (header + body).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub header: BlockHeader,
    pub body: BlockBody,
}

impl Block {
    /// Build a new block from its components, computing merkle roots automatically.
    ///
    /// `state_root` and `validator_set_hash` are provided by the caller
    /// (computed from chain state after applying transactions).
    pub fn new(
        previous_hash: Hash,
        block_height: u64,
        timestamp: u64,
        proposer: Address,
        proposer_credibility: u64,
        body: BlockBody,
        state_root: Hash,
        validator_set_hash: Hash,
    ) -> Self {
        let merkle_root_tx = body.compute_merkle_root_tx();
        let merkle_root_dtx = body.compute_merkle_root_dtx();
        let merkle_root_ptx = body.compute_merkle_root_ptx();

        let header = BlockHeader {
            version: 1,
            previous_hash,
            block_height,
            timestamp,
            proposer,
            proposer_credibility,
            merkle_root_tx,
            merkle_root_dtx,
            merkle_root_ptx,
            state_root,
            validator_set_hash,
        };

        Self { header, body }
    }

    /// Create the genesis block with no transactions.
    pub fn genesis(proposer: Address, timestamp: u64) -> Self {
        Self::new(
            Hash::zero(),
            0,
            timestamp,
            proposer,
            0,
            BlockBody::empty(),
            Hash::zero(), // state root computed later
            Hash::zero(), // validator set hash computed later
        )
    }

    /// Return the hash of this block (hash of the header).
    pub fn block_hash(&self) -> Hash {
        self.header.hash()
    }

    /// Verify that the merkle roots in the header match the body contents.
    pub fn verify_merkle_roots(&self) -> bool {
        self.header.merkle_root_tx == self.body.compute_merkle_root_tx()
            && self.header.merkle_root_dtx == self.body.compute_merkle_root_dtx()
            && self.header.merkle_root_ptx == self.body.compute_merkle_root_ptx()
    }
}

impl Hashable for Block {
    fn to_hash_bytes(&self) -> Vec<u8> {
        self.header.to_hash_bytes()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Keypair;

    fn now() -> u64 {
        1_700_000_000_000
    }

    #[test]
    fn test_genesis_block() {
        let proposer = Keypair::generate();
        let genesis = Block::genesis(proposer.address(), now());

        assert_eq!(genesis.header.block_height, 0);
        assert_eq!(genesis.header.previous_hash, Hash::zero());
        assert_eq!(genesis.header.version, 1);
        assert_eq!(genesis.body.total_transactions(), 0);
        assert!(genesis.verify_merkle_roots());
    }

    #[test]
    fn test_block_hash_deterministic() {
        let proposer = Keypair::generate();
        let block = Block::genesis(proposer.address(), now());

        let h1 = block.block_hash();
        let h2 = block.block_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_block_with_transactions() {
        let proposer = Keypair::generate();
        let sender = Keypair::generate();
        let recipient = Keypair::generate();

        let mut tx = Tx::new(sender.address(), recipient.address(), 50, 1, 0, now());
        tx.sign(&sender);

        let body = BlockBody {
            transactions: vec![tx],
            data_transactions: Vec::new(),
            reveal_transactions: Vec::new(),
            proof_transactions: Vec::new(),
        };

        let block = Block::new(
            Hash::zero(),
            1,
            now(),
            proposer.address(),
            1000,
            body,
            Hash::zero(),
            Hash::zero(),
        );

        assert_eq!(block.header.block_height, 1);
        assert_eq!(block.body.total_transactions(), 1);
        assert!(block.verify_merkle_roots());
        // Merkle root of txs should not be zero (we have 1 tx)
        assert_ne!(block.header.merkle_root_tx, Hash::zero());
        // No data/proof txs, so those roots should be zero
        assert_eq!(block.header.merkle_root_dtx, Hash::zero());
        assert_eq!(block.header.merkle_root_ptx, Hash::zero());
    }

    #[test]
    fn test_different_blocks_different_hashes() {
        let p1 = Keypair::generate();
        let p2 = Keypair::generate();

        let b1 = Block::genesis(p1.address(), now());
        let b2 = Block::genesis(p2.address(), now());

        assert_ne!(b1.block_hash(), b2.block_hash());
    }

    #[test]
    fn test_empty_body() {
        let body = BlockBody::empty();
        assert_eq!(body.total_transactions(), 0);
        assert_eq!(body.compute_merkle_root_tx(), Hash::zero());
        assert_eq!(body.compute_merkle_root_dtx(), Hash::zero());
        assert_eq!(body.compute_merkle_root_ptx(), Hash::zero());
    }

    #[test]
    fn test_block_serialization_roundtrip() {
        let proposer = Keypair::generate();
        let block = Block::genesis(proposer.address(), now());

        let json = serde_json::to_string(&block).expect("serialize");
        let block2: Block = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(block.block_hash(), block2.block_hash());
    }
}
