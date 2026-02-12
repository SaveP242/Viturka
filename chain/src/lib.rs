//! Viturka Chain — core blockchain library for the ZKML-native protocol.
//!
//! This crate implements the foundational data structures and logic for
//! the Viturka blockchain: transactions, blocks, consensus (Proof of Credibility),
//! and in-memory state management.

pub mod block;
pub mod consensus;
pub mod error;
pub mod proof;
pub mod state;
pub mod transaction;
pub mod types;
