//! Integration tests for the Viturka blockchain — full lifecycle scenarios.

use std::collections::HashMap;
use std::sync::Arc;

use viturka_chain::consensus;
use viturka_chain::proof::{MockProofSystem, ProofSystem, ProvingKey};
use viturka_chain::state::{ChainState, DDTXOStatus};
use viturka_chain::transaction::{DTx, PTx, ProofPublicInputs, RTx, Tx};
use viturka_chain::types::{Address, CategoryId, Hash, Hashable, Keypair, CID};

/// Create a test chain with 1 contributor, 12 validators, MockProofSystem, and proving keys.
fn setup() -> (
    ChainState,
    Keypair,
    Vec<Keypair>,
    HashMap<Address, ProvingKey>,
) {
    let contributor = Keypair::generate();
    let validators: Vec<Keypair> = (0..12).map(|_| Keypair::generate()).collect();

    let proof_system = Arc::new(MockProofSystem::new());

    let mut proving_keys = HashMap::new();
    let mut verification_keys = HashMap::new();
    for v in &validators {
        let pk = MockProofSystem::generate_proving_key();
        let vk = proof_system.derive_verification_key(&pk);
        proving_keys.insert(v.address(), pk);
        verification_keys.insert(v.address(), vk);
    }

    let genesis_accounts = vec![(contributor.address(), 50_000)];
    let genesis_validators: Vec<(Address, u64)> =
        validators.iter().map(|v| (v.address(), 2_000)).collect();
    let categories = vec![(CategoryId("defi_credit_v1".into()), 0.70)];

    let state = ChainState::initialize_genesis(
        genesis_accounts,
        genesis_validators,
        categories,
        proof_system,
        verification_keys,
    );
    (state, contributor, validators, proving_keys)
}

fn now() -> u64 {
    viturka_chain::state::current_timestamp()
}

/// Get keypairs from the validator list that are in the current top-10.
fn get_eligible_keypairs<'a>(
    state: &ChainState,
    validators: &'a [Keypair],
    count: usize,
) -> Vec<&'a Keypair> {
    let next_height = state.current_height + 1;
    let top10 = state.get_top_validators(next_height);
    validators
        .iter()
        .filter(|v| top10.contains(&v.address()))
        .take(count)
        .collect()
}

/// Generate a valid proof for a validator using the state's proof system.
fn make_proof(
    state: &ChainState,
    proving_keys: &HashMap<Address, ProvingKey>,
    validator: &Address,
    inputs: &ProofPublicInputs,
) -> Vec<u8> {
    let pk = proving_keys.get(validator).expect("proving key must exist");
    state.proof_system.generate_proof(pk, validator, inputs)
}

// ---------------------------------------------------------------------------
// Test: Full approval lifecycle (happy path)
// ---------------------------------------------------------------------------

#[test]
fn test_full_data_contribution_approved() {
    let (mut state, contributor, validators, proving_keys) = setup();
    let secret_key = b"test_encryption_key_for_aes256!!";

    // 1. Submit DTx
    let mut dtx = DTx::new(
        contributor.address(),
        CategoryId("defi_credit_v1".into()),
        Hash::compute(b"high_quality_training_data"),
        CID("QmEncrypted1".into()),
        Hash::compute(secret_key),
        1000,
        consensus::DTX_DEPOSIT,
        now(),
    );
    dtx.sign(&contributor);
    let dtx_hash = dtx.hash();

    state
        .mine_block_with(vec![], vec![dtx], vec![], vec![])
        .expect("DTx included");

    let ddtxo = state.ddtxos.get(&dtx_hash).expect("DDTXO created");
    assert_eq!(ddtxo.status, DDTXOStatus::Pending);
    assert_eq!(state.get_balance(&contributor.address()), 50_000 - 100);

    // 2. Advance to reveal window
    for _ in 0..9 {
        state.mine_empty_block().expect("mine");
    }

    // 3. Reveal
    let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
    rtx.sign(&contributor);
    state
        .mine_block_with(vec![], vec![], vec![rtx], vec![])
        .expect("RTx included");

    assert_eq!(
        state.ddtxos.get(&dtx_hash).unwrap().status,
        DDTXOStatus::Revealed
    );

    // 4. Submit 8 valid proofs with real proof generation
    let checkpoint = state
        .model_checkpoints
        .get(&CategoryId("defi_credit_v1".into()))
        .unwrap()
        .clone();

    let eligible = get_eligible_keypairs(&state, &validators, 8);
    assert!(eligible.len() >= 7, "need >=7 eligible validators");

    let mut proofs = Vec::new();
    for (i, v) in eligible.iter().enumerate() {
        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"high_quality_training_data"),
            prev_model_hash: checkpoint.model_hash.clone(),
            new_model_hash: Hash::compute(format!("model_{i}").as_bytes()),
            prev_accuracy: 0.70,
            new_accuracy: 0.78,
            test_set_hash: Hash::compute(b"test_set"),
        };
        let proof_bytes = make_proof(&state, &proving_keys, &v.address(), &inputs);
        let mut ptx = PTx::new(
            v.address(),
            dtx_hash.clone(),
            proof_bytes,
            inputs,
            CID(format!("QmWeights_{i}")),
            now(),
        );
        ptx.sign(v);
        proofs.push(ptx);
    }

    state
        .mine_block_with(vec![], vec![], vec![], proofs)
        .expect("PTx included");

    // 5. Mine to finalization
    while state.current_height < 26 {
        state.mine_empty_block().expect("mine");
    }

    // 6. Verify approval
    let ddtxo = state.ddtxos.get(&dtx_hash).expect("exists");
    assert_eq!(ddtxo.status, DDTXOStatus::Approved);
    assert_eq!(ddtxo.final_accuracy, Some(0.78));

    let balance = state.get_balance(&contributor.address());
    assert!(balance > 50_000, "Should profit: {balance}");

    let cp = state
        .model_checkpoints
        .get(&CategoryId("defi_credit_v1".into()))
        .unwrap();
    assert!((cp.accuracy - 0.78).abs() < 0.001);

    // Validators who submitted proofs should be in cooldown
    for v in &eligible {
        let info = state.validators.get(&v.address()).unwrap();
        assert!(
            consensus::is_in_cooldown(info.last_validation_block, state.current_height + 1),
            "Validator should be in cooldown"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: Rejection path (data doesn't improve accuracy)
// ---------------------------------------------------------------------------

#[test]
fn test_data_contribution_rejected_no_improvement() {
    let (mut state, contributor, validators, proving_keys) = setup();
    let secret_key = b"key_for_bad_data_contribution!!!";

    let mut dtx = DTx::new(
        contributor.address(),
        CategoryId("defi_credit_v1".into()),
        Hash::compute(b"useless_data"),
        CID("QmBadData".into()),
        Hash::compute(secret_key),
        50,
        consensus::DTX_DEPOSIT,
        now(),
    );
    dtx.sign(&contributor);
    let dtx_hash = dtx.hash();

    state
        .mine_block_with(vec![], vec![dtx], vec![], vec![])
        .unwrap();
    for _ in 0..9 {
        state.mine_empty_block().unwrap();
    }

    let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
    rtx.sign(&contributor);
    state
        .mine_block_with(vec![], vec![], vec![rtx], vec![])
        .unwrap();

    let checkpoint = state
        .model_checkpoints
        .get(&CategoryId("defi_credit_v1".into()))
        .unwrap()
        .clone();
    let eligible = get_eligible_keypairs(&state, &validators, 8);

    let mut proofs = Vec::new();
    for v in &eligible {
        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"useless_data"),
            prev_model_hash: checkpoint.model_hash.clone(),
            new_model_hash: Hash::compute(b"same"),
            prev_accuracy: 0.70,
            new_accuracy: 0.70, // no improvement
            test_set_hash: Hash::compute(b"test_set"),
        };
        let proof_bytes = make_proof(&state, &proving_keys, &v.address(), &inputs);
        let mut ptx = PTx::new(
            v.address(),
            dtx_hash.clone(),
            proof_bytes,
            inputs,
            CID("Qm".into()),
            now(),
        );
        ptx.sign(v);
        proofs.push(ptx);
    }

    state
        .mine_block_with(vec![], vec![], vec![], proofs)
        .unwrap();
    while state.current_height < 26 {
        state.mine_empty_block().unwrap();
    }

    let ddtxo = state.ddtxos.get(&dtx_hash).unwrap();
    assert_eq!(ddtxo.status, DDTXOStatus::Rejected);
    assert!(state.get_balance(&contributor.address()) < 50_000);
}

// ---------------------------------------------------------------------------
// Test: Expiration path (no reveal)
// ---------------------------------------------------------------------------

#[test]
fn test_data_contribution_expired() {
    let (mut state, contributor, _validators, _proving_keys) = setup();
    let secret_key = b"key_that_will_never_be_revealed!";

    let mut dtx = DTx::new(
        contributor.address(),
        CategoryId("defi_credit_v1".into()),
        Hash::compute(b"data"),
        CID("QmExpired".into()),
        Hash::compute(secret_key),
        100,
        consensus::DTX_DEPOSIT,
        now(),
    );
    dtx.sign(&contributor);
    let dtx_hash = dtx.hash();

    state
        .mine_block_with(vec![], vec![dtx], vec![], vec![])
        .unwrap();

    for _ in 0..12 {
        state.mine_empty_block().unwrap();
    }

    let ddtxo = state.ddtxos.get(&dtx_hash).unwrap();
    assert_eq!(ddtxo.status, DDTXOStatus::Expired);
    assert_eq!(state.get_balance(&contributor.address()), 50_000 - 100);
}

// ---------------------------------------------------------------------------
// Test: Ban path (key hash mismatch)
// ---------------------------------------------------------------------------

#[test]
fn test_contributor_banned_on_key_mismatch() {
    let (mut state, contributor, _validators, _proving_keys) = setup();
    let real_key = b"the_real_key_32_bytes_long______";

    let mut dtx = DTx::new(
        contributor.address(),
        CategoryId("defi_credit_v1".into()),
        Hash::compute(b"data"),
        CID("QmFraud".into()),
        Hash::compute(real_key),
        100,
        consensus::DTX_DEPOSIT,
        now(),
    );
    dtx.sign(&contributor);
    let dtx_hash = dtx.hash();

    state
        .mine_block_with(vec![], vec![dtx], vec![], vec![])
        .unwrap();

    let mut rtx = RTx::new(dtx_hash.clone(), b"wrong_key".to_vec(), contributor.address());
    rtx.sign(&contributor);

    let result = state.mine_block_with(vec![], vec![], vec![rtx], vec![]);
    assert!(result.is_err(), "Should fail due to key mismatch");

    let ddtxo = state.ddtxos.get(&dtx_hash).unwrap();
    assert_eq!(ddtxo.status, DDTXOStatus::Invalid);
    assert!(state.banned_addresses.contains(&contributor.address()));
}

// ---------------------------------------------------------------------------
// Test: Multiple transfers
// ---------------------------------------------------------------------------

#[test]
fn test_chain_of_transfers() {
    let (mut state, contributor, validators, _proving_keys) = setup();

    let mut txs = Vec::new();
    for (i, v) in validators.iter().take(3).enumerate() {
        let mut tx = Tx::new(
            contributor.address(),
            v.address(),
            1000,
            1,
            i as u64,
            now(),
        );
        tx.sign(&contributor);
        txs.push(tx);
    }

    state.mine_block_with(txs, vec![], vec![], vec![]).unwrap();

    assert_eq!(
        state.get_balance(&contributor.address()),
        50_000 - 3 * 1001
    );

    for v in validators.iter().take(3) {
        assert_eq!(state.get_balance(&v.address()), 1000);
    }
}

// ---------------------------------------------------------------------------
// Test: Credibility accumulation and top-10 rotation
// ---------------------------------------------------------------------------

#[test]
fn test_validator_cooldown_rotation() {
    let (mut state, _contributor, _validators, _proving_keys) = setup();

    let top = state.get_top_validators(1);
    assert_eq!(top.len(), 10);

    let top_5: Vec<Address> = top.iter().take(5).cloned().collect();
    for addr in &top_5 {
        if let Some(v) = state.validators.get_mut(addr) {
            v.last_validation_block = 1;
        }
    }

    let top_at_2 = state.get_top_validators(2);
    for addr in &top_5 {
        assert!(
            !top_at_2.contains(addr),
            "Validator should be in cooldown at block 2"
        );
    }

    let top_at_12 = state.get_top_validators(12);
    assert_eq!(top_at_12.len(), 10);
}

// ---------------------------------------------------------------------------
// Test: Insufficient proofs leads to rejection
// ---------------------------------------------------------------------------

#[test]
fn test_insufficient_proofs_rejects() {
    let (mut state, contributor, validators, proving_keys) = setup();
    let secret_key = b"key_for_insufficient_proofs_test";

    let mut dtx = DTx::new(
        contributor.address(),
        CategoryId("defi_credit_v1".into()),
        Hash::compute(b"data"),
        CID("QmNotEnough".into()),
        Hash::compute(secret_key),
        100,
        consensus::DTX_DEPOSIT,
        now(),
    );
    dtx.sign(&contributor);
    let dtx_hash = dtx.hash();

    state
        .mine_block_with(vec![], vec![dtx], vec![], vec![])
        .unwrap();
    for _ in 0..9 {
        state.mine_empty_block().unwrap();
    }

    let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
    rtx.sign(&contributor);
    state
        .mine_block_with(vec![], vec![], vec![rtx], vec![])
        .unwrap();

    // Submit only 3 proofs (below CONSENSUS_THRESHOLD of 7)
    let checkpoint = state
        .model_checkpoints
        .get(&CategoryId("defi_credit_v1".into()))
        .unwrap()
        .clone();
    let eligible = get_eligible_keypairs(&state, &validators, 3);

    let mut proofs = Vec::new();
    for v in &eligible {
        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"data"),
            prev_model_hash: checkpoint.model_hash.clone(),
            new_model_hash: Hash::compute(b"improved"),
            prev_accuracy: 0.70,
            new_accuracy: 0.80,
            test_set_hash: Hash::compute(b"test_set"),
        };
        let proof_bytes = make_proof(&state, &proving_keys, &v.address(), &inputs);
        let mut ptx = PTx::new(
            v.address(),
            dtx_hash.clone(),
            proof_bytes,
            inputs,
            CID("Qm".into()),
            now(),
        );
        ptx.sign(v);
        proofs.push(ptx);
    }

    state
        .mine_block_with(vec![], vec![], vec![], proofs)
        .unwrap();
    while state.current_height < 26 {
        state.mine_empty_block().unwrap();
    }

    let ddtxo = state.ddtxos.get(&dtx_hash).unwrap();
    assert_eq!(ddtxo.status, DDTXOStatus::Rejected);
}

// ---------------------------------------------------------------------------
// Test: Fake proof rejected on-chain
// ---------------------------------------------------------------------------

#[test]
fn test_fake_proof_rejected() {
    let (mut state, contributor, validators, _proving_keys) = setup();
    let secret_key = b"key_for_fake_proof_test_session!";

    let mut dtx = DTx::new(
        contributor.address(),
        CategoryId("defi_credit_v1".into()),
        Hash::compute(b"data"),
        CID("QmFakeProof".into()),
        Hash::compute(secret_key),
        100,
        consensus::DTX_DEPOSIT,
        now(),
    );
    dtx.sign(&contributor);
    let dtx_hash = dtx.hash();

    state
        .mine_block_with(vec![], vec![dtx], vec![], vec![])
        .unwrap();
    for _ in 0..9 {
        state.mine_empty_block().unwrap();
    }

    let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
    rtx.sign(&contributor);
    state
        .mine_block_with(vec![], vec![], vec![rtx], vec![])
        .unwrap();

    let checkpoint = state
        .model_checkpoints
        .get(&CategoryId("defi_credit_v1".into()))
        .unwrap()
        .clone();

    let eligible = get_eligible_keypairs(&state, &validators, 1);
    let v = eligible[0];

    let inputs = ProofPublicInputs {
        data_hash: Hash::compute(b"data"),
        prev_model_hash: checkpoint.model_hash.clone(),
        new_model_hash: Hash::compute(b"fake_model"),
        prev_accuracy: 0.70,
        new_accuracy: 0.90,
        test_set_hash: Hash::compute(b"test_set"),
    };
    let mut ptx = PTx::new(
        v.address(),
        dtx_hash.clone(),
        vec![0xFF; 32], // forged proof bytes
        inputs,
        CID("QmFake".into()),
        now(),
    );
    ptx.sign(v);

    let result = state.mine_block_with(vec![], vec![], vec![], vec![ptx]);
    assert!(
        result.is_err(),
        "Forged proof must be rejected by the proof system"
    );
}
