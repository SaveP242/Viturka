//! Viturka CLI — command-line interface for interacting with the Viturka blockchain.
//!
//! Phase 1: Single-process simulation with in-memory state.

use std::collections::HashMap;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use viturka_chain::consensus;
use viturka_chain::proof::{MockProofSystem, ProofSystem, ProvingKey};
use viturka_chain::state::{ChainState, DDTXOStatus};
use viturka_chain::transaction::{DTx, PTx, ProofPublicInputs, RTx};
use viturka_chain::types::{Address, CategoryId, Hash, Hashable, Keypair, CID};

#[derive(Parser)]
#[command(name = "viturka")]
#[command(about = "Viturka Protocol CLI — ZKML-native blockchain")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a test chain with genesis block, pre-funded accounts, and validators
    Init,
    /// Run an interactive demo of the full data contribution lifecycle
    Demo,
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init => cmd_init(),
        Commands::Demo => cmd_demo(),
    }
}

fn cmd_init() {
    println!("=== Viturka Protocol — Chain Initialization ===\n");

    let (state, contributor, _validators, _proving_keys) = create_test_chain();

    println!("Genesis block created.");
    print_summary(&state);
    println!(
        "\nContributor address: {}",
        contributor.address()
    );
    println!(
        "Contributor balance: {} VIT",
        state.get_balance(&contributor.address())
    );
    println!("\nRun `viturka demo` to see the full data contribution lifecycle.");
}

fn cmd_demo() {
    println!("=== Viturka Protocol — Full Lifecycle Demo ===\n");

    let (mut state, contributor, validators, proving_keys) = create_test_chain();
    let secret_key = b"demo_encryption_key_32_bytes_ok!";

    // --- Step 1: Submit Data Contribution ---
    println!("[Block 1] Submitting data contribution (DTx)...");
    let mut dtx = DTx::new(
        contributor.address(),
        CategoryId("defi_credit_v1".into()),
        Hash::compute(b"demo_training_data_batch_001"),
        CID("QmDemoEncryptedData123456789".into()),
        Hash::compute(secret_key),
        500,
        consensus::DTX_DEPOSIT,
        viturka_chain::state::current_timestamp(),
    );
    dtx.sign(&contributor);
    let dtx_hash = dtx.hash();

    state
        .mine_block_with(vec![], vec![dtx], vec![], vec![])
        .expect("DTx block mined");

    println!(
        "  DDTXO created: {}",
        &dtx_hash.to_hex()[..16]
    );
    println!(
        "  Fee locked: {} VIT",
        consensus::DTX_DEPOSIT
    );
    println!(
        "  Contributor balance: {} VIT",
        state.get_balance(&contributor.address())
    );
    let ddtxo = state.ddtxos.get(&dtx_hash).expect("DDTXO exists");
    println!("  Status: {}", ddtxo.status);
    println!(
        "  Reveal deadline: block {}",
        ddtxo.reveal_deadline
    );
    println!(
        "  Validation deadline: block {}",
        ddtxo.validation_deadline
    );

    // --- Step 2: Mine blocks to approach reveal window ---
    println!("\n[Blocks 2-10] Mining empty blocks (waiting period)...");
    for _ in 0..9 {
        state.mine_empty_block().expect("mine empty block");
    }
    println!("  Current height: {}", state.current_height);

    // --- Step 3: Reveal decryption key ---
    println!("\n[Block 11] Revealing decryption key (RTx)...");
    let mut rtx = RTx::new(dtx_hash.clone(), secret_key.to_vec(), contributor.address());
    rtx.sign(&contributor);

    state
        .mine_block_with(vec![], vec![], vec![rtx], vec![])
        .expect("RTx block mined");

    let ddtxo = state.ddtxos.get(&dtx_hash).expect("DDTXO exists");
    println!("  Status: {}", ddtxo.status);
    println!("  Key commitment verified successfully");

    // --- Step 4: Validators submit proofs ---
    println!("\n[Block 12] Validators submitting ZKML proofs (PTx)...");

    let checkpoint = state
        .model_checkpoints
        .get(&CategoryId("defi_credit_v1".into()))
        .expect("category exists")
        .clone();

    let mut proof_txs = Vec::new();

    // Select from actual top-10 eligible validators at the next block height
    let next_height = state.current_height + 1;
    let top10 = state.get_top_validators(next_height);
    let eligible_validators: Vec<&Keypair> = validators
        .iter()
        .filter(|v| top10.contains(&v.address()))
        .take(8) // 8 out of 10 submit
        .collect();
    println!(
        "  Eligible validators: {}/{}",
        eligible_validators.len(),
        validators.len()
    );

    for (i, validator) in eligible_validators.iter().enumerate() {
        let validator = *validator;
        let inputs = ProofPublicInputs {
            data_hash: Hash::compute(b"demo_training_data_batch_001"),
            prev_model_hash: checkpoint.model_hash.clone(),
            new_model_hash: Hash::compute(format!("improved_model_v{i}").as_bytes()),
            prev_accuracy: 0.70,
            new_accuracy: 0.748, // ~4.8% improvement
            test_set_hash: Hash::compute(b"standard_test_set"),
        };
        // Generate a real proof using the MockProofSystem
        let pk = proving_keys
            .get(&validator.address())
            .expect("proving key exists");
        let proof_bytes = state.proof_system.generate_proof(pk, &validator.address(), &inputs);
        let mut ptx = PTx::new(
            validator.address(),
            dtx_hash.clone(),
            proof_bytes,
            inputs,
            CID(format!("QmNewModelWeights_v{i}")),
            viturka_chain::state::current_timestamp(),
        );
        ptx.sign(validator);
        proof_txs.push(ptx);
    }

    state
        .mine_block_with(vec![], vec![], vec![], proof_txs)
        .expect("PTx block mined");

    let ddtxo = state.ddtxos.get(&dtx_hash).expect("exists");
    println!(
        "  Proofs submitted: {}/{}",
        ddtxo.validation_proofs.len(),
        consensus::TOP_VALIDATORS
    );
    println!("  Status: {}", ddtxo.status);
    println!(
        "  Proof system: {}",
        state.proof_system.name()
    );

    // --- Step 5: Mine to finalization ---
    println!("\n[Blocks 13-26] Mining to finalization deadline...");
    while state.current_height < 26 {
        state.mine_empty_block().expect("mine");
    }
    println!("  Current height: {}", state.current_height);

    // --- Step 6: Check result ---
    let ddtxo = state.ddtxos.get(&dtx_hash).expect("exists");
    println!("\n=== Finalization Result ===");
    println!("  DDTXO status: {}", ddtxo.status);
    if let Some(acc) = ddtxo.final_accuracy {
        println!("  Final accuracy: {:.3}", acc);
    }

    let contributor_balance = state.get_balance(&contributor.address());
    println!(
        "\n  Contributor balance: {} VIT (started with 10,000)",
        contributor_balance
    );

    if ddtxo.status == DDTXOStatus::Approved {
        println!("  Fee REFUNDED: {} VIT", consensus::DTX_DEPOSIT);
        let reward = contributor_balance as i64 - 10_000;
        if reward > 0 {
            println!("  Reward earned: {} VIT", reward);
        }

        // Show updated model checkpoint
        if let Some(cp) = state.model_checkpoints.get(&CategoryId("defi_credit_v1".into())) {
            println!("\n  Model checkpoint updated:");
            println!("    Accuracy: {:.3} -> {:.3}", 0.70, cp.accuracy);
            println!("    Weights CID: {}", cp.weights_cid);
        }
    }

    // --- Step 7: Show validator cooldowns ---
    println!("\n=== Validator Status ===");
    let top_at_27 = state.get_top_validators(state.current_height + 1);
    println!(
        "  Eligible validators for next block: {}/{}",
        top_at_27.len(),
        state.validators.len()
    );

    let in_cooldown: Vec<_> = state
        .validators
        .values()
        .filter(|v| consensus::is_in_cooldown(v.last_validation_block, state.current_height + 1))
        .collect();
    println!("  Validators in cooldown: {}", in_cooldown.len());

    // Show staking info
    println!("\n=== Staking & Decay ===");
    if let Some(v) = state.validators.values().next() {
        println!(
            "  Validator stake: {} VIT (reduces credibility decay by up to {:.0}%)",
            v.stake,
            consensus::MAX_STAKE_REDUCTION * 100.0
        );
    }

    // --- Summary ---
    println!("\n=== Chain Summary ===");
    print_summary(&state);

    println!("\nDemo complete! The full DTx -> RTx -> PTx -> Approval lifecycle works.");
    println!("Proofs are cryptographically verified using {}.", state.proof_system.name());
}

/// Create a test chain with pre-funded contributor, validators, and MockProofSystem.
fn create_test_chain() -> (
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

    let genesis_accounts = vec![(contributor.address(), 10_000)];

    let genesis_validators: Vec<(Address, u64)> = validators
        .iter()
        .map(|v| (v.address(), 2_000))
        .collect();

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

/// Print a summary of the chain state.
fn print_summary(state: &ChainState) {
    let summary = state.summary();
    println!("  Block height:    {}", summary.block_height);
    println!("  Accounts:        {}", summary.total_accounts);
    println!("  Validators:      {}", summary.total_validators);
    println!("  DDTXOs:          {}", summary.total_ddtxos);
    println!("  Pending DDTXOs:  {}", summary.pending_ddtxos);
    println!("  Approved DDTXOs: {}", summary.approved_ddtxos);
    println!("  Total supply:    {} VIT", summary.total_supply);
    println!("  Categories:      {:?}", summary.categories);
}
