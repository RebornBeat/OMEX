This is my orchestrator/mod.rs

//! Prompt Orchestrator - v0.4.0
//!
//! Orchestrates the full 11-stage flow from user input to response delivery.
//! This is the CENTRAL COORDINATOR that ties all pipelines together.
//!
//! STAGE 1:  Input Capture (from workspace_tab or meta_portion)
//! STAGE 2:  Text/Prompt Normalization + AMT (text modality + zero-shot)
//! STAGE 3:  Blueprint Assignment (100% match or create new)
//! STAGE 4:  Zero-Shot Simulation (with AMT traversal)
//! STAGE 5:  Consciousness Decision Gate (if enabled)
//! STAGE 6:  Context Aggregation PER STEP (context_aggregation)
//! STAGE 7:  Task Creation (task_manager)
//! STAGE 8:  Execution per blueprint step (with loops, sub-steps, dependencies)
//! STAGE 9:  Result Collection
//! STAGE 10: Post-execution consciousness (experience_memory)
//! STAGE 11: Response Delivery
//!
//! CRITICAL: This orchestrator respects I-Loop protection.
//! Tasks MUST wait for I-Loop to complete before starting.
//!
//! KEY FEATURES:
//! - Layer-by-layer AMT building from chunks (processes each chunk individually)
//! - 5 consecutive Valid validations required
//! - Blueprint step execution with loop/sub-step/dependency support
//! - Direct ZSEI access (no deprecated pipeline wrappers)
//! - Pipeline awareness for blueprint creation
//! - Coverage aspects derived from methodologies (not hardcoded)
//! - Queue-based task execution via TaskManager

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

// Import task module
use crate::task::{RefinementConfig, TaskData, TaskManager, TaskPriority, TaskQueueConfig};

// ============================================================================
// Types
// ============================================================================

/// Orchestration request from UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationRequest {
    /// User's prompt input
    pub prompt: String,
    /// Optional project context (scoped chat)
    pub project_id: Option<u64>,
    /// Optional workspace context
    pub workspace_id: Option<u64>,
    /// User ID for consciousness tracking
    pub user_id: u64,
    /// Device ID
    pub device_id: u64,
    /// Whether consciousness is enabled
    pub consciousness_enabled: bool,
    /// Token budget for context (overrides model default if set)
    pub token_budget: Option<u32>,
    /// Model configuration override
    pub model_config: Option<ModelConfigOverride>,
    /// Files attached to this prompt (paths or inline content)
    #[serde(default)]
    pub attached_files: Vec<AttachedFileSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigOverride {
    pub model_type: Option<String>,
    pub model_identifier: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub context_length: Option<u32>,
}

/// Orchestration response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResponse {
    pub success: bool,
    pub response: Option<String>,
    pub task_id: Option<u64>,
    pub blueprint_id: Option<u64>,
    pub stages_completed: Vec<StageResult>,
    pub consciousness_gate: Option<GateResult>,
    pub error: Option<String>,
    pub total_tokens_used: Option<u32>,
    pub execution_time_ms: u64,
    /// Methodologies used during this request
    pub methodologies_used: Vec<u64>,
    /// Categories created during this request
    pub categories_created: u32,
    /// Blueprints created during this request
    pub blueprints_created: u32,
    /// Clarification points requiring user input
    pub clarification_points: Vec<String>,
    /// Whether clarification is needed before proceeding
    pub needs_clarification: bool,
    /// AMT structure (for debugging/visualization)
    pub amt_summary: Option<AMTSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub stage: u8,
    pub name: String,
    pub success: bool,
    pub duration_ms: u64,
    pub output_summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub decision: String,
    pub confidence: f32,
    pub reasoning: String,
}

/// A file attached to the prompt for processing.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AttachedFileSpec {
    pub file_path: String,
    pub mime_type: Option<String>,
    pub is_inline: bool, // true if content provided inline, false if path only
    pub content_preview: Option<String>, // first 512 bytes if text-readable
}

/// Role of a file graph relative to the prompt intent.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FileGraphRole {
    Primary,       // the thing being worked on
    Supplementary, // provides context/reference
    #[default]
    RawData, // data to be processed
    Unknown,
}

/// Classification result for a file's graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedFileGraph {
    pub file_path: String,
    pub graph_id: u64,
    pub modality: String,
    pub role: FileGraphRole,
    pub reasoning: String,
}

/// Evidence that a modality was detected in a specific chunk span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityEvidence {
    pub chunk_index: u32,
    pub span_start: usize,
    pub span_end: usize,
    pub intent_reference: String,
}

/// A modality with a verified pipeline handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedModality {
    pub modality: String,
    pub pipeline_id: u64,
    pub evidence: Vec<ModalityEvidence>,
}

/// A modality detected in chunks but with no registered handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnhandledModality {
    pub modality: String,
    pub evidence: Vec<ModalityEvidence>,
}

/// Aggregated root modality list produced after all chunks are processed.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RootModalityList {
    pub verified_modalities: Vec<VerifiedModality>,
    pub unhandled_modalities: Vec<UnhandledModality>,
    pub total_chunk_count: u32,
}

/// State of a modality graph through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum ModalityGraphState {
    #[default]
    Created,
    SemanticEnriched,
    CrossLinked,
    Stable,
    Updated,
    Failed,
}

/// Tracks a modality graph's state within this orchestration session.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionGraphState {
    pub graph_id: u64,
    pub modality: String,
    pub pipeline_id: u64,
    pub state: ModalityGraphState,
    pub cross_modal_edge_count: usize,
    pub consecutive_stable_passes: u32,
}

/// Structured evidence assembled from all available graphs for one AMT layer pass.
/// Produced by `gather_layer_input()`, consumed by `build_amt_layer_by_layer()`.
#[derive(Debug, Clone, Default)]
pub struct LayerInput {
    /// Union of all keywords across chunk graphs
    pub keywords: Vec<String>,
    /// Union of all topics across chunk graphs
    pub topics: Vec<String>,
    /// Grammar relationships from all chunks — structured edge evidence
    pub grammar_evidence: Vec<GrammarEvidence>,
    /// Modality span evidence grouped by modality name
    pub modality_span_evidence: HashMap<String, Vec<ModalitySpanEvidence>>,
    /// File context from classified file graphs
    pub file_contexts: Vec<FileLayerContext>,
    /// Modality graph state contexts
    pub graph_contexts: Vec<GraphLayerContext>,
    /// Verified modality names from root_modality_list
    pub verified_modalities: Vec<String>,
    /// Original cleaned prompt text
    pub cleaned_prompt: String,
    /// Number of processed chunks
    pub chunk_count: u32,
    /// Ordered chunk graph IDs for reconstruction
    pub chunk_graph_ids: Vec<u64>,
}

/// Single grammar relationship extracted from a chunk.
#[derive(Debug, Clone)]
pub struct GrammarEvidence {
    pub from_text: String,
    pub to_text: String,
    pub edge_type: String,
    pub chunk_index: u32,
    pub tense: Option<String>,
    pub negated: bool,
}

/// A detected modality span within a chunk.
#[derive(Debug, Clone)]
pub struct ModalitySpanEvidence {
    pub chunk_index: u32,
    pub span_start: usize,
    pub span_end: usize,
    pub intent_reference: String,
}

/// File context contribution to a layer pass.
#[derive(Debug, Clone)]
pub struct FileLayerContext {
    pub file_path: String,
    pub modality: String,
    pub role: String, // "Primary" | "Supplementary" | "RawData"
    pub graph_id: u64,
}

/// Modality graph state at the time of a layer pass.
#[derive(Debug, Clone)]
pub struct GraphLayerContext {
    pub modality: String,
    pub graph_id: u64,
    pub state: String,
    pub cross_modal_edge_count: usize,
    pub pipeline_id: u64,
}

/// Result of ZSEI knowledge enrichment for a layer pass.
#[derive(Debug, Clone, Default)]
pub struct LayerKnowledge {
    /// Methodology IDs newly discovered for this layer
    pub new_methodology_ids: Vec<u64>,
    /// Related blueprint IDs found via keyword search
    pub related_blueprint_ids: Vec<u64>,
    /// Enrichment summaries from methodologies (for prompt injection)
    pub methodology_summaries: Vec<String>,
}

/// Multi-modal synthesis result — cross-modality patterns found in evidence.
#[derive(Debug, Clone, Default)]
pub struct ModalSynthesis {
    /// Modality pairs that share evidence (text references code, text references image, etc.)
    pub cross_modal_pairs: Vec<(String, String, String)>, // (modality_a, modality_b, relationship)
    /// A short narrative summary of what's across modalities — injected into AMT prompts
    pub cross_modal_summary: String,
    /// Modalities that have strong evidence (node_count > 0 or span_evidence.len() > 3)
    pub active_modalities: Vec<String>,
}

/// Quality score for a discovered AMT branch.
#[derive(Debug, Clone)]
pub struct BranchQuality {
    pub branch: String,
    pub evidence_score: f32,    // 0.0–1.0 based on chunk evidence
    pub methodology_score: f32, // 0.0–1.0 based on methodology support
    pub modal_coverage: f32,    // 0.0–1.0 fraction of active modalities it touches
    pub total_score: f32,       // weighted sum
    pub should_prune: bool,     // true if total_score < 0.2
}

/// Orchestrator-level step lifecycle. Distinct from types::task::StepStatus.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum OrchestratorStepStatus {
    #[default]
    Planned,
    WaitingOnDependency,
    Active,
    StageComplete(u32), // completed stage N, more stages remain
    WaitingPostReview,  // stage done, waiting for hook review to finish
    Complete,
    Failed,
    Rollback,
    Cancelled,
}

/// Per-step tracking state maintained by the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OrchestratorStepState {
    pub step_index: u32,
    pub pipeline_id: u64,
    pub pipeline_name: String,
    pub status: OrchestratorStepStatus,
    pub stages_completed: Vec<String>,
    pub stages_pending: Vec<String>,
    pub current_stage: Option<String>,
    pub graph_ids_read: Vec<String>,    // modality names read
    pub graph_ids_updated: Vec<String>, // modality names updated
    pub waiting_on_step_indices: Vec<u32>,
    pub version: u32,
    pub tokens_used: u32,
}

/// Result of a methodology cross-reference check for an AMT layer.
#[derive(Debug, Clone)]
pub enum MethodologyFinding {
    /// An existing methodology was added to the active set for this layer.
    Loaded(u64),
    /// A new methodology was synthesized and stored because none existed.
    Created(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMTSummary {
    pub total_nodes: usize,
    pub branch_count: usize,
    pub max_depth: usize,
    pub validation_status: String,
}

// ============================================================================
// Pipeline Registry Types
// ============================================================================

/// Pipeline info from index.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInfo {
    pub pipeline_id: u64,
    pub name: String,
    pub folder_name: String,
    pub category: String,
    pub description: String,
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub has_ui: bool,
    #[serde(default)]
    pub is_tab: bool,
    #[serde(default)]
    pub deprecated: bool,
}

/// Pipeline index from ZSEI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineIndex {
    pub version: u32,
    pub pipeline_count: u32,
    pub pipelines: Vec<PipelineInfo>,
    #[serde(default)]
    pub categories: HashMap<String, Vec<u64>>,
    #[serde(default)]
    pub next_custom_id: u64,
}

// ============================================================================
// AMT Types - Abstract Meaning Tree
// ============================================================================

/// AMT Node with chunk reference and methodology links
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMTNode {
    pub id: u64,
    pub node_type: AMTNodeType,
    pub content: String,
    pub source_chunk_indices: Vec<u32>,
    pub children: Vec<AMTNode>,
    pub relationships: Vec<AMTRelation>,
    pub methodology_ids: Vec<u64>,
    pub metadata: HashMap<String, String>,
    pub depth: u32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AMTNodeType {
    Root,           // Primary intent/goal
    Branch,         // Major sub-component/requirement
    Leaf,           // Specific detail/constraint
    Consideration,  // Security, edge case, dependency
    CrossReference, // Link to related branch
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMTRelation {
    pub target_id: u64,
    pub relation_type: AMTRelationType,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AMTRelationType {
    DependsOn,
    Requires,
    RelatesTo,
    Contradicts,
    Elaborates,
    SharedContext,
}

impl AMTNode {
    fn new(id: u64, node_type: AMTNodeType, content: String, depth: u32) -> Self {
        Self {
            id,
            node_type,
            content,
            source_chunk_indices: Vec::new(),
            children: Vec::new(),
            relationships: Vec::new(),
            methodology_ids: Vec::new(),
            metadata: HashMap::new(),
            depth,
            confidence: 1.0,
        }
    }

    fn count_nodes(&self) -> usize {
        1 + self.children.iter().map(|c| c.count_nodes()).sum::<usize>()
    }

    fn max_depth(&self) -> usize {
        if self.children.is_empty() {
            self.depth as usize
        } else {
            self.children
                .iter()
                .map(|c| c.max_depth())
                .max()
                .unwrap_or(self.depth as usize)
        }
    }

    fn branch_count(&self) -> usize {
        let own_branches = if self.node_type == AMTNodeType::Branch {
            1
        } else {
            0
        };
        own_branches
            + self
                .children
                .iter()
                .map(|c| c.branch_count())
                .sum::<usize>()
    }
}

// ============================================================================
// Chunk Types (from text modality)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawChunk {
    pub index: u32,
    pub text: String,
    pub start_char: u32,
    pub end_char: u32,
    pub token_count: u32,
    pub is_complete_paragraph: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedChunk {
    pub index: u32,
    pub original_text: String,
    pub cleaned_text: String,
    pub start_offset: u32,
    pub end_offset: u32,
    pub token_count: u32,
    pub keywords: Vec<String>,
    pub entities: Vec<ExtractedEntity>,
    pub topics: Vec<String>,
    pub overlap_from_previous: u32,
    pub overlap_to_next: u32,
    #[serde(default)]
    pub grammar_relationships: Vec<serde_json::Value>,
    #[serde(default)]
    pub detected_modalities: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,
    pub confidence: f32,
}

// ============================================================================
// Blueprint Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintStep {
    pub step_index: u32,
    pub action: String,
    pub description: String,
    pub pipeline_id: u64,
    pub context_requirements: Vec<String>,
    /// Loop configuration
    pub loop_config: Option<LoopConfig>,
    /// Sub-steps within this step
    pub sub_steps: Vec<BlueprintSubStep>,
    /// IDs of steps this depends on
    pub depends_on: Vec<u32>,
    /// Whether to wait for graph update before proceeding
    pub wait_for_graph_update: bool,
    /// Maximum retries on failure
    pub max_retries: u32,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopConfig {
    /// Loop type
    pub loop_type: LoopType,
    /// Maximum iterations (safety limit)
    pub max_iterations: u32,
    /// Condition for continuing (evaluated each iteration)
    pub continue_condition: String,
    /// Variable to iterate over (for ForEach)
    pub iterate_over: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LoopType {
    /// Loop while condition is true
    While,
    /// Loop until condition is true
    Until,
    /// Loop for each item in a collection
    ForEach,
    /// Fixed number of iterations
    Count,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintSubStep {
    pub sub_index: u32,
    pub action: String,
    pub pipeline_id: u64,
    pub input_mapping: HashMap<String, String>,
    pub output_mapping: HashMap<String, String>,
}

// ============================================================================
// Model Context Limits
// ============================================================================

fn get_model_context_limit(model_identifier: &str) -> u32 {
    match model_identifier {
        // Claude models
        s if s.contains("claude-3-opus") => 200000,
        s if s.contains("claude-3-sonnet") => 200000,
        s if s.contains("claude-3-haiku") => 200000,
        s if s.contains("claude-sonnet-4") => 200000,
        s if s.contains("claude-haiku-4") => 200000,
        s if s.contains("claude-opus-4") => 200000,
        // GPT models
        s if s.contains("gpt-4-turbo") => 128000,
        s if s.contains("gpt-4o") => 128000,
        s if s.contains("gpt-4") => 8192,
        s if s.contains("gpt-3.5") => 16385,
        // Local models
        s if s.contains("llama-3") => 8192,
        s if s.contains("llama-2") => 4096,
        s if s.contains("mistral") => 32768,
        s if s.contains("mixtral") => 32768,
        // BitNet models (smaller context)
        s if s.contains("bitnet") => 4096,
        // Default
        _ => 100000,
    }
}

// ============================================================================
// Internal State
// ============================================================================

/// Tracks a discovered intent with provenance
#[derive(Debug, Clone)]
struct IntentCapture {
    intent: String,
    is_parallel: bool, // true if this is an unrelated parallel intent
    source_chunk_indices: Vec<u32>,
    source_sentences: Vec<String>, // exact sentences/paragraphs from chunks
    node_id: u64,                  // assigned when AMT node is created
}

/// Tracks a discovered branch with methodology provenance
#[derive(Debug, Clone)]
struct BranchCapture {
    branch: String,
    parent_intent: String,
    source_methodology_ids: Vec<u64>, // which methodologies suggested this branch
    source_chunk_indices: Vec<u32>,   // chunks that mention this branch
    source_sentences: Vec<String>,
    node_id: u64,
}

/// Tracks a discovered detail/sub-task
#[derive(Debug, Clone)]
struct DetailCapture {
    content: String,
    detail_type: String, // "detail", "requirement", "constraint"
    parent_branch: String,
    parent_intent: String,
    source_chunk_indices: Vec<u32>,
    source_sentences: Vec<String>,
    node_id: u64,
}

/// Tracks cross-references between branches
#[derive(Debug, Clone)]
struct CrossRef {
    from_branch: String,
    to_branch: String,
    from_intent: String,
    to_intent: String,
    relation_type: AMTRelationType,
    description: String,
}

struct OrchestrationState {
    request: OrchestrationRequest,
    start_time: std::time::Instant,
    stages: Vec<StageResult>,

    // Model context management
    model_context_limit: u32,
    tokens_used_so_far: u32,

    // Stage 2 outputs
    raw_chunks: Vec<RawChunk>,

    // PHASE 1: FILE GRAPHS (produced before modality graphs, before classification)
    pub file_graphs: HashMap<String, u64>, // file_path → graph_id
    pub classified_file_graphs: Vec<ClassifiedFileGraph>, // primary/supplementary/raw roles
    pub chunk_graph_ids: Vec<u64>,         // ordered list of chunk graph IDs

    // PHASE 2: INITIAL MODALITY GRAPHS
    pub modality_graphs: HashMap<String, u64>, // modality_name → graph_id
    pub graph_states: HashMap<u64, SessionGraphState>, // graph_id → state
    pub root_modality_list: RootModalityList,  // aggregated from chunks
    pub initial_graphs_created: bool,
    pub cross_modal_index_id: Option<u64>,

    processed_chunks: Vec<ProcessedChunk>,
    cleaned_prompt: String,
    prompt_tokens: u32,
    keywords: Vec<String>,
    entities: Vec<ExtractedEntity>,
    topics: Vec<String>,

    // Methodology/Category tracking
    methodologies: Vec<u64>,
    categories: Vec<u64>,
    categories_created: u32,

    // AMT
    amt: Option<AMTNode>,
    amt_validated: bool,
    validation_streak: u32, // Need 5 consecutive Valid for completion
    needs_clarification: bool,
    clarification_points: Vec<String>,
    intent_captures: Vec<IntentCapture>,
    branch_captures: Vec<BranchCapture>,
    detail_captures: Vec<DetailCapture>,
    cross_refs: Vec<CrossRef>,
    amt_pass_count: u32,
    coverage_aspects: Vec<String>,

    // Blueprint
    blueprint_id: Option<u64>,
    blueprint_steps: Vec<BlueprintStep>,
    pub orch_step_states: HashMap<u32, OrchestratorStepState>, // step_index → state
    blueprints_created: u32,

    // Execution
    task_id: Option<u64>,
    step_results: Vec<StepResult>,
    final_response: Option<String>,
    step_contexts: HashMap<u32, String>,
    step_outputs: HashMap<u32, serde_json::Value>,

    // Consciousness
    gate_result: Option<GateResult>,
    voice_identity: Option<VoiceIdentity>,

    // Pipeline registry (loaded from index.json)
    available_pipelines: Vec<PipelineInfo>,
}

#[derive(Debug, Clone)]
struct StepResult {
    step_index: u32,
    pipeline_id: u64,
    output: serde_json::Value,
    tokens_used: u32,
    iterations: u32,
    sub_step_results: Vec<SubStepResult>,
}

#[derive(Debug, Clone)]
struct SubStepResult {
    sub_index: u32,
    output: serde_json::Value,
    success: bool,
}

/// Voice identity from consciousness self_model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceIdentity {
    pub tone: String,
    pub formality: f32,
    pub warmth: f32,
    pub directness: f32,
    pub humor_level: f32,
    pub vocabulary_style: String,
}

#[derive(Debug, Clone)]
struct ValidationResult {
    is_valid: bool,
    issues: Vec<String>,
}

// ============================================================================
// Pipeline Executor Trait
// ============================================================================

/// Trait for executing pipelines (implemented by runtime)
#[async_trait::async_trait]
pub trait PipelineExecutor: Send + Sync {
    async fn execute(
        &self,
        pipeline_id: u64,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, String>;
}

// ============================================================================
// ZSEI Direct Access Trait
// ============================================================================

#[async_trait::async_trait]
pub trait ZSEIAccess: Send + Sync {
    /// Execute a ZSEI query
    async fn query(&self, query: serde_json::Value) -> Result<serde_json::Value, String>;

    /// Perform traversal
    async fn traverse(&self, request: serde_json::Value) -> Result<serde_json::Value, String>;

    /// Create a container
    async fn create_container(
        &self,
        parent_id: u64,
        container: serde_json::Value,
    ) -> Result<u64, String>;

    /// Update a container
    async fn update_container(
        &self,
        container_id: u64,
        updates: serde_json::Value,
    ) -> Result<(), String>;

    /// Get container by ID
    async fn get_container(&self, container_id: u64) -> Result<Option<serde_json::Value>, String>;

    /// Search containers by keywords
    async fn search_by_keywords(
        &self,
        keywords: &[String],
        container_type: Option<&str>,
    ) -> Result<Vec<u64>, String>;

    /// Get all categories
    async fn get_categories(&self, modality: &str) -> Result<Vec<u64>, String>;
}

// ============================================================================
// Orchestrator Implementation
// ============================================================================

pub struct PromptOrchestrator {
    executor: Arc<dyn PipelineExecutor>,
    zsei: Arc<dyn ZSEIAccess>,
    task_manager: Arc<TaskManager>,
    pipeline_index: Arc<RwLock<Option<PipelineIndex>>>,
}

impl PromptOrchestrator {
    pub fn new(
        executor: Arc<dyn PipelineExecutor>,
        zsei: Arc<dyn ZSEIAccess>,
        task_manager: Arc<TaskManager>,
        pipeline_index: Arc<RwLock<Option<PipelineIndex>>>,
    ) -> Self {
        Self {
            executor,
            zsei,
            task_manager,
            pipeline_index,
        }
    }

    fn extract_json_from_response<'a>(s: &'a str, open: char, close: char) -> &'a str {
        let trimmed = s.trim();
        if let Some(start) = trimmed.find(open) {
            if let Some(end) = trimmed.rfind(close) {
                if end >= start {
                    return &trimmed[start..=end];
                }
            }
        }
        trimmed
    }

    /// Load pipeline index from ZSEI
    pub async fn load_pipeline_index(&self) -> Result<(), String> {
        // Try to load from ZSEI storage
        let index_result = self
            .zsei
            .query(serde_json::json!({
                "type": "GetPipelineIndex"
            }))
            .await;

        if let Ok(result) = index_result {
            if let Ok(index) = serde_json::from_value::<PipelineIndex>(result) {
                *self.pipeline_index.write().await = Some(index);
            }
        }
        Ok(())
    }

    /// Get available pipelines
    async fn get_available_pipelines(&self) -> Vec<PipelineInfo> {
        if let Some(index) = self.pipeline_index.read().await.as_ref() {
            index
                .pipelines
                .iter()
                .filter(|p| !p.deprecated)
                .cloned()
                .collect()
        }
    }

    /// Main entry point - orchestrates the full 11-stage flow
    pub async fn orchestrate(&self, request: OrchestrationRequest) -> OrchestrationResponse {
        let model_identifier = request
            .model_config
            .as_ref()
            .and_then(|c| c.model_identifier.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("claude-sonnet-4");

        let model_context_limit = request
            .model_config
            .as_ref()
            .and_then(|c| c.context_length)
            .unwrap_or_else(|| get_model_context_limit(model_identifier));

        let prompt_tokens = Self::estimate_tokens(&request.prompt);

        // Load pipeline index if not already loaded
        let _ = self.load_pipeline_index().await;
        let available_pipelines = self.get_available_pipelines().await;

        let mut state = OrchestrationState {
            request: request.clone(),
            start_time: std::time::Instant::now(),
            stages: Vec::new(),
            model_context_limit,
            tokens_used_so_far: prompt_tokens,
            raw_chunks: Vec::new(),
            file_graphs: HashMap::new(),
            classified_file_graphs: Vec::new(),
            chunk_graph_ids: Vec::new(),
            modality_graphs: HashMap::new(),
            graph_states: HashMap::new(),
            root_modality_list: RootModalityList::default(),
            initial_graphs_created: false,
            cross_modal_index_id: None,
            processed_chunks: Vec::new(),
            cleaned_prompt: String::new(),
            prompt_tokens,
            keywords: Vec::new(),
            entities: Vec::new(),
            topics: Vec::new(),
            methodologies: Vec::new(),
            categories: Vec::new(),
            categories_created: 0,
            amt: None,
            amt_validated: false,
            validation_streak: 0,
            needs_clarification: false,
            clarification_points: Vec::new(),
            intent_captures: Vec::new(),
            branch_captures: Vec::new(),
            detail_captures: Vec::new(),
            cross_refs: Vec::new(),
            amt_pass_count: 0,
            coverage_aspects: Vec::new(),
            blueprint_id: None,
            blueprint_steps: Vec::new(),
            orch_step_states: HashMap::new(),
            blueprints_created: 0,
            task_id: None,
            step_results: Vec::new(),
            final_response: None,
            step_contexts: HashMap::new(),
            step_outputs: HashMap::new(),
            gate_result: None,
            voice_identity: None,
            available_pipelines,
        };

        // Check I-Loop before starting (if consciousness enabled)
        if request.consciousness_enabled {
            if let Err(e) = self.wait_for_i_loop().await {
                return self.build_error_response(&mut state, format!("I-Loop wait failed: {}", e));
            }
        }

        let result = self.execute_stages(&mut state).await;

        match result {
            Ok(_) => self.build_success_response(&state),
            Err(e) => self.build_error_response(&mut state, e),
        }
    }

    async fn execute_stages(&self, state: &mut OrchestrationState) -> Result<(), String> {
        // STAGE 1: Input Capture (already done - prompt is in request)
        self.record_stage(state, 1, "Input Capture", true, "Prompt received");

        // STAGE 2: Process attached files → produce file graphs
        // (runs before chunking so file modality data can inform AMT)
        if !state.request.attached_files.is_empty() {
            let available_modalities: Vec<String> = state
                .root_modality_list
                .verified_modalities
                .iter()
                .map(|vm| vm.modality.clone())
                .collect();

            for file_spec in &state.request.attached_files.clone() {
                let modality = self.detect_file_modality(&file_spec.file_path);
                let pipeline_id = self.modality_name_to_pipeline_id(&modality);

                if pipeline_id == 0 {
                    // Unknown modality — store as unknown, do not halt
                    tracing::warn!(
                        "Unknown modality for file: {} — treating as text",
                        file_spec.file_path
                    );
                }

                let effective_pipeline = if pipeline_id == 0 {
                    100u64
                } else {
                    pipeline_id
                };

                let analysis_result = self
                    .process_modality(
                        file_spec.content_preview.as_deref().unwrap_or(""),
                        effective_pipeline,
                        &available_modalities,
                    )
                    .await?;

                // Create graph from analysis result
                let graph_input = serde_json::json!({
                    "action": {
                        "type": "CreateGraph",
                        "analysis_result": analysis_result.get("analysis").cloned().unwrap_or_default(),
                        "project_id": state.request.project_id.unwrap_or(0),
                        "link_to_existing": false
                    }
                });

                let graph_result = self
                    .executor
                    .execute(effective_pipeline, graph_input)
                    .await
                    .unwrap_or_default();

                let graph_id = graph_result
                    .get("graph_id")
                    .and_then(|g| g.as_u64())
                    .unwrap_or(Self::generate_id_static());

                state
                    .file_graphs
                    .insert(file_spec.file_path.clone(), graph_id);
            }
        }

        // STAGE 3: Text/Prompt Normalization
        self.prompt_normalization(state).await?;

        // STAGE 4: Text/Prompt Normalization
        self.gather_methodologies(state).await?;

        // STAGE 5: Text/Prompt Normalization
        self.build_amt(state).await?;

        // PHASE 6: Initial graph creation — BEFORE AMT
        self.aggregate_root_modalities(state).await;
        self.create_initial_modality_graphs(state)
            .await
            .map_err(|e| format!("Initial graph creation failed: {}", e))?;
        self.record_stage(
            state,
            2,
            "Initial Graph Creation",
            true,
            &format!(
                "{} modality graphs created ({}), cross-modal: {}",
                state.modality_graphs.len(),
                state
                    .modality_graphs
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                state
                    .graph_states
                    .values()
                    .map(|g| g.cross_modal_edge_count)
                    .sum::<usize>()
            ),
        );

        // If clarification needed, stop here and return to user
        if state.needs_clarification {
            return Ok(());
        }

        // STAGE 7: Blueprint Assignment
        self.stage_3_blueprint_assignment(state).await?;

        // STAGE 8: Zero-Shot Simulation (with AMT traversal)
        self.stage_4_zero_shot_simulation(state).await?;

        // STAGE 9: Consciousness Decision Gate
        if state.request.consciousness_enabled {
            self.stage_5_consciousness_gate(state).await?;
        } else {
            self.record_stage(state, 5, "Consciousness Gate", true, "Skipped (disabled)");
        }

        // STAGE 10-12: Context Aggregation + Task Creation + Execution
        self.stage_6_to_8_execute_steps(state).await?;

        // STAGE 13: Result Collection
        self.stage_9_result_collection(state).await?;

        // STAGE 14: Post-execution Consciousness
        if state.request.consciousness_enabled {
            self.stage_10_post_execution(state).await?;
        } else {
            self.record_stage(state, 10, "Post-execution", true, "Skipped (disabled)");
        }

        // STAGE 15: Response Delivery
        self.stage_11_response_delivery(state).await?;

        Ok(())
    }

    // ========================================================================
    // STAGE 2: Text/Prompt Normalization
    // ========================================================================

    async fn prompt_normalization(&self, state: &mut OrchestrationState) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // STEP 1: Analyze the text using text modality pipeline
        let available_modalities: Vec<String> = state
            .root_modality_list
            .verified_modalities
            .iter()
            .map(|vm| vm.modality.clone())
            .collect();

        let process_result = self
            .process_modality(&state.request.prompt, 100, &available_modalities)
            .await?;

        // STEP 2: Process each chunk (clean + extract keywords/entities/topics)
        let mut all_keywords: HashSet<String> = HashSet::new();
        let mut all_entities: Vec<ExtractedEntity> = Vec::new();
        let mut all_topics: HashSet<String> = HashSet::new();

        if let Some(processed_arr) = process_result
            .get("processed_chunks")
            .and_then(|p| p.as_array())
        {
            for processed_val in processed_arr {
                if let Ok(processed) =
                    serde_json::from_value::<ProcessedChunk>(processed_val.clone())
                {
                    // Collect all keywords
                    for kw in &processed.keywords {
                        all_keywords.insert(kw.clone());
                    }
                    // Collect all entities
                    all_entities.extend(processed.entities.clone());
                    // Collect all topics
                    for topic in &processed.topics {
                        all_topics.insert(topic.clone());
                    }
                    state.processed_chunks.push(processed);
                }
            }
        }

        // Collect chunk graph ID for cross-model reconstruction capability
        if let Some(cg) = process_result
            .get("chunk_graph")
            .and_then(|cg| cg.get("graph_id"))
            .and_then(|id| id.as_u64())
        {
            state.chunk_graph_ids.push(cg);
        }

        // Collect grammar relationships for AMT evidence
        // (stored in chunk graphs, also aggregated for cross-chunk analysis)
        if let Some(rels) = process_result
            .get("grammar_relationships")
            .and_then(|r| r.as_array())
        {
            // These feed into branch discovery in build_amt_layer_by_layer
            // Store count for logging; individual rels are in chunk graphs
            tracing::debug!(
                "Chunk {} produced {} grammar relationships",
                chunk.index,
                rels.len()
            );
        }

        // STEP 3: Reconstruct cleaned prompt from chunks
        let reconstruct_input = serde_json::json!({
            "action": {
                "type": "ReconstructFromChunks",
                "chunks": state.processed_chunks
            }
        });

        let reconstruct_result = self.executor.execute(100, reconstruct_input).await?;

        state.cleaned_prompt = reconstruct_result
            .get("reconstructed_text")
            .and_then(|t| t.as_str())
            .unwrap_or(&state.request.prompt)
            .to_string();

        state.prompt_tokens = Self::estimate_tokens(&state.cleaned_prompt);
        state.keywords = all_keywords.into_iter().collect();
        state.entities = all_entities;
        state.topics = all_topics.into_iter().collect();

        self.record_stage_timed(
            state,
            2,
            "Text Normalization",
            true,
            &format!(
                "Chunks: {}, ChunkGraphs: {}, Files: {} ({} classified), Intents: {}, Branches: {}, Details: {}, Cross-refs: {}, Methodologies: {}, Categories: {} ({} created), Passes: {}, Validated: {}",
                state.processed_chunks.len(),
                state.chunk_graph_ids.len(),          // NEW
                state.file_graphs.len(),               // NEW
                state.classified_file_graphs.len(),    // NEW
                state.intent_captures.len(),
                state.intent_captures.len(),
                state.branch_captures.len(),
                state.detail_captures.len(),
                state.cross_refs.len(),
                state.methodologies.len(),
                state.categories.len(),
                state.categories_created,
                state.amt_pass_count,
                state.amt_validated,
            ),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // STAGE 3: Gather Methodologies
    // ========================================================================

    async fn gather_methodologies(&self, state: &mut OrchestrationState) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // Search methodologies by keywords via ZSEI
        let methodology_ids = self
            .zsei
            .search_by_keywords(
                &state.keywords.iter().take(20).cloned().collect::<Vec<_>>(),
                Some("Methodology"),
            )
            .await
            .unwrap_or_default();

        state.methodologies = methodology_ids;

        // Get categories from methodologies and cross-reference
        let mut methodology_categories: HashSet<u64> = HashSet::new();

        for method_id in &state.methodologies {
            if let Ok(Some(container)) = self.zsei.get_container(*method_id).await {
                if let Some(cats) = container
                    .get("local_state")
                    .and_then(|ls| ls.get("context"))
                    .and_then(|ctx| ctx.get("categories"))
                    .and_then(|c| c.as_array())
                {
                    for cat in cats {
                        if let Some(cat_id) = cat.as_u64() {
                            methodology_categories.insert(cat_id);
                        }
                    }
                }
            }
        }

        // Get all existing categories
        let mut existing_category_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Check methodology_categories (already loaded container IDs)
        for &cat_id in &methodology_categories {
            if let Ok(Some(container)) = self.zsei.get_container(cat_id).await {
                if let Some(name) = container
                    .get("local_state")
                    .and_then(|ls| ls.get("metadata"))
                    .and_then(|m| m.get("name"))
                    .and_then(|n| n.as_str())
                {
                    existing_category_names.insert(name.to_lowercase());
                }
                // Also check keywords as category names may be stored there
                if let Some(kws) = container
                    .get("local_state")
                    .and_then(|ls| ls.get("context"))
                    .and_then(|ctx| ctx.get("keywords"))
                    .and_then(|k| k.as_array())
                {
                    for kw in kws {
                        if let Some(kw_str) = kw.as_str() {
                            existing_category_names.insert(kw_str.to_lowercase());
                        }
                    }
                }
            }
        }

        // Also search ZSEI for existing categories matching each topic
        for topic in &state.topics {
            if topic.is_empty() {
                continue;
            }
            let topic_lower = topic.to_lowercase();

            // Use search_by_keywords to find if a category with this name exists
            let existing_matches = self
                .zsei
                .search_by_keywords(&[topic.clone()], Some("Category"))
                .await
                .unwrap_or_default();

            let needs_creation =
                !existing_category_names.contains(&topic_lower) && existing_matches.is_empty();

            if needs_creation {
                let new_category = serde_json::json!({
                    "container_type": "Category",
                    "modality": "Text",
                    "metadata": {
                        "name": topic,
                        "description": format!("Auto-created category for topic: {}", topic),
                        "created_by": "orchestrator"
                    },
                    "context": {
                        "keywords": [topic_lower],
                        "topics": []
                    }
                });

                if let Ok(new_id) = self.zsei.create_container(0, new_category).await {
                    state.categories.push(new_id);
                    state.categories_created += 1;
                    existing_category_names.insert(topic_lower);
                }
            } else if let Some(&first_match) = existing_matches.first() {
                // Add to categories if found but not already tracked
                if !state.categories.contains(&first_match) {
                    state.categories.push(first_match);
                }
            }
        }

        // File graph classification — runs AFTER text processing gives us keywords/topics/intent
        // NEVER before, because we need AMT intent for accurate role determination
        if !state.file_graphs.is_empty() {
            self.classify_file_graphs_post_creation(state).await?;
        }

        // STEP 6: Build AMT layer-by-layer from chunks (processes each chunk individually)
        state.amt = Some(self.build_amt_layer_by_layer(state).await?);

        // STEP 7: Validate AMT (need 5 consecutive Valid)
        state.amt_validated = state.amt_pass_count > 0;
        state.validation_streak = 5; // convergence was achieved inside the builder

        self.record_stage_timed(
            state,
            3,
            "Gather Methodologies",
            true,
            &format!(
                "Chunks: {}, ChunkGraphs: {}, Files: {} ({} classified), Intents: {}, Branches: {}, Details: {}, Cross-refs: {}, Methodologies: {}, Categories: {} ({} created), Passes: {}, Validated: {}",
                state.processed_chunks.len(),
                state.chunk_graph_ids.len(),          // NEW
                state.file_graphs.len(),               // NEW
                state.classified_file_graphs.len(),    // NEW
                state.intent_captures.len(),
                state.intent_captures.len(),
                state.branch_captures.len(),
                state.detail_captures.len(),
                state.cross_refs.len(),
                state.methodologies.len(),
                state.categories.len(),
                state.categories_created,
                state.amt_pass_count,
                state.amt_validated,
            ),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // STAGE 4: AMT Build
    // ========================================================================
    async fn build_amt(&self, state: &mut OrchestrationState) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // File graph classification — runs AFTER text processing gives us keywords/topics/intent
        // NEVER before, because we need AMT intent for accurate role determination
        if !state.file_graphs.is_empty() {
            self.classify_file_graphs_post_creation(state).await?;
        }

        // STEP 6: Build AMT layer-by-layer from chunks (processes each chunk individually)
        state.amt = Some(self.build_amt_layer_by_layer(state).await?);

        // STEP 7: Validate AMT (need 5 consecutive Valid)
        state.amt_validated = state.amt_pass_count > 0;
        state.validation_streak = 5; // convergence was achieved inside the builder

        self.record_stage_timed(
            state,
            4,
            "Build AMT",
            true,
            &format!(
                "Chunks: {}, ChunkGraphs: {}, Files: {} ({} classified), Intents: {}, Branches: {}, Details: {}, Cross-refs: {}, Methodologies: {}, Categories: {} ({} created), Passes: {}, Validated: {}",
                state.processed_chunks.len(),
                state.chunk_graph_ids.len(),          // NEW
                state.file_graphs.len(),               // NEW
                state.classified_file_graphs.len(),    // NEW
                state.intent_captures.len(),
                state.intent_captures.len(),
                state.branch_captures.len(),
                state.detail_captures.len(),
                state.cross_refs.len(),
                state.methodologies.len(),
                state.categories.len(),
                state.categories_created,
                state.amt_pass_count,
                state.amt_validated,
            ),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    /// Build AMT layer-by-layer from processed chunks (processes each chunk individually)
    async fn build_amt_layer_by_layer(
        &self,
        state: &OrchestrationState,
    ) -> Result<AMTNode, String> {
        let max_outer_passes = 10;
        let convergence_threshold = 5; // passes without new insights before done
        let mut consecutive_no_new = 0u32;
        let mut node_id_counter = 1u64;

        // Initial modality graphs are available in state.modality_graphs.
        // Branch discovery prompts can reference detected modalities from graph metadata.
        // Use state.root_modality_list.verified_modalities for evidence of what's present.
        let detected_modality_names: Vec<String> = state
            .root_modality_list
            .verified_modalities
            .iter()
            .map(|vm| vm.modality.clone())
            .collect();

        // Outer convergence loop
        'outer: for outer_pass in 0..max_outer_passes {
            state.amt_pass_count += 1;
            let mut new_insights_this_pass = false;

            // --- PHASE 1A: Intent discovery ---
            // Build context of already-known intents for deduplication
            let known_intents_json: Vec<serde_json::Value> = state
                .intent_captures
                .iter()
                .map(|ic| serde_json::json!({"intent": ic.intent, "is_parallel": ic.is_parallel}))
                .collect();

            for chunk in &state.processed_chunks {
                let intent_prompt = format!(
                    r#"Analyze this text chunk to identify goals or intents expressed in it.
        A chunk may express MULTIPLE unrelated intents (parallel) or a single intent.

        ALREADY KNOWN INTENTS (do NOT repeat these):
        {}

        CHUNK {} of {}:
        {}

        MODALITIES DETECTED IN CONTENT: {}

        Return ONLY valid JSON with no explanation:
        {{
            "new_intents": [
                {{
                    "intent": "clear description of this goal/intent",
                    "is_parallel": true,
                    "source_sentence": "the exact sentence or paragraph from the chunk expressing this"
                }}
            ]
        }}
        If no new intents are found, return: {{"new_intents": []}}"#,
                    serde_json::to_string(&known_intents_json).unwrap_or_default(),
                    chunk.index + 1,
                    state.processed_chunks.len(),
                    &chunk.cleaned_text[..chunk.cleaned_text.len().min(1500)],
                    detected_modality_names.join(", ")
                );

                let intent_input = serde_json::json!({
                    "prompt": intent_prompt,
                    "max_tokens": 500,
                    "temperature": 0.2,
                    "system_context": "Extract new intents not already listed. Return only valid JSON. No explanation."
                });

                if let Ok(result) = self.executor.execute(9, intent_input).await {
                    let response = result
                        .get("response")
                        .and_then(|r| r.as_str())
                        .unwrap_or("{}");
                    let json_str = Self::extract_json_from_response(response, '{', '}');
                    let parsed = serde_json::from_str::<serde_json::Value>(json_str.trim())
                        .unwrap_or_else(|_| serde_json::json!({"new_intents": []}));

                    if let Some(new_intents) = parsed.get("new_intents").and_then(|n| n.as_array())
                    {
                        for intent_val in new_intents {
                            let intent_str = intent_val
                                .get("intent")
                                .and_then(|i| i.as_str())
                                .unwrap_or("")
                                .to_string();
                            let is_parallel = intent_val
                                .get("is_parallel")
                                .and_then(|p| p.as_bool())
                                .unwrap_or(false);
                            let source_sentence = intent_val
                                .get("source_sentence")
                                .and_then(|s| s.as_str())
                                .unwrap_or("")
                                .to_string();

                            if intent_str.is_empty() {
                                continue;
                            }

                            // Check for duplicates (case-insensitive substring match)
                            let already_known = state.intent_captures.iter().any(|ic| {
                                ic.intent
                                    .to_lowercase()
                                    .contains(&intent_str.to_lowercase())
                                    || intent_str
                                        .to_lowercase()
                                        .contains(&ic.intent.to_lowercase())
                            });

                            if !already_known {
                                state.intent_captures.push(IntentCapture {
                                    intent: intent_str,
                                    is_parallel,
                                    source_chunk_indices: vec![chunk.index],
                                    source_sentences: if source_sentence.is_empty() {
                                        vec![]
                                    } else {
                                        vec![source_sentence]
                                    },
                                    node_id: node_id_counter,
                                });
                                node_id_counter += 1;
                                new_insights_this_pass = true;
                            } else {
                                // Aggregate: add this chunk as an additional source
                                if let Some(existing) =
                                    state.intent_captures.iter_mut().find(|ic| {
                                        ic.intent
                                            .to_lowercase()
                                            .contains(&intent_str.to_lowercase())
                                    })
                                {
                                    if !existing.source_chunk_indices.contains(&chunk.index) {
                                        existing.source_chunk_indices.push(chunk.index);
                                        if !source_sentence.is_empty() {
                                            existing.source_sentences.push(source_sentence);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // If no intents found at all, create a default
            if state.intent_captures.is_empty() {
                state.intent_captures.push(IntentCapture {
                    intent: "Process user request".to_string(),
                    is_parallel: false,
                    source_chunk_indices: (0..state.processed_chunks.len() as u32).collect(),
                    source_sentences: vec![],
                    node_id: node_id_counter,
                });
                node_id_counter += 1;
                new_insights_this_pass = true;
            }

            // --- PHASE 1B: Branch discovery via methodologies ---
            for &method_id in &state.methodologies {
                if let Ok(Some(method_container)) = self.zsei.get_container(method_id).await {
                    // Extract methodology content
                    let method_name = method_container
                        .get("local_state")
                        .and_then(|ls| ls.get("metadata"))
                        .and_then(|m| m.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("Unknown methodology")
                        .to_string();
                    let method_description = method_container
                        .get("local_state")
                        .and_then(|ls| ls.get("context"))
                        .and_then(|ctx| ctx.get("keywords"))
                        .map(|kw| kw.to_string())
                        .unwrap_or_default();

                    let intents_summary: Vec<String> = state
                        .intent_captures
                        .iter()
                        .map(|ic| ic.intent.clone())
                        .collect();

                    // Already known branches for dedup
                    let known_branches_json: Vec<serde_json::Value> = state.branch_captures
                        .iter()
                        .map(|bc| serde_json::json!({"branch": bc.branch, "intent": bc.parent_intent}))
                        .collect();

                    let branch_prompt = format!(
                        r#"You are applying the methodology "{}" to a set of user intents.
        Methodology context: {}

        USER INTENTS:
        {}

        ALREADY IDENTIFIED BRANCHES (do NOT repeat these):
        {}

        Based on this methodology, what additional branches (sub-components, requirements, or considerations) should be addressed for each intent?
        Only suggest branches NOT already in the known list.

        Return ONLY valid JSON:
        {{
            "branches": [
                {{
                    "branch": "specific branch description",
                    "parent_intent": "the intent this branch belongs to",
                    "rationale": "why this methodology requires this branch"
                }}
            ]
        }}
        If no new branches apply, return: {{"branches": []}}"#,
                        method_name,
                        &method_description[..method_description.len().min(300)],
                        intents_summary.join("\n"),
                        serde_json::to_string(&known_branches_json).unwrap_or_default()
                    );

                    let branch_input = serde_json::json!({
                        "prompt": branch_prompt,
                        "max_tokens": 600,
                        "temperature": 0.3,
                        "system_context": "Suggest branches per methodology. Return only valid JSON. No explanation."
                    });

                    if let Ok(result) = self.executor.execute(9, branch_input).await {
                        let response = result
                            .get("response")
                            .and_then(|r| r.as_str())
                            .unwrap_or("{}");
                        let json_str = Self::extract_json_from_response(response, '{', '}');
                        let parsed = serde_json::from_str::<serde_json::Value>(json_str.trim())
                            .unwrap_or_else(|_| serde_json::json!({"branches": []}));

                        if let Some(branches) = parsed.get("branches").and_then(|b| b.as_array()) {
                            for branch_val in branches {
                                let branch_str = branch_val
                                    .get("branch")
                                    .and_then(|b| b.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let parent_intent = branch_val
                                    .get("parent_intent")
                                    .and_then(|p| p.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                if branch_str.is_empty() {
                                    continue;
                                }

                                // Find actual parent intent (fuzzy match)
                                let resolved_parent = state
                                    .intent_captures
                                    .iter()
                                    .find(|ic| {
                                        ic.intent
                                            .to_lowercase()
                                            .contains(&parent_intent.to_lowercase())
                                            || parent_intent
                                                .to_lowercase()
                                                .contains(&ic.intent.to_lowercase())
                                    })
                                    .map(|ic| ic.intent.clone())
                                    .unwrap_or_else(|| {
                                        state
                                            .intent_captures
                                            .first()
                                            .map(|ic| ic.intent.clone())
                                            .unwrap_or_default()
                                    });

                                let already_exists = state.branch_captures.iter().any(|bc| {
                                    bc.parent_intent == resolved_parent
                                        && (bc
                                            .branch
                                            .to_lowercase()
                                            .contains(&branch_str.to_lowercase())
                                            || branch_str
                                                .to_lowercase()
                                                .contains(&bc.branch.to_lowercase()))
                                });

                                if !already_exists {
                                    state.branch_captures.push(BranchCapture {
                                        branch: branch_str,
                                        parent_intent: resolved_parent,
                                        source_methodology_ids: vec![method_id],
                                        source_chunk_indices: vec![],
                                        source_sentences: vec![],
                                        node_id: node_id_counter,
                                    });
                                    node_id_counter += 1;
                                    new_insights_this_pass = true;
                                } else {
                                    // Aggregate: add methodology as additional source
                                    if let Some(existing) =
                                        state.branch_captures.iter_mut().find(|bc| {
                                            bc.parent_intent == resolved_parent
                                                && bc
                                                    .branch
                                                    .to_lowercase()
                                                    .contains(&branch_str.to_lowercase())
                                        })
                                    {
                                        if !existing.source_methodology_ids.contains(&method_id) {
                                            existing.source_methodology_ids.push(method_id);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // --- PHASE 2: Detail discovery over chunks ---
            for chunk in &state.processed_chunks {
                // Build context of existing branches for this chunk
                let branches_summary: Vec<serde_json::Value> = state
                    .branch_captures
                    .iter()
                    .map(|bc| {
                        serde_json::json!({
                            "branch": bc.branch,
                            "intent": bc.parent_intent
                        })
                    })
                    .collect();

                let known_details_json: Vec<serde_json::Value> = state
                    .detail_captures
                    .iter()
                    .filter(|dc| {
                        // Only known details for branches possibly covered by this chunk
                        state
                            .branch_captures
                            .iter()
                            .find(|bc| bc.branch == dc.parent_branch)
                            .map(|bc| {
                                bc.source_chunk_indices.contains(&chunk.index)
                                    || bc.source_chunk_indices.is_empty()
                            })
                            .unwrap_or(true)
                    })
                    .map(|dc| {
                        serde_json::json!({
                            "detail": dc.content,
                            "branch": dc.parent_branch
                        })
                    })
                    .collect();

                let detail_prompt = format!(
                    r#"Analyze this text chunk for specific details, requirements, and constraints that address the identified branches.

        BRANCHES TO ADDRESS:
        {}

        ALREADY IDENTIFIED DETAILS (do NOT repeat):
        {}

        CHUNK {} of {}:
        {}

        For each branch this chunk addresses, extract specific details. Also identify any completely NEW branches not in the list above.

        Return ONLY valid JSON:
        {{
            "details": [
                {{
                    "content": "specific detail, requirement, or constraint",
                    "type": "detail|requirement|constraint",
                    "parent_branch": "exact branch name this belongs to",
                    "source_sentence": "the exact sentence or paragraph from the chunk"
                }}
            ],
            "new_branches": [
                {{
                    "branch": "newly discovered branch",
                    "parent_intent": "intent it belongs to",
                    "source_sentence": "exact text"
                }}
            ]
        }}"#,
                    serde_json::to_string(&branches_summary).unwrap_or_default(),
                    serde_json::to_string(&known_details_json).unwrap_or_default(),
                    chunk.index + 1,
                    state.processed_chunks.len(),
                    &chunk.cleaned_text[..chunk.cleaned_text.len().min(1500)]
                );

                let detail_input = serde_json::json!({
                    "prompt": detail_prompt,
                    "max_tokens": 700,
                    "temperature": 0.3,
                    "system_context": "Extract details per branch. Return only valid JSON. No explanation."
                });

                if let Ok(result) = self.executor.execute(9, detail_input).await {
                    let response = result
                        .get("response")
                        .and_then(|r| r.as_str())
                        .unwrap_or("{}");
                    let json_str = Self::extract_json_from_response(response, '{', '}');
                    let parsed = serde_json::from_str::<serde_json::Value>(json_str.trim())
                        .unwrap_or_else(|_| serde_json::json!({"details": [], "new_branches": []}));

                    // Process new details
                    if let Some(details) = parsed.get("details").and_then(|d| d.as_array()) {
                        for detail_val in details {
                            let content = detail_val
                                .get("content")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            let detail_type = detail_val
                                .get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("detail")
                                .to_string();
                            let parent_branch = detail_val
                                .get("parent_branch")
                                .and_then(|p| p.as_str())
                                .unwrap_or("")
                                .to_string();
                            let source_sentence = detail_val
                                .get("source_sentence")
                                .and_then(|s| s.as_str())
                                .unwrap_or("")
                                .to_string();

                            if content.is_empty() || parent_branch.is_empty() {
                                continue;
                            }

                            // Resolve parent branch (fuzzy)
                            let resolved_branch = state
                                .branch_captures
                                .iter()
                                .find(|bc| {
                                    bc.branch
                                        .to_lowercase()
                                        .contains(&parent_branch.to_lowercase())
                                        || parent_branch
                                            .to_lowercase()
                                            .contains(&bc.branch.to_lowercase())
                                })
                                .map(|bc| bc.branch.clone())
                                .unwrap_or(parent_branch.clone());

                            // Find parent intent for this branch
                            let resolved_intent = state
                                .branch_captures
                                .iter()
                                .find(|bc| bc.branch == resolved_branch)
                                .map(|bc| bc.parent_intent.clone())
                                .unwrap_or_default();

                            let already_exists = state.detail_captures.iter().any(|dc| {
                                dc.parent_branch == resolved_branch
                                    && (dc.content.to_lowercase().contains(&content.to_lowercase())
                                        || content
                                            .to_lowercase()
                                            .contains(&dc.content.to_lowercase()))
                            });

                            if !already_exists {
                                // Also update branch's chunk indices
                                if let Some(branch) = state
                                    .branch_captures
                                    .iter_mut()
                                    .find(|bc| bc.branch == resolved_branch)
                                {
                                    if !branch.source_chunk_indices.contains(&chunk.index) {
                                        branch.source_chunk_indices.push(chunk.index);
                                    }
                                    if !source_sentence.is_empty()
                                        && !branch.source_sentences.contains(&source_sentence)
                                    {
                                        branch.source_sentences.push(source_sentence.clone());
                                    }
                                }

                                state.detail_captures.push(DetailCapture {
                                    content,
                                    detail_type,
                                    parent_branch: resolved_branch,
                                    parent_intent: resolved_intent,
                                    source_chunk_indices: vec![chunk.index],
                                    source_sentences: if source_sentence.is_empty() {
                                        vec![]
                                    } else {
                                        vec![source_sentence]
                                    },
                                    node_id: node_id_counter,
                                });
                                node_id_counter += 1;
                                new_insights_this_pass = true;
                            } else {
                                // Aggregate
                                if let Some(existing) =
                                    state.detail_captures.iter_mut().find(|dc| {
                                        dc.parent_branch == resolved_branch
                                            && dc
                                                .content
                                                .to_lowercase()
                                                .contains(&content.to_lowercase())
                                    })
                                {
                                    if !existing.source_chunk_indices.contains(&chunk.index) {
                                        existing.source_chunk_indices.push(chunk.index);
                                    }
                                    if !source_sentence.is_empty()
                                        && !existing.source_sentences.contains(&source_sentence)
                                    {
                                        existing.source_sentences.push(source_sentence);
                                    }
                                }
                            }
                        }
                    }

                    // Process new branches discovered during detail pass
                    if let Some(new_branches) =
                        parsed.get("new_branches").and_then(|nb| nb.as_array())
                    {
                        for branch_val in new_branches {
                            let branch_str = branch_val
                                .get("branch")
                                .and_then(|b| b.as_str())
                                .unwrap_or("")
                                .to_string();
                            let parent_intent = branch_val
                                .get("parent_intent")
                                .and_then(|p| p.as_str())
                                .unwrap_or("")
                                .to_string();
                            let source_sentence = branch_val
                                .get("source_sentence")
                                .and_then(|s| s.as_str())
                                .unwrap_or("")
                                .to_string();

                            if branch_str.is_empty() {
                                continue;
                            }

                            let already_exists = state.branch_captures.iter().any(|bc| {
                                bc.branch
                                    .to_lowercase()
                                    .contains(&branch_str.to_lowercase())
                                    || branch_str
                                        .to_lowercase()
                                        .contains(&bc.branch.to_lowercase())
                            });

                            if !already_exists {
                                let resolved_parent = state
                                    .intent_captures
                                    .iter()
                                    .find(|ic| {
                                        ic.intent
                                            .to_lowercase()
                                            .contains(&parent_intent.to_lowercase())
                                            || parent_intent
                                                .to_lowercase()
                                                .contains(&ic.intent.to_lowercase())
                                    })
                                    .map(|ic| ic.intent.clone())
                                    .unwrap_or_else(|| {
                                        state
                                            .intent_captures
                                            .first()
                                            .map(|ic| ic.intent.clone())
                                            .unwrap_or_default()
                                    });

                                state.branch_captures.push(BranchCapture {
                                    branch: branch_str,
                                    parent_intent: resolved_parent,
                                    source_methodology_ids: vec![],
                                    source_chunk_indices: vec![chunk.index],
                                    source_sentences: if source_sentence.is_empty() {
                                        vec![]
                                    } else {
                                        vec![source_sentence]
                                    },
                                    node_id: node_id_counter,
                                });
                                node_id_counter += 1;
                                new_insights_this_pass = true;
                            }
                        }
                    }
                }
            }

            // Cross-reference methodologies for the current layer state
            let method_findings = self
                .cross_reference_methodologies_for_layer(state, outer_pass as u32)
                .await;

            let has_new_methodology = !method_findings.is_empty();

            if !new_insights_this_pass && !has_new_methodology {
                consecutive_no_new += 1;
                if consecutive_no_new >= convergence_threshold {
                    break 'outer;
                }
            } else {
                if has_new_methodology {
                    // New methodologies may surface new branches — reset convergence
                    new_insights_this_pass = true;
                }
                consecutive_no_new = 0;
            }
        } // end 'outer

        // --- PHASE 3: Cross-reference linking ---
        let branch_list: Vec<(String, String)> = state
            .branch_captures
            .iter()
            .map(|bc| (bc.branch.clone(), bc.parent_intent.clone()))
            .collect();

        // Check pairs (limit to avoid combinatorial explosion: max 50 pairs)
        let max_pairs = 50usize;
        let mut pair_count = 0;
        'pairs: for i in 0..branch_list.len() {
            for j in (i + 1)..branch_list.len() {
                if pair_count >= max_pairs {
                    break 'pairs;
                }
                pair_count += 1;

                let (branch_a, intent_a) = &branch_list[i];
                let (branch_b, intent_b) = &branch_list[j];

                // Skip if same intent (same-intent relationships are handled by hierarchy)
                if intent_a == intent_b {
                    continue;
                }

                let crossref_prompt = format!(
                    r#"Are these two branches related to each other?

        BRANCH A (from intent: "{}"): {}
        BRANCH B (from intent: "{}"): {}

        If they are related, describe how.

        Return ONLY valid JSON:
        {{
            "related": true,
            "relationship_type": "depends_on|requires|relates_to|contradicts|shared_context",
            "description": "brief explanation"
        }}
        If not related: {{"related": false}}"#,
                    intent_a, branch_a, intent_b, branch_b
                );

                let crossref_input = serde_json::json!({
                    "prompt": crossref_prompt,
                    "max_tokens": 150,
                    "temperature": 0.2,
                    "system_context": "Identify cross-branch relationships. Return only valid JSON."
                });

                if let Ok(result) = self.executor.execute(9, crossref_input).await {
                    let response = result
                        .get("response")
                        .and_then(|r| r.as_str())
                        .unwrap_or("{}");
                    let json_str = Self::extract_json_from_response(response, '{', '}');
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str.trim()) {
                        if parsed
                            .get("related")
                            .and_then(|r| r.as_bool())
                            .unwrap_or(false)
                        {
                            let rel_type_str = parsed
                                .get("relationship_type")
                                .and_then(|rt| rt.as_str())
                                .unwrap_or("relates_to");
                            let description = parsed
                                .get("description")
                                .and_then(|d| d.as_str())
                                .unwrap_or("")
                                .to_string();

                            let relation_type = match rel_type_str {
                                "depends_on" => AMTRelationType::DependsOn,
                                "requires" => AMTRelationType::Requires,
                                "contradicts" => AMTRelationType::Contradicts,
                                "shared_context" => AMTRelationType::SharedContext,
                                _ => AMTRelationType::RelatesTo,
                            };

                            state.cross_refs.push(CrossRef {
                                from_branch: branch_a.clone(),
                                to_branch: branch_b.clone(),
                                from_intent: intent_a.clone(),
                                to_intent: intent_b.clone(),
                                relation_type,
                                description,
                            });
                        }
                    }
                }
            }
        }

        // --- BUILD AMT FROM CAPTURES ---

        // Determine root vs parallel structure
        let parallel_intents: Vec<&IntentCapture> = state
            .intent_captures
            .iter()
            .filter(|ic| ic.is_parallel)
            .collect();
        let primary_intents: Vec<&IntentCapture> = state
            .intent_captures
            .iter()
            .filter(|ic| !ic.is_parallel)
            .collect();

        // Create top-level root
        let root_intent = if primary_intents.len() == 1 {
            primary_intents[0].intent.clone()
        } else if !primary_intents.is_empty() {
            format!(
                "Multiple goals: {}",
                primary_intents
                    .iter()
                    .map(|ic| ic.intent.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        } else if !state.intent_captures.is_empty() {
            state.intent_captures[0].intent.clone()
        } else {
            "Process user request".to_string()
        };

        let mut root_node = AMTNode::new(node_id_counter, AMTNodeType::Root, root_intent, 0);
        root_node.source_chunk_indices = (0..state.processed_chunks.len() as u32).collect();
        root_node.methodology_ids = state.methodologies.clone();
        node_id_counter += 1;

        // Build intent → branch → detail tree
        let all_intents_combined: Vec<&IntentCapture> = state.intent_captures.iter().collect();

        for intent_capture in &all_intents_combined {
            // If multiple parallel intents, each becomes a Branch under Root
            // If single intent, branches go directly under Root
            let (intent_parent_node, intent_level) = if state.intent_captures.len() > 1 {
                // Create intent node as a Branch
                let mut intent_node = AMTNode::new(
                    intent_capture.node_id,
                    AMTNodeType::Branch,
                    intent_capture.intent.clone(),
                    1,
                );
                intent_node.source_chunk_indices = intent_capture.source_chunk_indices.clone();
                for sentence in &intent_capture.source_sentences {
                    intent_node.metadata.insert(
                        format!("source_sentence_{}", intent_node.metadata.len()),
                        sentence.clone(),
                    );
                }

                // Branches for this intent
                let branches_for_intent: Vec<&BranchCapture> = state
                    .branch_captures
                    .iter()
                    .filter(|bc| bc.parent_intent == intent_capture.intent)
                    .collect();

                for branch_capture in branches_for_intent {
                    let mut branch_node = AMTNode::new(
                        branch_capture.node_id,
                        AMTNodeType::Branch,
                        branch_capture.branch.clone(),
                        2,
                    );
                    branch_node.source_chunk_indices = branch_capture.source_chunk_indices.clone();
                    branch_node.methodology_ids = branch_capture.source_methodology_ids.clone();
                    for sentence in &branch_capture.source_sentences {
                        branch_node.metadata.insert(
                            format!("source_sentence_{}", branch_node.metadata.len()),
                            sentence.clone(),
                        );
                    }

                    // Details for this branch
                    let details_for_branch: Vec<&DetailCapture> = state
                        .detail_captures
                        .iter()
                        .filter(|dc| dc.parent_branch == branch_capture.branch)
                        .collect();

                    for detail_capture in details_for_branch {
                        let node_type = match detail_capture.detail_type.as_str() {
                            "constraint" => AMTNodeType::Consideration,
                            _ => AMTNodeType::Leaf,
                        };
                        let mut detail_node = AMTNode::new(
                            detail_capture.node_id,
                            node_type,
                            detail_capture.content.clone(),
                            3,
                        );
                        detail_node
                            .metadata
                            .insert("type".to_string(), detail_capture.detail_type.clone());
                        detail_node.source_chunk_indices =
                            detail_capture.source_chunk_indices.clone();
                        for sentence in &detail_capture.source_sentences {
                            detail_node.metadata.insert(
                                format!("source_sentence_{}", detail_node.metadata.len()),
                                sentence.clone(),
                            );
                        }
                        branch_node.children.push(detail_node);
                    }

                    // Cross-references for this branch
                    for cross_ref in &state.cross_refs {
                        if cross_ref.from_branch == branch_capture.branch {
                            let target_node_id = state
                                .branch_captures
                                .iter()
                                .find(|bc| bc.branch == cross_ref.to_branch)
                                .map(|bc| bc.node_id)
                                .unwrap_or(0);
                            if target_node_id > 0 {
                                branch_node.relationships.push(AMTRelation {
                                    target_id: target_node_id,
                                    relation_type: cross_ref.relation_type.clone(),
                                    confidence: 0.8,
                                });
                            }
                        }
                    }

                    intent_node.children.push(branch_node);
                }

                root_node.children.push(intent_node);
                continue; // skip the direct-branch path below
            };

            // Single intent: branches go directly under root
            let branches_for_intent: Vec<&BranchCapture> = state
                .branch_captures
                .iter()
                .filter(|bc| bc.parent_intent == intent_capture.intent)
                .collect();

            for branch_capture in branches_for_intent {
                let mut branch_node = AMTNode::new(
                    branch_capture.node_id,
                    AMTNodeType::Branch,
                    branch_capture.branch.clone(),
                    1,
                );
                branch_node.source_chunk_indices = branch_capture.source_chunk_indices.clone();
                branch_node.methodology_ids = branch_capture.source_methodology_ids.clone();

                let details_for_branch: Vec<&DetailCapture> = state
                    .detail_captures
                    .iter()
                    .filter(|dc| dc.parent_branch == branch_capture.branch)
                    .collect();

                for detail_capture in details_for_branch {
                    let node_type = match detail_capture.detail_type.as_str() {
                        "constraint" => AMTNodeType::Consideration,
                        _ => AMTNodeType::Leaf,
                    };
                    let mut detail_node = AMTNode::new(
                        detail_capture.node_id,
                        node_type,
                        detail_capture.content.clone(),
                        2,
                    );
                    detail_node
                        .metadata
                        .insert("type".to_string(), detail_capture.detail_type.clone());
                    detail_node.source_chunk_indices = detail_capture.source_chunk_indices.clone();
                    branch_node.children.push(detail_node);
                }

                for cross_ref in &state.cross_refs {
                    if cross_ref.from_branch == branch_capture.branch {
                        let target_node_id = state
                            .branch_captures
                            .iter()
                            .find(|bc| bc.branch == cross_ref.to_branch)
                            .map(|bc| bc.node_id)
                            .unwrap_or(0);
                        if target_node_id > 0 {
                            branch_node.relationships.push(AMTRelation {
                                target_id: target_node_id,
                                relation_type: cross_ref.relation_type.clone(),
                                confidence: 0.8,
                            });
                        }
                    }
                }

                root_node.children.push(branch_node);
            }
        }

        Ok(root_node)
    }

    // ========================================================================
    // STAGE 3: Blueprint Assignment
    // ========================================================================

    async fn stage_3_blueprint_assignment(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // Search for blueprint with 100% match
        let blueprint_ids = self
            .zsei
            .search_by_keywords(
                &state.keywords.iter().take(15).cloned().collect::<Vec<_>>(),
                Some("Blueprint"),
            )
            .await
            .unwrap_or_default();

        let mut best_match: Option<(u64, f32)> = None;

        for bp_id in blueprint_ids {
            if let Ok(Some(container)) = self.zsei.get_container(bp_id).await {
                // Calculate match score
                let bp_keywords: Vec<String> = container
                    .get("local_state")
                    .and_then(|ls| ls.get("context"))
                    .and_then(|ctx| ctx.get("keywords"))
                    .and_then(|k| serde_json::from_value(k.clone()).ok())
                    .unwrap_or_default();

                let state_keywords_set: HashSet<_> =
                    state.keywords.iter().map(|s| s.to_lowercase()).collect();
                let bp_keywords_set: HashSet<_> =
                    bp_keywords.iter().map(|s| s.to_lowercase()).collect();

                let intersection = state_keywords_set.intersection(&bp_keywords_set).count();
                let union = state_keywords_set.union(&bp_keywords_set).count();

                let match_score = if union > 0 {
                    intersection as f32 / union as f32
                } else {
                    0.0
                };

                if match_score > best_match.map(|(_, s)| s).unwrap_or(0.0) {
                    best_match = Some((bp_id, match_score));
                }
            }
        }

        // Only use if 100% match (or very close - 95%+)
        if let Some((bp_id, score)) = best_match {
            if score >= 0.95 {
                state.blueprint_id = Some(bp_id);

                // Load blueprint steps
                if let Ok(Some(container)) = self.zsei.get_container(bp_id).await {
                    state.blueprint_steps = container
                        .get("local_state")
                        .and_then(|ls| ls.get("storage"))
                        .and_then(|s| s.get("steps"))
                        .and_then(|steps| serde_json::from_value(steps.clone()).ok())
                        .unwrap_or_default();
                }

                self.record_stage_timed(
                    state,
                    3,
                    "Blueprint Assignment",
                    true,
                    &format!(
                        "Using existing blueprint {} (match: {:.0}%)",
                        bp_id,
                        score * 100.0
                    ),
                    stage_start.elapsed().as_millis() as u64,
                );
                return Ok(());
            }
        }

        // No 100% match - create new blueprint
        let amt = state.amt.as_ref().ok_or("No AMT available")?;

        // Generate blueprint from AMT with pipeline awareness
        let available_pipelines_desc: String = state
            .available_pipelines
            .iter()
            .filter(|p| !p.deprecated)
            .map(|p| format!("  - {} (ID: {}): {}", p.name, p.pipeline_id, p.description))
            .collect::<Vec<_>>()
            .join("\n");

        let blueprint_prompt = format!(
            r#"Create a blueprint (execution plan) from this AMT.

AMT ROOT: {}
BRANCHES:
{}

AVAILABLE PIPELINES:
{}

METHODOLOGIES: {:?}

For each step, select the most appropriate pipeline from the list.
If no existing pipeline can handle a requirement, add it to missing_capabilities.

Return JSON:
{{
    "name": "Blueprint name",
    "description": "What this blueprint does",
    "steps": [
        {{
            "step_index": 0,
            "action": "action_name",
            "description": "What this step does",
            "pipeline_id": 9,
            "context_requirements": ["full_context"],
            "depends_on": [],
            "wait_for_graph_update": false,
            "max_retries": 1
        }}
    ],
    "missing_capabilities": ["capability1", "capability2"]
}}"#,
            amt.content,
            amt.children
                .iter()
                .map(|c| format!(
                    "- {}: {} children, chunk refs: {:?}",
                    c.content,
                    c.children.len(),
                    c.source_chunk_indices
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            available_pipelines_desc,
            state.methodologies
        );

        let bp_input = serde_json::json!({
            "prompt": blueprint_prompt,
            "max_tokens": 1000,
            "temperature": 0.3,
            "system_context": "Generate execution blueprints. Respond with JSON only."
        });

        let bp_result = self.executor.execute(9, bp_input).await?;
        let response = bp_result
            .get("response")
            .and_then(|r| r.as_str())
            .unwrap_or("{}");
        let bp_json = Self::parse_json_object(response);

        let name = bp_json
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("Generated Blueprint")
            .to_string();
        let description = bp_json
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();

        // Check for missing capabilities
        let missing_capabilities: Vec<String> = bp_json
            .get("missing_capabilities")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if !missing_capabilities.is_empty() {
            // Log missing capabilities - could trigger pipeline creation in future
            tracing::warn!(
                "Blueprint has missing capabilities: {:?}",
                missing_capabilities
            );
        }

        state.blueprint_steps = bp_json
            .get("steps")
            .and_then(|s| serde_json::from_value(s.clone()).ok())
            .unwrap_or_else(|| {
                vec![BlueprintStep {
                    step_index: 0,
                    action: "execute_prompt".to_string(),
                    description: "Process the user prompt".to_string(),
                    pipeline_id: 9,
                    context_requirements: vec!["full_context".to_string()],
                    loop_config: None,
                    sub_steps: Vec::new(),
                    depends_on: Vec::new(),
                    wait_for_graph_update: false,
                    max_retries: 1,
                    timeout_ms: None,
                }]
            });

        // Store blueprint in ZSEI
        let blueprint_container = serde_json::json!({
            "container_type": "Blueprint",
            "metadata": {
                "name": name,
                "description": description,
                "created_by": "orchestrator"
            },
            "context": {
                "keywords": state.keywords,
                "topics": state.topics,
                "methodology_ids": state.methodologies
            },
            "storage": {
                "steps": state.blueprint_steps,
                "missing_capabilities": missing_capabilities
            }
        });

        if let Ok(new_id) = self.zsei.create_container(0, blueprint_container).await {
            state.blueprint_id = Some(new_id);
            state.blueprints_created += 1;
        }

        self.record_stage_timed(
            state,
            3,
            "Blueprint Assignment",
            true,
            &format!(
                "Created new blueprint with {} steps (missing: {})",
                state.blueprint_steps.len(),
                missing_capabilities.len()
            ),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // STAGE 4: Zero-Shot Simulation
    // ========================================================================

    async fn stage_4_zero_shot_simulation(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        let amt = match &state.amt {
            Some(a) => a,
            None => {
                self.record_stage_timed(
                    state,
                    4,
                    "Zero-Shot Simulation",
                    true,
                    "Skipped (no AMT)",
                    0,
                );
                return Ok(());
            }
        };

        // Simulate execution using AMT traversal
        let simulate_prompt = format!(
            r#"Simulate executing this plan and predict outcomes.

AMT STRUCTURE:
- Root intent: {}
- Branches: {}

BLUEPRINT STEPS:
{}

For each step, predict:
1. What information will be needed
2. What output will be produced
3. Potential issues or clarifications needed

Return JSON:
{{
    "simulation_confidence": 0.0-1.0,
    "step_predictions": [
        {{"step": 0, "needs": ["info1"], "produces": ["output1"], "risks": ["risk1"]}}
    ],
    "overall_feasibility": "high/medium/low",
    "clarifications_needed": []
}}"#,
            amt.content,
            amt.children
                .iter()
                .map(|c| &c.content)
                .collect::<Vec<_>>()
                .join(", "),
            state
                .blueprint_steps
                .iter()
                .map(|s| format!("Step {}: {} - {}", s.step_index, s.action, s.description))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let sim_input = serde_json::json!({
            "prompt": simulate_prompt,
            "max_tokens": 800,
            "temperature": 0.3,
            "system_context": "Simulate execution and predict outcomes. Respond with JSON only."
        });

        let sim_result = self.executor.execute(9, sim_input).await?;
        let response = sim_result
            .get("response")
            .and_then(|r| r.as_str())
            .unwrap_or("{}");
        let sim_json = Self::parse_json_object(response);

        let confidence = sim_json
            .get("simulation_confidence")
            .and_then(|c| c.as_f64())
            .unwrap_or(0.7);

        let feasibility = sim_json
            .get("overall_feasibility")
            .and_then(|f| f.as_str())
            .unwrap_or("medium");

        // Check for clarifications needed
        if let Some(clarifications) = sim_json
            .get("clarifications_needed")
            .and_then(|c| c.as_array())
        {
            for c in clarifications {
                if let Some(c_str) = c.as_str() {
                    if !c_str.is_empty() {
                        state.clarification_points.push(c_str.to_string());
                    }
                }
            }
        }

        if !state.clarification_points.is_empty() && confidence < 0.5 {
            state.needs_clarification = true;
        }

        self.record_stage_timed(
            state,
            4,
            "Zero-Shot Simulation",
            true,
            &format!(
                "Confidence: {:.0}%, Feasibility: {}",
                confidence * 100.0,
                feasibility
            ),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // STAGE 5: Consciousness Decision Gate
    // ========================================================================

    async fn stage_5_consciousness_gate(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // Call decision_gate pipeline (#39)
        let input = serde_json::json!({
            "action": "Evaluate",
            "task_id": 0,
            "task_summary": &state.cleaned_prompt[..state.cleaned_prompt.len().min(500)],
            "blueprint_id": state.blueprint_id.unwrap_or(0),
            "user_id": state.request.user_id,
            "amt_summary": {
                "intent": state.amt.as_ref().map(|a| &a.content),
                "branch_count": state.amt.as_ref().map(|a| a.branch_count()).unwrap_or(0)
            }
        });

        let result = self.executor.execute(39, input).await?;

        let decision = result
            .get("gate")
            .and_then(|g| g.get("decision"))
            .and_then(|d| d.as_str())
            .unwrap_or("Proceed");

        let confidence = result
            .get("gate")
            .and_then(|g| g.get("confidence"))
            .and_then(|c| c.as_f64())
            .unwrap_or(0.8) as f32;

        let reasoning = result
            .get("gate")
            .and_then(|g| g.get("reasoning"))
            .and_then(|r| r.as_str())
            .unwrap_or("No reasoning provided")
            .to_string();

        state.gate_result = Some(GateResult {
            decision: decision.to_string(),
            confidence,
            reasoning: reasoning.clone(),
        });

        if decision == "Decline" {
            return Err(format!("Consciousness gate declined: {}", reasoning));
        }

        self.record_stage_timed(
            state,
            5,
            "Consciousness Gate",
            true,
            &format!("Decision: {} ({:.0}%)", decision, confidence * 100.0),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // STAGES 6-8: Context Aggregation + Task Creation + Step Execution
    // ========================================================================

    async fn stage_6_to_8_execute_steps(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // STAGE 7: Create task via TaskManager
        let mut inputs = HashMap::new();
        inputs.insert(
            "prompt".to_string(),
            serde_json::json!(state.cleaned_prompt),
        );
        inputs.insert(
            "blueprint_id".to_string(),
            serde_json::json!(state.blueprint_id),
        );
        if let Some(ref amt) = state.amt {
            inputs.insert("amt_intent".to_string(), serde_json::json!(amt.content));
        }

        // Enqueue task via TaskManager
        let task_result = self
            .task_manager
            .enqueue_task(
                state.blueprint_id,
                inputs,
                state.request.user_id,
                state.request.device_id,
                state.request.workspace_id,
                state.request.project_id,
                TaskPriority::Normal,
            )
            .await;

        match task_result {
            Ok(task_id) => {
                state.task_id = Some(task_id);
            }
            Err(e) => {
                self.record_stage(state, 7, "Task Creation", false, &format!("Failed: {}", e));
                return Err(e.to_string());
            }
        }

        self.record_stage(
            state,
            7,
            "Task Creation",
            state.task_id.is_some(),
            &format!("Task: {:?}", state.task_id),
        );

        // STAGES 6 & 8: Execute steps
        let steps = state.blueprint_steps.clone();
        let mut all_outputs: Vec<String> = Vec::new();
        let mut completed: HashSet<u32> = HashSet::new();

        // Build dependency order
        let mut step_queue: Vec<&BlueprintStep> = steps.iter().collect();
        let mut iterations = 0;
        let max_iterations = steps.len() * 2;

        while !step_queue.is_empty() && iterations < max_iterations {
            iterations += 1;

            // Find steps whose dependencies are satisfied
            let ready_steps: Vec<_> = step_queue
                .iter()
                .filter(|s| s.depends_on.iter().all(|dep| completed.contains(dep)))
                .cloned()
                .collect();

            if ready_steps.is_empty() && !step_queue.is_empty() {
                // Force execute first remaining step (break deadlock)
                if let Some(step) = step_queue.first().cloned() {
                    let result = self.execute_step(state, step, &all_outputs).await?;
                    let output_text = self.extract_output_text(&result.output);
                    all_outputs.push(output_text.clone());
                    state.step_results.push(result.clone());
                    state.step_outputs.insert(
                        step.step_index,
                        serde_json::json!({"output": all_outputs.last()}),
                    );
                    completed.insert(step.step_index);
                    step_queue.retain(|s| s.step_index != step.step_index);

                    // Update TaskManager
                    if let Some(task_id) = state.task_id {
                        let _ = self
                            .task_manager
                            .update_step(
                                task_id,
                                step.step_index,
                                "completed",
                                result.tokens_used,
                                Some(output_text[..200.min(output_text.len())].to_string()),
                                None,
                            )
                            .await;

                        let _ = self
                            .task_manager
                            .update_progress(task_id, completed.len() as u32, steps.len() as u32)
                            .await;
                    }
                }
            } else {
                for step in ready_steps {
                    let result = self.execute_step(state, step, &all_outputs).await?;
                    let output_text = self.extract_output_text(&result.output);
                    all_outputs.push(output_text.clone());
                    state.step_results.push(result.clone());
                    state.step_outputs.insert(
                        step.step_index,
                        serde_json::json!({"output": all_outputs.last()}),
                    );
                    completed.insert(step.step_index);
                    step_queue.retain(|s| s.step_index != step.step_index);

                    // Update TaskManager
                    if let Some(task_id) = state.task_id {
                        let _ = self
                            .task_manager
                            .update_step(
                                task_id,
                                step.step_index,
                                "completed",
                                result.tokens_used,
                                Some(output_text[..200.min(output_text.len())].to_string()),
                                None,
                            )
                            .await;

                        let _ = self
                            .task_manager
                            .update_progress(task_id, completed.len() as u32, steps.len() as u32)
                            .await;
                    }
                }
            }
        }

        // Combine outputs into final response
        state.final_response = if all_outputs.len() == 1 {
            Some(all_outputs[0].clone())
        } else if !all_outputs.is_empty() {
            Some(all_outputs.join("\n\n"))
        } else {
            None
        };

        self.record_stage_timed(
            state,
            8,
            "Step Execution",
            state.final_response.is_some(),
            &format!(
                "{} steps executed, tokens: {}",
                state.step_results.len(),
                state.tokens_used_so_far
            ),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    /// Execute a single blueprint step (handles loops, sub-steps, retries)
    async fn execute_step(
        &self,
        state: &mut OrchestrationState,
        step: &BlueprintStep,
        previous_outputs: &[String],
    ) -> Result<StepResult, String> {
        let mut total_iterations = 0;
        let mut sub_step_results = Vec::new();
        let mut final_output = serde_json::json!({});

        // Handle loop configuration
        let (iterations, should_loop) = if let Some(loop_config) = &step.loop_config {
            match loop_config.loop_type {
                LoopType::Count => (loop_config.max_iterations, true),
                LoopType::While | LoopType::Until => (loop_config.max_iterations, true),
                LoopType::ForEach => {
                    // Get iteration count from iterate_over
                    let count = step.context_requirements.len() as u32;
                    (count.max(1), true)
                }
            }
        } else {
            (1, false)
        };

        for iteration in 0..iterations {
            total_iterations = iteration + 1;

            // STAGE 6: Context aggregation for this step
            let context_input = serde_json::json!({
                "action": "ForQuery",
                "query": format!("{} - {}", state.cleaned_prompt, step.description),
                "token_budget": state.model_context_limit / 4,
                "project_id": state.request.project_id,
                "workspace_id": state.request.workspace_id,
                "priority_order": step.context_requirements,
                "step_index": step.step_index,
                "iteration": iteration
            });

            let context_result = self.executor.execute(21, context_input).await?;
            let step_context = context_result
                .get("context")
                .and_then(|c| c.get("context_text"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();

            state
                .step_contexts
                .insert(step.step_index, step_context.clone());

            // Build full context with previous outputs
            let full_context = if !previous_outputs.is_empty() {
                format!(
                    "{}\n\nPrevious step outputs:\n{}",
                    step_context,
                    previous_outputs
                        .iter()
                        .enumerate()
                        .map(|(i, o)| format!("Step {}: {}", i + 1, &o[..o.len().min(300)]))
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            } else {
                step_context
            };

            // Execute sub-steps first if any
            for sub_step in &step.sub_steps {
                let sub_input = self.build_sub_step_input(state, sub_step, &full_context)?;
                let sub_result = self.executor.execute(sub_step.pipeline_id, sub_input).await;

                sub_step_results.push(SubStepResult {
                    sub_index: sub_step.sub_index,
                    output: sub_result.clone().unwrap_or_default(),
                    success: sub_result.is_ok(),
                });
            }

            // Execute main step
            let step_prompt = format!(
                "Step {}: {}\n\nContext:\n{}\n\nOriginal request: {}",
                step.step_index + 1,
                step.description,
                full_context,
                &state.cleaned_prompt[..state.cleaned_prompt.len().min(500)]
            );

            let exec_input = serde_json::json!({
                "prompt": step_prompt,
                "max_tokens": state.model_context_limit / 4,
                "temperature": 0.7,
                "action": step.action
            });

            let mut retries = 0;
            let mut exec_result = self
                .executor
                .execute(step.pipeline_id, exec_input.clone())
                .await;

            while exec_result.is_err() && retries < step.max_retries {
                retries += 1;
                tokio::time::sleep(tokio::time::Duration::from_millis(100 * retries as u64)).await;
                exec_result = self
                    .executor
                    .execute(step.pipeline_id, exec_input.clone())
                    .await;
            }

            final_output = exec_result?;

            // Wait for graph update if configured
            if step.wait_for_graph_update {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }

            // Check loop continuation condition
            if should_loop {
                if let Some(loop_config) = &step.loop_config {
                    let should_continue = match loop_config.loop_type {
                        LoopType::Count => iteration + 1 < loop_config.max_iterations,
                        LoopType::While => {
                            // Evaluate condition (simplified - check if output indicates completion)
                            let output_text = self.extract_output_text(&final_output);
                            !output_text.to_lowercase().contains("complete")
                                && !output_text.to_lowercase().contains("done")
                        }
                        LoopType::Until => {
                            let output_text = self.extract_output_text(&final_output);
                            output_text.to_lowercase().contains("continue")
                                || !output_text.to_lowercase().contains("complete")
                        }
                        LoopType::ForEach => iteration + 1 < iterations,
                    };

                    if !should_continue {
                        break;
                    }
                }
            }
        }

        let tokens_used = final_output
            .get("tokens_used")
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;

        state.tokens_used_so_far += tokens_used;

        // Fire OnStepComplete hook — living system integration
        self.on_step_complete(
            state,
            step,
            &StepResult {
                step_index: step.step_index,
                pipeline_id: step.pipeline_id,
                output: final_output.clone(),
                tokens_used,
                iterations: total_iterations,
                sub_step_results: sub_step_results.clone(),
            },
        )
        .await;

        Ok(StepResult {
            step_index: step.step_index,
            pipeline_id: step.pipeline_id,
            output: final_output,
            tokens_used,
            iterations: total_iterations,
            sub_step_results,
        })
    }

    /// Hook fires on step completion. Detects graph changes, reviews AMT,
    /// synthesizes new steps if AMT expanded. This IS the living system.
    async fn on_step_complete(
        &self,
        state: &mut OrchestrationState,
        step: &BlueprintStep,
        result: &StepResult,
    ) {
        // Detect graph modifications from step output
        let graphs_updated: Vec<String> = result
            .output
            .get("graphs_updated")
            .and_then(|g| g.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if graphs_updated.is_empty() {
            return;
        }

        // Update graph states for modified graphs
        for modality in &graphs_updated {
            if let Some(&gid) = state.modality_graphs.get(modality) {
                if let Some(gs) = state.graph_states.get_mut(&gid) {
                    gs.state = ModalityGraphState::Updated;
                    gs.consecutive_stable_passes = 0;
                }
            }
        }

        // Review AMT alignment — check if new intents emerged from graph updates
        let review_prompt = format!(
            r#"Step {} just completed and updated these modality graphs: {}.
    Current AMT intent: {}
    Current branches: {}

    Did this step's completion reveal any new intents or requirements not already in the AMT?

    Return ONLY valid JSON:
    {{"new_intents": [], "new_branches": [], "amt_needs_expansion": false}}"#,
            step.step_index,
            graphs_updated.join(", "),
            state.amt.as_ref().map(|a| a.content.as_str()).unwrap_or(""),
            state
                .branch_captures
                .iter()
                .map(|b| b.branch.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let input = serde_json::json!({
            "prompt": review_prompt,
            "max_tokens": 300,
            "temperature": 0.2,
            "system_context": "AMT alignment review. Return only valid JSON."
        });

        if let Ok(result) = self.executor.execute(9, input).await {
            let raw = result
                .get("response")
                .and_then(|r| r.as_str())
                .unwrap_or("{}");
            let start = raw.find('{').unwrap_or(0);
            let end = raw.rfind('}').map(|i| i + 1).unwrap_or(raw.len());

            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw[start..end]) {
                let amt_expanded = v
                    .get("amt_needs_expansion")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);

                if amt_expanded {
                    // New steps will be synthesized at the next iteration of execute_all_steps
                    // because new intent/branch captures are added here
                    if let Some(new_branches) = v.get("new_branches").and_then(|nb| nb.as_array()) {
                        for branch_val in new_branches {
                            if let Some(branch_str) = branch_val.as_str() {
                                let already_exists = state.branch_captures.iter().any(|bc| {
                                    bc.branch.to_lowercase() == branch_str.to_lowercase()
                                });
                                if !already_exists {
                                    // Add to branch_captures so blueprint creation can pick it up
                                    state.branch_captures.push(BranchCapture {
                                        branch: branch_str.to_string(),
                                        parent_intent: state
                                            .intent_captures
                                            .first()
                                            .map(|ic| ic.intent.clone())
                                            .unwrap_or_default(),
                                        source_methodology_ids: vec![],
                                        source_chunk_indices: vec![],
                                        source_sentences: vec![format!(
                                            "Discovered during step {} execution",
                                            step.step_index
                                        )],
                                        node_id: Self::generate_id_static(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn build_sub_step_input(
        &self,
        state: &OrchestrationState,
        sub_step: &BlueprintSubStep,
        context: &str,
    ) -> Result<serde_json::Value, String> {
        let mut input = serde_json::json!({
            "action": sub_step.action,
            "context": context,
            "prompt": state.cleaned_prompt
        });

        // Apply input mappings
        for (key, value) in &sub_step.input_mapping {
            if let Some(obj) = input.as_object_mut() {
                obj.insert(key.clone(), serde_json::json!(value));
            }
        }

        Ok(input)
    }

    fn extract_output_text(&self, output: &serde_json::Value) -> String {
        output
            .get("response")
            .and_then(|r| r.as_str())
            .or_else(|| output.get("output").and_then(|o| o.as_str()))
            .or_else(|| output.get("result").and_then(|r| r.as_str()))
            .unwrap_or("")
            .to_string()
    }

    // ========================================================================
    // STAGE 9: Result Collection
    // ========================================================================

    async fn stage_9_result_collection(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // Complete or fail task via TaskManager
        if let Some(task_id) = state.task_id {
            if state.final_response.is_some() {
                let outputs = state
                    .step_results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "step": r.step_index,
                            "output": self.extract_output_text(&r.output),
                            "tokens": r.tokens_used
                        })
                    })
                    .collect::<Vec<_>>();

                let _ = self
                    .task_manager
                    .complete_task(
                        task_id,
                        Some(serde_json::json!({
                            "response": state.final_response,
                            "steps": outputs
                        })),
                        state.tokens_used_so_far,
                    )
                    .await;
            } else {
                let _ = self
                    .task_manager
                    .fail_task(task_id, "No response generated".to_string())
                    .await;
            }
        }

        self.record_stage_timed(
            state,
            9,
            "Result Collection",
            true,
            &format!("Collected {} step results", state.step_results.len()),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // STAGE 10: Post-execution Consciousness
    // ========================================================================

    async fn stage_10_post_execution(&self, state: &mut OrchestrationState) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        if !state.request.consciousness_enabled {
            self.record_stage_timed(
                state,
                10,
                "Post-execution",
                true,
                "Consciousness disabled - skipped",
                stage_start.elapsed().as_millis() as u64,
            );
            return Ok(());
        }

        // Consciousness hooks are now handled by TaskManager automatically
        // when complete_task or fail_task is called.
        //
        // Additional consciousness processing can be done here if needed:

        // Store experience with more detail via consciousness pipelines
        let experience_input = serde_json::json!({
            "action": "StoreExperience",
            "experience_type": if state.final_response.is_some() { "task_success" } else { "task_failure" },
            "summary": &state.cleaned_prompt[..state.cleaned_prompt.len().min(200)],
            "task_id": state.task_id,
            "user_id": state.request.user_id,
            "tags": state.topics.clone(),
            "keywords": state.keywords.iter().take(10).cloned().collect::<Vec<_>>(),
            "methodologies_used": state.methodologies.clone(),
            "blueprint_id": state.blueprint_id,
            "significance": if state.final_response.is_some() { 0.5 } else { 0.3 },
            "tokens_used": state.tokens_used_so_far
        });

        // Pipeline 41 = CoreMemoryFormation
        let _ = self.executor.execute(41, experience_input).await;

        // Update relationship if we know the user
        let relationship_input = serde_json::json!({
            "action": "RecordInteraction",
            "user_id": state.request.user_id,
            "interaction_type": "task_completion",
            "outcome": if state.final_response.is_some() { "positive" } else { "negative" },
            "topics": state.topics.clone()
        });

        // Pipeline 47 = RelationshipDevelopment
        let _ = self.executor.execute(47, relationship_input).await;

        // Update emotional state
        let emotion_input = serde_json::json!({
            "action": "ProcessTrigger",
            "trigger_type": if state.final_response.is_some() { "task_success" } else { "task_failure" },
            "source": "orchestrator",
            "intensity": 0.5,
            "context": &state.cleaned_prompt[..state.cleaned_prompt.len().min(100)]
        });

        // Pipeline 43 = EmotionalBaselineUpdate
        let _ = self.executor.execute(43, emotion_input).await;

        self.record_stage_timed(
            state,
            10,
            "Post-execution Consciousness",
            true,
            "Experience stored, relationship updated, emotions processed",
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // STAGE 11: Response Delivery
    // ========================================================================

    async fn stage_11_response_delivery(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();

        // Apply voice identity if consciousness enabled
        if state.request.consciousness_enabled && state.final_response.is_some() {
            // Get voice identity from self_model (Pipeline 46 = NarrativeConstruction)
            let voice_input = serde_json::json!({
                "action": "GetVoice"
            });

            if let Ok(voice_result) = self.executor.execute(46, voice_input).await {
                if let Some(voice) = voice_result.get("voice") {
                    state.voice_identity = serde_json::from_value(voice.clone()).ok();

                    // Apply voice identity to response if significantly different from neutral
                    if let (Some(response), Some(voice_id)) =
                        (&state.final_response, &state.voice_identity)
                    {
                        let needs_restyle = voice_id.formality < 0.4
                            || voice_id.formality > 0.6
                            || voice_id.warmth < 0.4
                            || voice_id.warmth > 0.6;

                        if needs_restyle && response.len() > 50 {
                            let style_input = serde_json::json!({
                                "prompt": format!(
                                    "Rephrase this response to match this voice identity:\n\
                                    Tone: {}\n\
                                    Formality: {:.1}\n\
                                    Warmth: {:.1}\n\
                                    Directness: {:.1}\n\
                                    \nOriginal response:\n{}",
                                    voice_id.tone,
                                    voice_id.formality,
                                    voice_id.warmth,
                                    voice_id.directness,
                                    response
                                ),
                                "max_tokens": 2000,
                                "temperature": 0.7,
                                "system_context": "Apply voice identity while maintaining content accuracy."
                            });

                            if let Ok(styled) = self.executor.execute(9, style_input).await {
                                if let Some(new_response) =
                                    styled.get("response").and_then(|r| r.as_str())
                                {
                                    state.final_response = Some(new_response.to_string());
                                }
                            }
                        }
                    }
                }
            }

            // Update consciousness dashboard (Pipeline 54 = MetaPortionConsciousness)
            let dashboard_input = serde_json::json!({
                "action": "Update",
                "task_completed": true,
                "task_id": state.task_id,
                "success": state.final_response.is_some(),
                "tokens_used": state.tokens_used_so_far,
                "methodologies_used": state.methodologies.len(),
                "blueprint_id": state.blueprint_id
            });
            let _ = self.executor.execute(54, dashboard_input).await;
        }

        // Generate task recommendations for next steps (Pipeline 23 = TaskRecommendation)
        let recommend_input = serde_json::json!({
            "action": "Suggest",
            "context": &state.cleaned_prompt[..state.cleaned_prompt.len().min(200)],
            "completed_task_id": state.task_id,
            "topics": state.topics.clone(),
            "keywords": state.keywords.iter().take(5).cloned().collect::<Vec<_>>()
        });
        let _ = self.executor.execute(23, recommend_input).await;

        self.record_stage_timed(
            state,
            11,
            "Response Delivery",
            state.final_response.is_some(),
            &format!(
                "Response: {} chars, Voice: {}, Tokens: {}",
                state.final_response.as_ref().map(|r| r.len()).unwrap_or(0),
                state.voice_identity.is_some(),
                state.tokens_used_so_far
            ),
            stage_start.elapsed().as_millis() as u64,
        );

        Ok(())
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    async fn wait_for_i_loop(&self) -> Result<(), String> {
        let max_wait_ms = 30000;
        let check_interval_ms = 100;
        let mut waited = 0u64;

        loop {
            let input = serde_json::json!({ "action": "GetStatus" });

            if let Ok(result) = self.executor.execute(44, input).await {
                let active = result
                    .get("active")
                    .and_then(|a| a.as_bool())
                    .unwrap_or(false);
                if !active {
                    return Ok(());
                }
            } else {
                return Ok(());
            }

            if waited >= max_wait_ms {
                return Err("Timeout waiting for I-Loop".to_string());
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(check_interval_ms)).await;
            waited += check_interval_ms;
        }
    }

    /// File graphs are created FIRST, before any classification.
    /// MUST be called before classify_file_graphs_post_creation.
    /// Analyze text via a modality pipeline. Returns the analysis result.
    /// Callers handle graph creation and storage separately.
    async fn process_modality(
        &self,
        text: &str,
        pipeline_id: u64,
        available_modalities: &[String],
    ) -> Result<serde_json::Value, String> {
        let analyze_input = serde_json::json!({
            "action": {
                "type": "Analyze",
                "text": text,
                "depth": "Standard",
                "extract_entities": true,
                "extract_topics": true,
                "extract_structure": true,
                "available_modalities": available_modalities
            }
        });

        self.executor.execute(pipeline_id, analyze_input).await
    }

    /// Detect modality string from a file path using extension and path hints.
    fn detect_file_modality(&self, file_path: &str) -> String {
        let path_lower = file_path.to_lowercase();
        // Path hints (highest priority for ambiguous extensions)
        if path_lower.contains("radar") || path_lower.contains("sar") {
            return "radar".to_string();
        }
        if path_lower.contains("sonar") || path_lower.contains("bathymet") {
            return "sonar".to_string();
        }
        if path_lower.contains("thermal") || path_lower.contains("infrared") {
            return "thermal".to_string();
        }
        if path_lower.contains("hyperspectral") || path_lower.contains("multispectral") {
            return "hyperspectral".to_string();
        }
        if path_lower.contains("imu") || path_lower.contains("accelerom") {
            return "imu".to_string();
        }

        // Extension-based detection
        let ext = std::path::Path::new(file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "c" | "rb" | "swift" | "kt"
            | "sql" | "yaml" | "yml" | "toml" | "json" | "xml" => "code".to_string(),
            "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff" | "svg" => "image".to_string(),
            "mp3" | "wav" | "flac" | "ogg" | "aac" | "m4a" => "audio".to_string(),
            "mp4" | "mkv" | "avi" | "mov" | "webm" => "video".to_string(),
            "blend" | "obj" | "glb" | "gltf" | "fbx" | "stl" | "usd" | "urdf" => "3d".to_string(),
            "pdb" | "mol" | "sdf" | "cif" => "chemistry".to_string(),
            "fasta" | "fa" | "fastq" | "vcf" | "gff" | "bam" | "sam" => "dna".to_string(),
            "edf" | "bdf" | "gdf" | "fif" => "eeg".to_string(),
            "geojson" | "kml" | "shp" | "gpx" => "geospatial".to_string(),
            "step" | "stp" | "iges" | "brep" => "parametric_cad".to_string(),
            "pcd" | "las" | "laz" => "depth".to_string(),
            "xtf" | "jsf" | "s7k" | "gsf" => "sonar".to_string(),
            "iq" | "cphd" | "sicd" | "nitf" => "radar".to_string(),
            "hdr" | "bil" | "bip" | "bsq" | "h4" => "hyperspectral".to_string(),
            "tex" | "latex" | "nb" => "math".to_string(),
            "txt" | "md" | "rst" | "pdf" | "docx" | "doc" => "text".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Map modality name string to pipeline ID.
    fn modality_name_to_pipeline_id(&self, modality: &str) -> u64 {
        match modality {
            "text" => 100,
            "code" => 101,
            "image" => 102,
            "audio" => 103,
            "video" => 104,
            "math" => 105,
            "chemistry" => 106,
            "dna" => 107,
            "eeg" => 108,
            "3d" => 109,
            "sound" => 110,
            "biology" => 111,
            "proteomics" => 112,
            "haptic" => 113,
            "thermal" => 114,
            "depth" => 115,
            "imu" => 116,
            "geospatial" => 117,
            "electromagnetic" => 118,
            "bci" => 119,
            "parametric_cad" => 120,
            "kinematics" => 121,
            "control_systems" => 122,
            "network_topology" => 123,
            "radar" => 124,
            "sonar" => 125,
            "hyperspectral" => 126,
            _ => 0,
        }
    }

    /// Also add this static helper for use in non-&self contexts:
    fn generate_id_static() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Classify file graphs as primary / supplementary / raw_data.
    ///
    /// CRITICAL: This is called AFTER file graphs AND the text graph already exist.
    /// It operates on GRAPH METADATA only — never on raw file content.
    /// This eliminates the chunk-size assumption entirely.
    async fn classify_file_graphs_post_creation(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        if state.file_graphs.is_empty() {
            return Ok(());
        }

        // 1. Extract text graph intent summary from metadata (keywords + topics, never raw text)
        let text_intent_summary = format!(
            "Prompt keywords: [{}]. Topics: [{}]. AMT intent: {}",
            state
                .keywords
                .iter()
                .take(15)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            state
                .topics
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            state
                .amt
                .as_ref()
                .map(|a| a.content.as_str())
                .unwrap_or("not yet built"),
        );

        // 2. Collect file graph summaries — metadata only, NOT raw file content
        let file_summaries: Vec<serde_json::Value> = state.file_graphs.iter().map(|(path, &graph_id)| {
            let modality = self.detect_file_modality(path);
            serde_json::json!({
                "file_path": path,
                "graph_id": graph_id,
                "modality": modality,
                "graph_summary": format!("File graph {} for {} file: {}", graph_id, modality, path)
            })
        }).collect();

        // 3. LLM: compare file graph metadata against prompt intent
        let prompt = format!(
            r#"Given the prompt's intent summary and a set of file graph metadata,
    classify each file graph's role relative to the prompt.

    Prompt intent:
    {}

    File graphs (metadata only — no raw file content):
    {}

    For each file, classify as:
    - "primary"       — this file IS the main thing being worked on
    - "supplementary" — this file provides reference context or background
    - "raw_data"      — this file is raw data to be processed as pipeline input

    Return ONLY valid JSON array:
    [{{
      "file_path": "...",
      "graph_id": 0,
      "role": "primary|supplementary|raw_data",
      "reasoning": "brief"
    }}]"#,
            text_intent_summary,
            serde_json::to_string_pretty(&file_summaries).unwrap_or_default()
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.1,
            "system_context": "File role classification. Return only valid JSON array."
        });

        let result = self.executor.execute(9, input).await.unwrap_or_default();
        let raw = result
            .get("response")
            .and_then(|r| r.as_str())
            .unwrap_or("[]");
        let json_str = Self::extract_json_from_response(raw, '[', ']');

        let classifications: Vec<ClassifiedFileGraph> =
            serde_json::from_str::<Vec<serde_json::Value>>(&json_str)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|v| {
                    Some(ClassifiedFileGraph {
                        file_path: v["file_path"].as_str()?.to_string(),
                        graph_id: v["graph_id"].as_u64().or_else(|| {
                            state
                                .file_graphs
                                .get(v["file_path"].as_str().unwrap_or(""))
                                .copied()
                        })?,
                        modality: self.detect_file_modality(v["file_path"].as_str().unwrap_or("")),
                        role: match v["role"].as_str().unwrap_or("raw_data") {
                            "primary" => FileGraphRole::Primary,
                            "supplementary" => FileGraphRole::Supplementary,
                            _ => FileGraphRole::RawData,
                        },
                        reasoning: v["reasoning"].as_str().unwrap_or("").to_string(),
                    })
                })
                .collect();

        state.classified_file_graphs = classifications;
        Ok(())
    }

    /// Aggregate modality detections from all processed chunk graphs into the root modality list.
    /// Verifies each modality has a registered pipeline handler.
    async fn aggregate_root_modalities(&self, state: &mut OrchestrationState) {
        let mut modality_evidence: HashMap<String, Vec<ModalityEvidence>> = HashMap::new();

        for chunk in &state.processed_chunks {
            for detection in &chunk.detected_modalities {
                if detection.modality == "true_text" || detection.modality == "unknown" {
                    continue;
                }
                modality_evidence
                    .entry(detection.modality.clone())
                    .or_default()
                    .push(ModalityEvidence {
                        chunk_index: chunk.index,
                        span_start: detection.span_start,
                        span_end: detection.span_end,
                        intent_reference: detection.intent_reference.clone(),
                    });
            }
        }

        // Also include modalities from file graphs
        for classified in &state.classified_file_graphs {
            modality_evidence
                .entry(classified.modality.clone())
                .or_default()
                .push(ModalityEvidence {
                    chunk_index: u32::MAX, // sentinel: from file, not chunk
                    span_start: 0,
                    span_end: 0,
                    intent_reference: format!("file:{}", classified.file_path),
                });
        }

        // Always include text
        if !modality_evidence.contains_key("text") {
            modality_evidence.insert("text".to_string(), vec![]);
        }

        let mut verified = Vec::new();
        let mut unhandled = Vec::new();

        for (modality, evidence) in modality_evidence {
            let pipeline_id = self.modality_name_to_pipeline_id(&modality);
            if pipeline_id > 0 {
                verified.push(VerifiedModality {
                    modality,
                    pipeline_id,
                    evidence,
                });
            } else {
                unhandled.push(UnhandledModality { modality, evidence });
            }
        }

        state.root_modality_list = RootModalityList {
            verified_modalities: verified,
            unhandled_modalities: unhandled,
            total_chunk_count: state.processed_chunks.len() as u32,
        };
    }

    /// Phase 2: Create all initial modality graphs before AMT building.
    /// Three passes: structural (parallel) → semantic enrichment (text-first) →
    /// cross-modal reference (iterative 5x stable).
    async fn create_initial_modality_graphs(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        // PASS 1: Structural creation
        let verified = state.root_modality_list.verified_modalities.clone();

        for vm in &verified {
            let modality = vm.modality.clone();
            let pipeline_id = vm.pipeline_id;
            let project_id = state.request.project_id.unwrap_or(0);

            let modality_text: String = state
                .processed_chunks
                .iter()
                .flat_map(|chunk| {
                    chunk
                        .detected_modalities
                        .iter()
                        .filter(|d| d.modality == modality)
                        .map(|d| {
                            let end = d.span_end.min(chunk.cleaned_text.len());
                            chunk.cleaned_text[d.span_start.min(end)..end].to_string()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
                .join("\n");

            let analyze_input = serde_json::json!({
                "action": {
                    "type": "Analyze",
                    "text": if modality == "text" {
                        state.cleaned_prompt.clone()
                    } else {
                        modality_text
                    },
                    "depth": "Standard",
                    "extract_entities": true,
                    "extract_topics": true,
                    "extract_structure": false
                }
            });

            let analysis = self
                .executor
                .execute(pipeline_id, analyze_input)
                .await
                .unwrap_or_default();

            let graph_input = serde_json::json!({
                "action": {
                    "type": "CreateGraph",
                    "analysis_result": analysis.get("analysis").cloned().unwrap_or_default(),
                    "project_id": project_id,
                    "link_to_existing": false
                }
            });

            let graph_result = self
                .executor
                .execute(pipeline_id, graph_input)
                .await
                .unwrap_or_default();

            let graph_id = graph_result
                .get("graph_id")
                .and_then(|g| g.as_u64())
                .unwrap_or(Self::generate_id_static());

            state.modality_graphs.insert(modality.clone(), graph_id);
            state.graph_states.insert(
                graph_id,
                SessionGraphState {
                    graph_id,
                    modality: modality.clone(),
                    pipeline_id,
                    state: ModalityGraphState::Created,
                    cross_modal_edge_count: 0,
                    consecutive_stable_passes: 0,
                },
            );
        }

        // PASS 2: Semantic enrichment (text-first)
        if let Some(&text_graph_id) = state.modality_graphs.get("text") {
            let text_pipeline_id = self.modality_name_to_pipeline_id("text");
            let hook_input = serde_json::json!({
                "action": {
                    "type": "TriggerSemanticHook",
                    "graph_id": text_graph_id,
                    "hook_type": "OnInferRelationships"
                }
            });
            let _ = self.executor.execute(text_pipeline_id, hook_input).await;
            if let Some(gs) = state.graph_states.get_mut(&text_graph_id) {
                gs.state = ModalityGraphState::SemanticEnriched;
            }
        }

        let other_graphs: Vec<(String, u64)> = state
            .modality_graphs
            .iter()
            .filter(|(k, _)| *k != "text")
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        for (modality, graph_id) in &other_graphs {
            let pipeline_id = self.modality_name_to_pipeline_id(modality);
            let hook_input = serde_json::json!({
                "action": {
                    "type": "TriggerSemanticHook",
                    "graph_id": graph_id,
                    "hook_type": "OnInferRelationships"
                }
            });
            let _ = self.executor.execute(pipeline_id, hook_input).await;
            if let Some(gs) = state.graph_states.get_mut(graph_id) {
                gs.state = ModalityGraphState::SemanticEnriched;
            }
        }

        // PASS 3: Cross-modal references
        self.build_cross_modal_references_until_stable(state)
            .await?;

        state.initial_graphs_created = true;

        // Register in task store
        if let Some(task_id) = state.task_id {
            let register_input = serde_json::json!({
                "action": "RegisterGraphs",
                "task_id": task_id,
                "modality_graph_ids": state.modality_graphs,
            });
            let _ = self.executor.execute(5, register_input).await;
        }

        Ok(())
    }

    /// Iterative cross-modal reference building with 5x consecutive stable termination.
    async fn build_cross_modal_references_until_stable(
        &self,
        state: &mut OrchestrationState,
    ) -> Result<(), String> {
        let mut stable_count = 0u32;
        let mut pass = 0u32;

        loop {
            pass += 1;
            let new_edges = self.run_cross_modal_reference_pass(state, pass).await;

            if new_edges == 0 {
                stable_count += 1;
                if stable_count >= 5 {
                    break;
                }
            } else {
                stable_count = 0;
            }

            if pass > 30 {
                break;
            } // safety ceiling
        }

        // Mark all graphs CrossLinked → Stable
        for gs in state.graph_states.values_mut() {
            gs.state = ModalityGraphState::Stable;
            gs.consecutive_stable_passes = 5;
        }

        Ok(())
    }

    /// Single cross-modal reference pass. Returns count of new edges discovered.
    /// Pass 1-2: text ↔ others. Pass 3+: inter-modality.
    async fn run_cross_modal_reference_pass(
        &self,
        state: &mut OrchestrationState,
        pass_number: u32,
    ) -> usize {
        let mut new_edges = 0usize;

        let text_graph_id = state.modality_graphs.get("text").copied();
        let all_graphs: Vec<(String, u64)> = state
            .modality_graphs
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        // Phase A: Text → all other modalities
        if let Some(text_gid) = text_graph_id {
            for (target_modality, target_gid) in all_graphs.iter().filter(|(k, _)| k != "text") {
                let text_pipeline = self.modality_name_to_pipeline_id("text");
                let link_input = serde_json::json!({
                    "action": {
                        "type": "LinkToModality",
                        "source_graph_id": text_gid,
                        "target_graph_id": target_gid,
                        "target_modality": target_modality,
                        "relationship": "ReferencesModality"
                    }
                });
                if let Ok(result) = self.executor.execute(text_pipeline, link_input).await {
                    if result.get("link_result").is_some() {
                        new_edges += 1;
                        if let Some(gs) = state.graph_states.get_mut(target_gid) {
                            gs.cross_modal_edge_count += 1;
                        }
                    }
                }
            }
        }

        // Phase B: All → text back-references (pass 2+)
        if pass_number >= 2 {
            if let Some(text_gid) = text_graph_id {
                for (source_modality, source_gid) in all_graphs.iter().filter(|(k, _)| k != "text")
                {
                    let source_pipeline = self.modality_name_to_pipeline_id(source_modality);
                    let link_input = serde_json::json!({
                        "action": {
                            "type": "LinkToModality",
                            "source_graph_id": source_gid,
                            "target_graph_id": text_gid,
                            "target_modality": "text",
                            "relationship": "ReferencedBy"
                        }
                    });
                    if let Ok(result) = self.executor.execute(source_pipeline, link_input).await {
                        if result.get("link_result").is_some() {
                            new_edges += 1;
                        }
                    }
                }
            }
        }

        // Phase C: Inter-modality (pass 3+)
        if pass_number >= 3 {
            for i in 0..all_graphs.len() {
                for j in (i + 1)..all_graphs.len() {
                    let (a_mod, a_gid) = &all_graphs[i];
                    let (b_mod, b_gid) = &all_graphs[j];
                    if a_mod == "text" || b_mod == "text" {
                        continue;
                    }

                    let a_pipeline = self.modality_name_to_pipeline_id(a_mod);
                    let link_input = serde_json::json!({
                        "action": {
                            "type": "LinkToModality",
                            "source_graph_id": a_gid,
                            "target_graph_id": b_gid,
                            "target_modality": b_mod,
                            "relationship": "RelatesTo"
                        }
                    });
                    if let Ok(result) = self.executor.execute(a_pipeline, link_input).await {
                        if result.get("link_result").is_some() {
                            new_edges += 1;
                        }
                    }
                }
            }
        }

        new_edges
    }

    /// Cross-reference methodologies for a specific AMT layer.
    /// For each branch in the layer, finds required methodologies.
    /// Loads existing ones from ZSEI or synthesizes new ones when gaps exist.
    /// Returns a list of findings (loaded or created methodology IDs).
    async fn cross_reference_methodologies_for_layer(
        &self,
        state: &mut OrchestrationState,
        layer: u32,
    ) -> Vec<MethodologyFinding> {
        let mut findings = Vec::new();

        // Collect branches that belong to this AMT layer depth
        let layer_branches: Vec<String> = state
            .branch_captures
            .iter()
            .filter(|bc| {
                // Branches at depth = layer (root intent = layer 0, branches = layer 1, etc.)
                // Use intent index as a proxy for layer depth
                let intent_idx = state
                    .intent_captures
                    .iter()
                    .position(|ic| ic.intent == bc.parent_intent)
                    .unwrap_or(0);
                intent_idx + 1 == layer as usize || (layer == 1 && intent_idx == 0)
            })
            .map(|bc| bc.branch.clone())
            .collect();

        if layer_branches.is_empty() {
            return findings;
        }

        // Ask LLM which methodology domains are required for these branches
        let prompt = format!(
            r#"Given these AMT branches at layer {}, identify which methodology domains are needed.
    A methodology domain is a named area of systematic knowledge (e.g., "Software Testing",
    "Data Privacy", "Scientific Rigor", "API Design").

    Branches:
    {}

    Return ONLY valid JSON array:
    ["domain1", "domain2"]"#,
            layer,
            layer_branches
                .iter()
                .map(|b| format!("- {}", b))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.2,
            "system_context": "Methodology domain identification. Return only valid JSON array."
        });

        let required_domains: Vec<String> = match self.executor.execute(9, input).await {
            Ok(result) => {
                let raw = result
                    .get("response")
                    .and_then(|r| r.as_str())
                    .unwrap_or("[]");
                let start = raw.find('[').unwrap_or(0);
                let end = raw.rfind(']').map(|i| i + 1).unwrap_or(raw.len());
                serde_json::from_str::<Vec<String>>(&raw[start..end]).unwrap_or_default()
            }
            Err(_) => return findings,
        };

        for domain in &required_domains {
            // Search ZSEI for existing methodology matching this domain
            let existing = self
                .zsei
                .search_by_keywords(&[domain.clone()], Some("Methodology"))
                .await
                .unwrap_or_default();

            if let Some(&existing_id) = existing.first() {
                if !state.methodologies.contains(&existing_id) {
                    state.methodologies.push(existing_id);
                    findings.push(MethodologyFinding::Loaded(existing_id));
                }
            } else {
                // Synthesize a new methodology for this domain
                let synth_prompt = format!(
                    r#"Create a concise methodology for the domain: "{}"

    Return ONLY valid JSON:
    {{
      "name": "{}",
      "description": "what this methodology covers",
      "category": "domain category",
      "principles": ["principle 1", "principle 2"],
      "keywords": ["keyword1", "keyword2"]
    }}"#,
                    domain, domain
                );

                let synth_input = serde_json::json!({
                    "prompt": synth_prompt,
                    "max_tokens": 400,
                    "temperature": 0.2,
                    "system_context": "Methodology synthesis. Return only valid JSON."
                });

                if let Ok(synth_result) = self.executor.execute(9, synth_input).await {
                    let raw = synth_result
                        .get("response")
                        .and_then(|r| r.as_str())
                        .unwrap_or("{}");
                    let start = raw.find('{').unwrap_or(0);
                    let end = raw.rfind('}').map(|i| i + 1).unwrap_or(raw.len());

                    let methodology_container = serde_json::json!({
                        "container_type": "Methodology",
                        "metadata": {
                            "name": domain,
                            "description": format!("Auto-synthesized methodology for: {}", domain),
                            "created_by": "orchestrator_layer_crossref"
                        },
                        "context": {
                            "keywords": [domain.to_lowercase()],
                            "topics": [domain.to_lowercase()]
                        },
                        "storage": serde_json::from_str::<serde_json::Value>(&raw[start..end])
                            .unwrap_or_default()
                    });

                    if let Ok(new_id) = self.zsei.create_container(0, methodology_container).await {
                        state.methodologies.push(new_id);
                        findings.push(MethodologyFinding::Created(new_id));
                    }
                }
            }
        }

        findings
    }

    fn record_stage(
        &self,
        state: &mut OrchestrationState,
        stage: u8,
        name: &str,
        success: bool,
        summary: &str,
    ) {
        state.stages.push(StageResult {
            stage,
            name: name.to_string(),
            success,
            duration_ms: 0,
            output_summary: Some(summary.to_string()),
        });
    }

    fn record_stage_timed(
        &self,
        state: &mut OrchestrationState,
        stage: u8,
        name: &str,
        success: bool,
        summary: &str,
        duration_ms: u64,
    ) {
        state.stages.push(StageResult {
            stage,
            name: name.to_string(),
            success,
            duration_ms,
            output_summary: Some(summary.to_string()),
        });
    }

    fn build_success_response(&self, state: &OrchestrationState) -> OrchestrationResponse {
        OrchestrationResponse {
            success: !state.needs_clarification,
            response: state.final_response.clone(),
            task_id: state.task_id,
            blueprint_id: state.blueprint_id,
            stages_completed: state.stages.clone(),
            consciousness_gate: state.gate_result.clone(),
            error: None,
            total_tokens_used: Some(state.tokens_used_so_far),
            execution_time_ms: state.start_time.elapsed().as_millis() as u64,
            methodologies_used: state.methodologies.clone(),
            categories_created: state.categories_created,
            blueprints_created: state.blueprints_created,
            clarification_points: state.clarification_points.clone(),
            needs_clarification: state.needs_clarification,
            amt_summary: state.amt.as_ref().map(|amt| AMTSummary {
                total_nodes: amt.count_nodes(),
                branch_count: amt.branch_count(),
                max_depth: amt.max_depth(),
                validation_status: if state.amt_validated {
                    "Validated".to_string()
                } else {
                    format!("Streak: {}/5", state.validation_streak)
                },
            }),
        }
    }

    fn build_error_response(
        &self,
        state: &mut OrchestrationState,
        error: String,
    ) -> OrchestrationResponse {
        OrchestrationResponse {
            success: false,
            response: None,
            task_id: state.task_id,
            blueprint_id: state.blueprint_id,
            stages_completed: state.stages.clone(),
            consciousness_gate: state.gate_result.clone(),
            error: Some(error),
            total_tokens_used: Some(state.tokens_used_so_far),
            execution_time_ms: state.start_time.elapsed().as_millis() as u64,
            methodologies_used: state.methodologies.clone(),
            categories_created: state.categories_created,
            blueprints_created: state.blueprints_created,
            clarification_points: state.clarification_points.clone(),
            needs_clarification: state.needs_clarification,
            amt_summary: None,
        }
    }

    fn estimate_tokens(text: &str) -> u32 {
        ((text.len() + 3) / 4) as u32
    }

    fn parse_json_object(s: &str) -> serde_json::Value {
        let trimmed = s.trim();
        let json_str = if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                &trimmed[start..=end]
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        serde_json::from_str(json_str).unwrap_or_else(|_| serde_json::json!({}))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct MockExecutor;

    #[async_trait::async_trait]
    impl PipelineExecutor for MockExecutor {
        async fn execute(
            &self,
            pipeline_id: u64,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, String> {
            match pipeline_id {
                9 => Ok(serde_json::json!({
                    "response": "Test response from LLM",
                    "tokens_used": 100
                })),
                100 => {
                    // Text modality
                    let action_type = input
                        .get("action")
                        .and_then(|a| a.get("type"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");

                    match action_type {
                        "Analyze" => Ok(serde_json::json!({
                            "analysis": {
                                "word_count": 5,
                                "sentence_count": 1,
                                "paragraph_count": 1,
                                "character_count": 25,
                                "entities": [],
                                "topics": [],
                                "keywords": [],
                                "structure": {
                                    "sections": [],
                                    "has_title": false,
                                    "has_abstract": false,
                                    "has_toc": false,
                                    "document_type": "Unknown"
                                }
                            },
                            "processed_chunks": [{
                                "index": 0,
                                "original_text": "Test",
                                "cleaned_text": "Test cleaned",
                                "start_offset": 0,
                                "end_offset": 12,
                                "token_count": 3,
                                "keywords": ["test"],
                                "entities": [],
                                "topics": ["testing"],
                                "overlap_from_previous": 0,
                                "overlap_to_next": 0
                            }],
                            "chunk_graph": {
                                "graph_id": 42
                            }
                        })),
                        "ReconstructFromChunks" => Ok(serde_json::json!({
                            "reconstructed_text": "Test cleaned text"
                        })),
                        _ => Ok(serde_json::json!({"success": true})),
                    }
                }
                _ => Ok(serde_json::json!({"success": true})),
            }
        }
    }

    struct MockZSEI;

    #[async_trait::async_trait]
    impl ZSEIAccess for MockZSEI {
        async fn query(&self, _query: serde_json::Value) -> Result<serde_json::Value, String> {
            Ok(serde_json::json!({"containers": []}))
        }

        async fn traverse(&self, _request: serde_json::Value) -> Result<serde_json::Value, String> {
            Ok(serde_json::json!({"results": []}))
        }

        async fn create_container(
            &self,
            _parent_id: u64,
            _container: serde_json::Value,
        ) -> Result<u64, String> {
            Ok(1001)
        }

        async fn update_container(
            &self,
            _container_id: u64,
            _updates: serde_json::Value,
        ) -> Result<(), String> {
            Ok(())
        }

        async fn get_container(
            &self,
            _container_id: u64,
        ) -> Result<Option<serde_json::Value>, String> {
            Ok(Some(serde_json::json!({
                "local_state": {
                    "context": {
                        "keywords": ["test"],
                        "categories": []
                    },
                    "storage": {
                        "principles": ["Consider error handling", "Ensure security"]
                    }
                }
            })))
        }

        async fn search_by_keywords(
            &self,
            _keywords: &[String],
            _container_type: Option<&str>,
        ) -> Result<Vec<u64>, String> {
            Ok(vec![])
        }

        async fn get_categories(&self, _modality: &str) -> Result<Vec<u64>, String> {
            Ok(vec![])
        }
    }

    // Implement ZSEIAccess for task module too
    #[async_trait::async_trait]
    impl crate::task::ZSEIAccess for MockZSEI {
        async fn get_container(
            &self,
            _container_id: u64,
        ) -> Result<Option<serde_json::Value>, String> {
            Ok(Some(serde_json::json!({
                "local_state": {
                    "context": { "keywords": ["test"] },
                    "storage": { "principles": ["test principle"] }
                }
            })))
        }

        async fn search_by_keywords(
            &self,
            _keywords: &[String],
            _container_type: Option<&str>,
        ) -> Result<Vec<u64>, String> {
            Ok(vec![])
        }

        async fn get_categories(&self, _modality: &str) -> Result<Vec<u64>, String> {
            Ok(vec![])
        }

        async fn create_container(
            &self,
            _parent_id: u64,
            _container: serde_json::Value,
        ) -> Result<u64, String> {
            Ok(1001)
        }

        async fn update_container(
            &self,
            _container_id: u64,
            _updates: serde_json::Value,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_basic_orchestration() {
        let executor = Arc::new(MockExecutor);
        let zsei = Arc::new(MockZSEI);

        let task_config = TaskQueueConfig {
            consciousness_enabled: false,
            storage_path: "/tmp/test_tasks".to_string(),
            ..Default::default()
        };
        let refinement_config = RefinementConfig {
            enabled: false,
            ..Default::default()
        };
        let task_manager = Arc::new(TaskManager::new(task_config, refinement_config).unwrap());

        let orchestrator =
            PromptOrchestrator::new(executor, zsei, task_manager, Arc::new(RwLock::new(None)));

        let request = OrchestrationRequest {
            prompt: "Hello, how are you?".to_string(),
            project_id: None,
            workspace_id: None,
            user_id: 1,
            device_id: 1,
            consciousness_enabled: false,
            token_budget: Some(10000),
            model_config: None,
            attached_files: Vec::new(),
        };

        let response = orchestrator.orchestrate(request).await;

        assert!(!response.stages_completed.is_empty());
        // First stage should always complete
        assert_eq!(response.stages_completed[0].stage, 1);
        assert!(response.stages_completed[0].success);
    }

    #[test]
    fn test_amt_node_counting() {
        let mut root = AMTNode::new(1, AMTNodeType::Root, "Root".to_string(), 0);
        let mut branch = AMTNode::new(2, AMTNodeType::Branch, "Branch".to_string(), 1);
        branch
            .children
            .push(AMTNode::new(3, AMTNodeType::Leaf, "Leaf".to_string(), 2));
        root.children.push(branch);

        assert_eq!(root.count_nodes(), 3);
        assert_eq!(root.branch_count(), 1);
        assert_eq!(root.max_depth(), 2);
    }

    #[test]
    fn test_parse_json_object() {
        let result = PromptOrchestrator::parse_json_object(
            r#"Some text before {"key": "value"} some text after"#,
        );
        assert_eq!(result.get("key").and_then(|v| v.as_str()), Some("value"));

        let empty_result = PromptOrchestrator::parse_json_object("no json here");
        assert!(empty_result
            .as_object()
            .map(|o| o.is_empty())
            .unwrap_or(true));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(PromptOrchestrator::estimate_tokens("test"), 1);
        assert_eq!(
            PromptOrchestrator::estimate_tokens("test test test test"),
            5
        );
        assert_eq!(PromptOrchestrator::estimate_tokens(""), 0);
    }
}

this is my modalities/text/main.rs

//! OZONE Studio - Text Modality Pipeline (ID: 100)
//!
//! Analyzes text and creates structural graphs for:
//! - Entities (people, places, organizations) via zero-shot LLM
//! - Topics and themes via zero-shot LLM
//! - Keywords via zero-shot LLM
//! - Document structure (sections, paragraphs)
//! - Relationships between concepts
//! - Cross-references
//!
//! This pipeline uses zero-shot LLM calls for accurate extraction
//! instead of regex-based patterns.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn default_version() -> u32 {
    1
}
fn default_hotness() -> f32 {
    0.5
}

// ============================================================================
// PIPELINE METADATA
// ============================================================================

pub const PIPELINE_ID: u64 = 100;
pub const PIPELINE_NAME: &str = "TextAnalysisPipeline";
pub const PIPELINE_VERSION: &str = "0.4.0";
pub const PIPELINE_MODALITY: &str = "text";

// ═══════════════════════════════════════════════════════════════════════════
// UNIVERSAL TYPES (per-pipeline copy; JSON is the contract)
// ═══════════════════════════════════════════════════════════════════════════

/// Source-object provenance for edges and nodes.
/// Answers: "what object was this derived from?"
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum EdgeProvenance {
    #[default]
    Unknown,
    DerivedFromPrompt,
    DerivedFromChunk(u32),
    DerivedFromChunkGraph(u64),
    DerivedFromModalityGraph(u64),
    DerivedFromFile(String),
    DerivedFromFileGraph(u64),
    DerivedFromAMT,
    DerivedFromAMTBranch,
    DerivedFromBlueprint(u32),
    DerivedFromBlueprintStep(u32),
    DerivedFromMethodology(u64),
    DerivedFromCrossModal,
    DerivedFromHook,
    VersionOf(u32),
    ForkedFrom(u64),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ProvisionalStatus {
    #[default]
    Planned,
    Generating,
    Generated,
    Validated,
    Finalized,
    Failed,
    RolledBack,
}

/// Merged ChangeType — both drafts combined.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ChangeType {
    #[default]
    Created,
    Updated,
    Modified,
    RelationshipAdded,
    CrossModalLinkAdded,
    EnrichedBySemantic,
    EnrichedByHook,
    ProvisionalFinalized,
    RolledBack,
    Finalized,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VersionNote {
    pub version: u32,
    pub note: String,
    pub step_index: Option<u32>,
    pub timestamp: String,
    pub change_type: ChangeType,
}

/// Inline cross-modal reference stored on each node for fast lookup.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CrossModalRef {
    pub target_graph_id: u64,
    pub target_node_id: u64,
    pub target_modality: String,
    pub relationship: String,
    pub bidirectional: bool,
}

/// Graph lifecycle state machine.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum GraphStateType {
    #[default]
    Created,
    SemanticEnriched,
    CrossLinked,
    Stable,
    Updated,
    ReValidating,
    Failed,
    Archived,
}

/// Convenience struct for cross-pipeline graph state tracking.
/// Extracted from node fields; used by orchestrator queries.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeCore {
    pub node_id: u64,
    pub node_type: String,
    pub label: String,
    pub content: String,
    pub provisional: bool,
    pub provisional_status: ProvisionalStatus,
    pub provenance: EdgeProvenance,
    pub source_chunk_id: Option<u64>,
    pub source_file_id: Option<u64>,
    pub created_by_step: Option<u32>,
    pub updated_by_step: Option<u32>,
    pub version: u32,
    pub version_notes: Vec<VersionNote>,
    pub materialized_path: Option<String>,
    pub keywords: Vec<String>,
    pub embedding_hint: Option<String>,
    pub hotness_score: f32,
    pub source_chunk_index: Option<u32>,
    pub source_start_char: Option<usize>,
    pub source_end_char: Option<usize>,
    pub cross_modal_refs: Vec<CrossModalRef>,
    pub time_range_start: Option<f32>,
    pub time_range_end: Option<f32>,
}

/// Standard fields for ALL modality graph edges.
/// These are added inline to each pipeline's edge struct.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EdgeMeta {
    pub provenance: EdgeProvenance,
    pub created_by_step: Option<u32>,
    pub version: u32,
    pub version_notes: Vec<VersionNote>,
    pub is_cross_modal: bool,
    pub cross_modal_index_id: Option<u64>,
}

// ============================================================================
// EXECUTOR TRAIT
// ============================================================================

/// Trait for executing other pipelines (injected by runtime)
#[async_trait::async_trait]
pub trait PipelineExecutor: Send + Sync {
    async fn execute(
        &self,
        pipeline_id: u64,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, String>;
}

// ============================================================================
// PROCESSING PATHS & STRUCTURED NODES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingPath {
    Path1, // Clean chunk as a whole
    Path2, // Granular 1-by-1 sentence extraction and validation
}

impl Default for ProcessingPath {
    fn default() -> Self {
        ProcessingPath::Path2
    }
}

#[derive(Debug, Clone, Default)]
pub struct SectionState {
    pub current_section_id: Option<u64>,
    pub current_level: Option<u8>,
    pub current_title: Option<String>,
    pub formatting_pattern: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ParagraphState {
    pub current_paragraph_id: Option<u64>,
    pub open_paragraph: bool,
    pub start_offset_in_chunk: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum GrammarNodeType {
    Sentence,
    MainClause,
    SubordinateClause,
    RelativeClause,
    ComplementClause,
    AdverbialClause,
    ConditionalClause,
    ComparativeClause,
    CoordinateClause,
    ParentheticalClause,
    EllipticalClause,
    QuotedClause,
    Phrase,
    NounPhrase,
    VerbPhrase,
    AdjectivePhrase,
    AdverbPhrase,
    PrepositionalPhrase,
    InfinitivePhrase,
    ParticipialPhrase,
    GerundPhrase,
    AbsolutePhrase,
    AppositivePhrase,
    CoordinatePhrase,
    ParentheticalPhrase,
    Predicate,
    Verb,
    MainVerb,
    AuxiliaryVerb,
    ModalVerb,
    LinkingVerb,
    HelpingVerb,
    PhrasalVerb,
    Copula,
    PredicateComplement,
    PredicateNominative,
    PredicateAdjective,
    Subject,
    ImpliedSubject,
    ExpletiveSubject,
    DirectObject,
    IndirectObject,
    ObjectComplement,
    SubjectComplement,
    Complement,
    Noun,
    CommonNoun,
    ProperNoun,
    CollectiveNoun,
    MassNoun,
    CountNoun,
    ConcreteNoun,
    AbstractNoun,
    CompoundNoun,
    Pronoun,
    PersonalPronoun,
    ReflexivePronoun,
    ReciprocalPronoun,
    RelativePronoun,
    DemonstrativePronoun,
    InterrogativePronoun,
    IndefinitePronoun,
    PossessivePronoun,
    Determiner,
    Article,
    Demonstrative,
    PossessiveDeterminer,
    Quantifier,
    Numeral,
    DistributiveDeterminer,
    Modifier,
    Adjective,
    AttributiveAdjective,
    PredicativeAdjective,
    ComparativeAdjective,
    SuperlativeAdjective,
    Adverb,
    AdverbOfTime,
    AdverbOfPlace,
    AdverbOfManner,
    AdverbOfDegree,
    AdverbOfFrequency,
    SentenceAdverb,
    Preposition,
    SimplePreposition,
    CompoundPreposition,
    PhrasalPreposition,
    Conjunction,
    CoordinatingConjunction,
    SubordinatingConjunction,
    CorrelativeConjunction,
    RelativePronounClause,
    RelativeModifier,
    AdjectivalModifier,
    AdverbialModifier,
    NominalModifier,
    DeterminerModifier,
    PossessiveModifier,
    NumericModifier,
    QuantifierModifier,
    Apposition,
    Vocative,
    Parenthetical,
    PrepositionalObject,
    ObjectOfPreposition,
    ComplementOfPreposition,
    InfinitiveMarker,
    Infinitive,
    Gerund,
    PresentParticiple,
    PastParticiple,
    ClauseSubject,
    ClausePredicate,
    ClauseObject,
    ClauseComplement,
    ClauseModifier,
    Negation,
    NegativeMarker,
    NegativeDeterminer,
    NegativePronoun,
    QuestionMarker,
    InterrogativeWord,
    TagQuestion,
    WhPhrase,
    Comparison,
    ComparativeMarker,
    SuperlativeMarker,
    EqualityMarker,
    Coordination,
    Coordinator,
    CoordinatedElement,
    AgreementMarker,
    CaseMarker,
    NumberMarker,
    GenderMarker,
    PersonMarker,
    Punctuation,
    Comma,
    Period,
    Colon,
    Semicolon,
    Dash,
    Hyphen,
    Parenthesis,
    Quotation,
    Apostrophe,
    Ellipsis,
    Exclamation,
    QuestionMark,
    Token,
    Word,
    Symbol,
    Number,
    Letter,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrammarProperties {
    pub tense: Option<String>,
    pub aspect: Option<String>,
    pub mood: Option<String>,
    pub voice: Option<String>,
    pub person: Option<String>,
    pub number: Option<String>,
    pub gender: Option<String>,
    pub case: Option<String>,
    pub definiteness: Option<String>,
    pub comparison: Option<String>,
    pub polarity: Option<String>,
    pub subtype: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarNode {
    pub node_id: u64,
    pub node_type: GrammarNodeType,
    pub text: String,
    pub position: TextPosition,
    pub children: Vec<GrammarNode>,
    pub properties: GrammarProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceNode {
    pub node_id: u64,
    pub node_type: GrammarNodeType,
    pub content: String,
    pub original_content: String,
    pub position: TextPosition,
    pub chunk_id: u32,
    pub chunk_offset: usize,
    pub paragraph_id: Option<u64>,
    pub section_id: Option<u64>,
    pub properties: GrammarProperties,
    pub grammar_nodes: Vec<GrammarNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParagraphNode {
    pub node_id: u64,
    pub node_type: GrammarNodeType,
    pub sentence_count: u32,
    pub parent_section: Option<u64>,
    pub position: TextPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionNode {
    pub node_id: u64,
    pub node_type: GrammarNodeType,
    pub content: String,
    pub level: u8,
    pub formatting_pattern: Option<String>,
    pub parent_section: Option<u64>,
    pub position: TextPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentNode {
    pub node_id: u64,
    pub node_type: GrammarNodeType,
    pub content: String,
    pub document_type: String,
    pub title: Option<String>,
    pub position: TextPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossSentenceRelationship {
    pub from_sentence_id: u64,
    pub to_sentence_id: u64,
    pub relationship_type: String,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreferenceMention {
    pub sentence_id: u64,
    pub text: String,
    pub grammar_role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreferenceChain {
    pub chain_id: u64,
    pub canonical_form: String,
    pub mentions: Vec<CoreferenceMention>,
}

// ============================================================================
// INPUT/OUTPUT TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct TextModalityInput {
    pub action: TextModalityAction,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TextModalityAction {
    /// Analyze text and create structural representation
    Analyze {
        text: String,
        #[serde(default = "default_max_chunk_tokens")]
        max_chunk_tokens: u32,
        #[serde(default)]
        depth: AnalysisDepth,
        #[serde(default)]
        extract_entities: bool,
        #[serde(default)]
        extract_topics: bool,
        #[serde(default)]
        extract_structure: bool,
        #[serde(default)]
        available_modalities: Vec<String>,
        #[serde(default)]
        processing_path: ProcessingPath,
    },

    /// Create a graph from analysis results
    CreateGraph {
        analysis_result: TextAnalysisResult,
        project_id: u64,
        #[serde(default)]
        link_to_existing: bool,
    },

    /// Update existing graph with new text
    UpdateGraph { graph_id: u64, delta: TextDelta },

    /// Query the text graph
    QueryGraph {
        graph_id: u64,
        query: TextGraphQuery,
    },

    /// Get existing graph
    GetGraph { graph_id: u64 },

    /// Clean and normalize a text chunk via zero-shot LLM
    CleanChunk { chunk: RawChunk },

    /// Reconstruct cleaned prompt from processed chunks
    ReconstructFromChunks { chunks: Vec<ProcessedChunk> },

    /// Extract keywords via zero-shot LLM
    ExtractKeywords {
        text: String,
        #[serde(default = "default_max_keywords")]
        max_keywords: usize,
    },

    /// Extract entities via zero-shot LLM
    ExtractEntities { text: String },

    /// Extract topics via zero-shot LLM
    ExtractTopics { text: String },

    /// Detect modalities present in a chunk via zero-shot (5x stable loop)
    DetectModalities {
        text: String,
        include_true_text: bool,
        dynamic_modality_list: bool,
        #[serde(default)]
        available_modalities: Vec<String>,
    },

    /// Create a persistent chunk graph from a processed chunk
    CreateChunkGraph {
        chunk: ProcessedChunk,
        root_graph_id: u64,
    },

    /// Extract grammar relationships via zero-shot LLM
    ExtractGrammarRelationships { text: String, chunk_index: u32 },

    /// Normalize text (basic cleaning without LLM)
    Normalize {
        text: String,
        #[serde(default)]
        chunk_size: Option<usize>,
        #[serde(default)]
        overlap: usize,
    },

    /// Trigger ZSEI semantic analysis hook
    TriggerSemanticHook {
        graph_id: u64,
        hook_type: ZSEIHookType,
    },

    /// Link to another modality graph
    LinkToModality {
        source_graph_id: u64,
        target_graph_id: u64,
        target_modality: String,
        relationship: CrossModalityRelation,
    },
}

fn default_max_keywords() -> usize {
    20
}
fn default_max_chunk_tokens() -> u32 {
    2000
}
fn default_overlap_tokens() -> u32 {
    200
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
pub enum AnalysisDepth {
    Surface,
    #[default]
    Standard,
    Deep,
    Comprehensive,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextModalityOutput {
    pub success: bool,
    pub error: Option<String>,

    // Analysis results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analysis: Option<TextAnalysisResult>,

    // Graph results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_id: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<TextGraph>,

    // Chunking results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<Vec<RawChunk>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_chunks: Option<Vec<ProcessedChunk>>,

    // Normalization results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized_text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cleaned_text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reconstructed_text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<u32>,

    // Extraction results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<Keyword>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub entities: Option<Vec<Entity>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub topics: Option<Vec<String>>,

    // Hook results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hook_result: Option<HookResult>,

    // Cross-modality link result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_result: Option<LinkResult>,

    // Metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_relationships: Option<Vec<ChunkGrammarRelationship>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub modality_detections: Option<Vec<ChunkModalityDetection>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_graph: Option<ChunkGraph>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentence_analyses: Option<Vec<SentenceAnalysis>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub true_text_spans: Option<Vec<TextSpan>>,
}

impl Default for TextModalityOutput {
    fn default() -> Self {
        Self {
            success: false,
            error: None,
            analysis: None,
            graph_id: None,
            graph: None,
            chunks: None,
            processed_chunks: None,
            normalized_text: None,
            cleaned_text: None,
            reconstructed_text: None,
            token_count: None,
            keywords: None,
            entities: None,
            topics: None,
            hook_result: None,
            link_result: None,
            processing_time_ms: None,
            grammar_relationships: None,
            modality_detections: None,
            chunk_graph: None,
            sentence_analyses: None,
            true_text_spans: None,
        }
    }
}

// ── OVERLAP RESOLUTION TYPES ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OverlapResolution {
    pub has_overlap: bool,
    pub overlap_type: OverlapType,
    pub current_keep_end: usize, // char index in original current chunk text
    pub next_start_offset: usize, // char offset to skip in next chunk
    pub duplicate_belongs_in: DuplicateOwner,
    pub resolution_method: ResolutionMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OverlapType {
    #[default]
    None,
    SentenceCutoff,
    ParagraphCutoff,
    WordCutoff,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DuplicateOwner {
    #[default]
    CurrentChunk,
    NextChunk,
    Neither,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ResolutionMethod {
    #[default]
    LLMZeroShot,
    RuleBased, // emergency fallback only — not default path
}

// ── GRAMMAR TYPES ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum VerbType {
    #[default]
    Action, // run, create, modify
    Linking, // is, seems, becomes
    Helping, // has, will, can
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkGrammarRelationship {
    pub from_text: String,
    pub to_text: String,
    pub edge_type: String,
    pub tense: Option<String>,
    pub negated: bool,
    pub verb: String,
    pub verb_type: VerbType,
    pub subject: String,
    pub object: Option<String>,
    pub source_sentence_start: Option<usize>,
    pub source_sentence_end: Option<usize>,
    pub chunk_index: u32,
}

// ── MODALITY DETECTION TYPES ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkModalityDetection {
    pub modality: String,
    pub span_start: usize,
    pub span_end: usize,
    pub intent_reference: String,
    pub chunk_index: u32,
    pub parent_node_id: Option<u64>, // Tied to SentenceNode or ParagraphNode
}

// ── CHUNK GRAPH TYPES ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceBoundary {
    pub start: usize, // char offset relative to chunk start
    pub end: usize,
    pub sentence_type: SentenceType,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SentenceType {
    #[default]
    Declarative,
    Interrogative,
    Imperative,
    Fragment,
    CodeBlock,
    MathExpression,
}

/// Chunk graph — one per processed chunk. Persistent historical evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkGraph {
    pub graph_id: u64,
    pub chunk_index: u32,
    pub prompt_start_char: usize,
    pub prompt_end_char: usize,
    pub sentence_boundaries: Vec<SentenceBoundary>,
    pub paragraph_breaks: Vec<usize>,
    pub cleaned_text: String,
    pub overlap_resolution: Option<OverlapResolution>,
    pub keywords: Vec<String>,
    pub topics: Vec<String>,
    pub sentence_nodes: Vec<SentenceNode>,
    pub paragraph_nodes: Vec<ParagraphNode>,
    pub section_nodes: Vec<SectionNode>,
    pub document_nodes: Vec<DocumentNode>,
    pub cross_sentence_relationships: Vec<CrossSentenceRelationship>,
    pub coreference_chains: Vec<CoreferenceChain>,
    pub modality_detections: Vec<ChunkModalityDetection>,
    pub root_modality_list_contribution: Vec<String>,
    pub created_at: String,
}

/// Result of smart context reconstruction from chunk graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructedContext {
    pub text: String,
    pub included_chunk_indices: Vec<u32>,
    pub total_chars: usize,
    pub estimated_tokens: usize,
}

// ============================================================================
// CHUNK TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RawChunk {
    pub index: u32,
    pub text: String,
    pub start_char: u32,
    pub end_char: u32,
    pub token_count: u32,
    #[serde(default)]
    pub is_complete_paragraph: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessedChunk {
    pub index: u32,
    pub original_text: String,
    pub cleaned_text: String,
    pub start_offset: u32,
    pub end_offset: u32,
    pub token_count: u32,
    pub keywords: Vec<String>,
    pub entities: Vec<ExtractedEntity>,
    pub topics: Vec<String>,
    #[serde(default)]
    pub overlap_from_previous: u32,
    #[serde(default)]
    pub overlap_to_next: u32,
    pub sentence_nodes: Vec<SentenceNode>,
    pub paragraph_nodes: Vec<ParagraphNode>,
    pub section_nodes: Vec<SectionNode>,
    pub document_nodes: Vec<DocumentNode>,
    pub cross_sentence_relationships: Vec<CrossSentenceRelationship>,
    pub coreference_chains: Vec<CoreferenceChain>,
    pub detected_modalities: Vec<ChunkModalityDetection>,
    pub chunk_graph_id: Option<u64>,
    pub prompt_start_char: usize,
    pub prompt_end_char: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,
    pub confidence: f32,
    pub start_offset: Option<usize>,
    pub end_offset: Option<usize>,
}

// ============================================================================
// ANALYSIS TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextAnalysisResult {
    pub word_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub character_count: usize,
    pub entities: Vec<Entity>,
    pub topics: Vec<Topic>,
    pub keywords: Vec<Keyword>,
    pub structure: DocumentStructure,
    pub language: Option<String>,
    pub sentiment: Option<Sentiment>,
    pub readability_score: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub start_offset: usize,
    pub end_offset: usize,
    pub confidence: f32,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Time,
    Money,
    Percentage,
    Product,
    Event,
    Technology,
    Concept,
    Custom(String),
}

impl EntityType {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "PERSON" | "PER" => EntityType::Person,
            "ORGANIZATION" | "ORG" => EntityType::Organization,
            "LOCATION" | "LOC" | "GPE" => EntityType::Location,
            "DATE" => EntityType::Date,
            "TIME" => EntityType::Time,
            "MONEY" | "CURRENCY" => EntityType::Money,
            "PERCENTAGE" | "PERCENT" => EntityType::Percentage,
            "PRODUCT" => EntityType::Product,
            "EVENT" => EntityType::Event,
            "TECHNOLOGY" | "TECH" => EntityType::Technology,
            "CONCEPT" => EntityType::Concept,
            other => EntityType::Custom(other.to_string()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Topic {
    pub name: String,
    pub keywords: Vec<String>,
    pub relevance: f32,
    pub category: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Keyword {
    pub term: String,
    pub frequency: usize,
    pub relevance: f32,
    pub is_phrase: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DocumentStructure {
    pub sections: Vec<Section>,
    pub has_title: bool,
    pub has_abstract: bool,
    pub has_toc: bool,
    pub document_type: DocumentType,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Section {
    pub id: String,
    pub title: Option<String>,
    pub level: u8,
    pub start_offset: usize,
    pub end_offset: usize,
    pub subsections: Vec<Section>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum DocumentType {
    Article,
    Report,
    Email,
    Code,
    Chat,
    Documentation,
    Prompt,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Sentiment {
    pub overall: f32,
    pub positive: f32,
    pub negative: f32,
    pub neutral: f32,
}

// ============================================================================
// GRAPH TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextGraph {
    pub graph_id: u64,
    pub modality: String,
    pub version: String,
    pub nodes: Vec<TextGraphNode>,
    pub edges: Vec<TextGraphEdge>,
    pub metadata: HashMap<String, Value>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextGraphNode {
    pub node_id: u64,
    pub node_type: TextNodeType,
    pub content: String,
    pub position: Option<TextPosition>,
    pub properties: HashMap<String, Value>,
    pub semantic_annotations: Vec<SemanticAnnotation>,
    // UNIVERSAL NODE FIELDS
    #[serde(default)]
    pub provisional: bool,
    #[serde(default)]
    pub provisional_status: ProvisionalStatus,
    #[serde(default)]
    pub provenance: EdgeProvenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_chunk_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_by_step: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_by_step: Option<u32>,
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub version_notes: Vec<VersionNote>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub materialized_path: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_hint: Option<String>,
    #[serde(default = "default_hotness")]
    pub hotness_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_chunk_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_start_char: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_end_char: Option<usize>,
    #[serde(default)]
    pub cross_modal_refs: Vec<CrossModalRef>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TextNodeType {
    Document,
    Section,
    Paragraph,
    Sentence,
    Entity,
    Topic,
    Keyword,
    Reference,
    Chunk,
    ModalityReference,
    TrueTextSpan,
    FileReference,
    ChunkReference,
    SupplementarySection,
    GrammarSubject,
    GrammarObject,
    InferredConcept,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextPosition {
    pub start_offset: usize,
    pub end_offset: usize,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SemanticAnnotation {
    pub annotation_type: String,
    pub value: String,
    pub confidence: f32,
    pub source: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextGraphEdge {
    pub edge_id: u64,
    pub from_node: u64,
    pub to_node: u64,
    pub edge_type: TextEdgeType,
    pub weight: f32,
    pub properties: HashMap<String, Value>,
    // UNIVERSAL EDGE FIELDS
    #[serde(default)]
    pub provenance: EdgeProvenance,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_by_step: Option<u32>,
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub version_notes: Vec<VersionNote>,
    #[serde(default)]
    pub is_cross_modal: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_modal_index_id: Option<u64>,
    // Grammar info for grammar-derived edges
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_info: Option<GrammarEdgeInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarEdgeInfo {
    pub verb: String,
    pub verb_type: VerbType,
    pub tense: String,
    pub negated: bool,
    pub source_sentence: String,
}

/// Sentence-level analysis used in SentenceAnalysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentInfo {
    pub label: String, // positive / negative / neutral
    pub score: f32,
}

/// A span that is genuinely text (not embedded code/math/etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    pub start_char: usize,
    pub end_char: usize,
    pub chunk_index: u32,
    pub is_true_text: bool,
}

/// Full sentence analysis produced per-chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceAnalysis {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub subject: String,
    pub verb: String,
    pub verb_type: VerbType,
    pub object: Option<String>,
    pub tense: String,
    pub relationships: Vec<ChunkGrammarRelationship>,
    pub sentiment: Option<SentimentInfo>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TextEdgeType {
    // Structural
    Contains,
    ContainedBy,
    Follows,
    Precedes,

    // Semantic (added by ZSEI)
    References,
    Contradicts,
    Supports,
    Elaborates,
    Summarizes,

    // Cross-modality
    DescribesCode,
    DescribesImage,
    DescribesAudio,
    DescribesVideo,
    TranscribedFrom,

    // Universal Semantic
    Performs,
    Affects,
    Implies,
    TemporalPrecedes,
    TemporalFollows,
    CausedBy,
    Enables,
    Prevents,
    PartOf,
    HasPart,
    FunctionalRole,
    InstanceOf,
    HasInstance,
    SimilarTo,
    RelatesTo,

    // Versioning / Derivation
    DerivedFrom,
    VersionOf,
    RefinesTo,
    ForkedFrom,

    // Cross-Modality
    ReferencesModality,
    ReferencedBy,
    DescribedBy,
    Describes,
    ImplementedIn,
    Implements,
    VisualizedAs,
    Visualizes,
    SyncedTo,
    SyncedBy,
    AnnotatedBy,

    // Supplementary
    SupplementsPrompt,
    ContextProvides,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextDelta {
    pub operation: DeltaOperation,
    pub position: Option<TextPosition>,
    pub content: Option<String>,
    pub affected_nodes: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum DeltaOperation {
    Insert,
    Delete,
    Replace,
    Reorder,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextGraphQuery {
    pub query_type: TextQueryType,
    pub parameters: HashMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum TextQueryType {
    FindEntities,
    FindTopics,
    FindKeywords,
    FindReferences,
    GetStructure,
    SemanticSearch,
    PathBetween,
    GetNodesByType,
    Custom,
}

// ============================================================================
// CROSS-MODALITY TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum CrossModalityRelation {
    DescribesCode,
    DescribesImage,
    DescribesAudio,
    DescribesVideo,
    TranscribedFrom,
    IllustratedBy,
    Custom(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LinkResult {
    pub link_id: u64,
    pub source_graph_id: u64,
    pub target_graph_id: u64,
    pub relationship: CrossModalityRelation,
    pub created_at: String,
}

// ============================================================================
// ZSEI HOOK TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum ZSEIHookType {
    OnGraphCreated,
    OnEdgeCompletion,
    OnInferRelationships,
    OnCrossModalityLink,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HookResult {
    pub hook_type: ZSEIHookType,
    pub success: bool,
    pub nodes_processed: usize,
    pub edges_added: usize,
    pub annotations_added: usize,
    pub processing_time_ms: u64,
    pub errors: Vec<String>,
}

// ============================================================================
// PIPELINE IMPLEMENTATION
// ============================================================================

pub struct TextModalityPipeline {
    executor: Arc<dyn PipelineExecutor>,
    graph_cache: tokio::sync::RwLock<HashMap<u64, TextGraph>>,
}

impl TextModalityPipeline {
    pub fn new(executor: Arc<dyn PipelineExecutor>) -> Self {
        Self {
            executor,
            graph_cache: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    pub async fn execute(&self, input: TextModalityInput) -> TextModalityOutput {
        let start_time = std::time::Instant::now();

        let mut output = match input.action {
            TextModalityAction::Analyze {
                text,
                max_chunk_tokens,
                depth,
                extract_entities,
                extract_topics,
                extract_structure,
                available_modalities,
                processing_path,
            } => {
                self.analyze_text(
                    &text,
                    max_chunk_tokens,
                    depth,
                    extract_entities,
                    extract_topics,
                    extract_structure,
                    &available_modalities,
                    processing_path,
                )
                .await
            }

            TextModalityAction::CreateGraph {
                analysis_result,
                project_id,
                link_to_existing,
            } => {
                self.create_graph(analysis_result, project_id, link_to_existing)
                    .await
            }

            TextModalityAction::UpdateGraph { graph_id, delta } => {
                self.update_graph(graph_id, delta).await
            }

            TextModalityAction::QueryGraph { graph_id, query } => {
                self.query_graph(graph_id, query).await
            }

            TextModalityAction::GetGraph { graph_id } => self.get_graph(graph_id).await,

            TextModalityAction::CreateChunkGraph {
                chunk,
                root_graph_id,
            } => {
                let graph = self.create_chunk_graph(&chunk, root_graph_id);
                TextModalityOutput {
                    success: true,
                    chunk_graph: Some(graph),
                    ..Default::default()
                }
            }

            TextModalityAction::CleanChunk { chunk } => self.clean_chunk(chunk).await,

            TextModalityAction::ReconstructFromChunks { chunks } => {
                self.reconstruct_from_chunks(&chunks)
            }

            TextModalityAction::ExtractKeywords { text, max_keywords } => {
                self.extract_keywords_llm(&text, max_keywords).await
            }

            TextModalityAction::ExtractGrammarRelationships { text, chunk_index } => {
                let rels = self
                    .extract_grammar_relationships_from_text(&text, chunk_index)
                    .await;
                TextModalityOutput {
                    success: true,
                    grammar_relationships: Some(rels),
                    ..Default::default()
                }
            }

            TextModalityAction::ExtractEntities { text } => self.extract_entities_llm(&text).await,

            TextModalityAction::ExtractTopics { text } => self.extract_topics_llm(&text).await,

            TextModalityAction::TriggerSemanticHook {
                graph_id,
                hook_type,
            } => self.trigger_semantic_hook(graph_id, hook_type).await,

            TextModalityAction::DetectModalities {
                text,
                include_true_text,
                dynamic_modality_list,
                available_modalities,
            } => {
                let available = if dynamic_modality_list && !available_modalities.is_empty() {
                    available_modalities
                };

                let detections = self.detect_modalities_stable(&text, 0, &available).await;

                let filtered: Vec<ChunkModalityDetection> = if include_true_text {
                    detections
                } else {
                    detections
                        .into_iter()
                        .filter(|d| d.modality != "true_text")
                        .collect()
                };

                TextModalityOutput {
                    success: true,
                    modality_detections: Some(filtered),
                    ..Default::default()
                }
            }

            TextModalityAction::LinkToModality {
                source_graph_id,
                target_graph_id,
                target_modality,
                relationship,
            } => {
                self.link_to_modality(
                    source_graph_id,
                    target_graph_id,
                    &target_modality,
                    relationship,
                )
                .await
            }
        };

        output.processing_time_ms = Some(start_time.elapsed().as_millis() as u64);
        output
    }

    /// Estimate token count (roughly 4 chars per token for English)
    fn estimate_tokens(text: &str) -> u32 {
        ((text.len() + 3) / 4) as u32
    }

    /// Get overlap text from end of string
    fn get_end_overlap(text: &str, overlap_tokens: u32) -> String {
        let overlap_chars = (overlap_tokens * 4) as usize;
        if text.len() <= overlap_chars {
            text.to_string()
        } else {
            let start = text.len() - overlap_chars;
            let adjusted_start = text[start..]
                .find(' ')
                .map(|i| start + i + 1)
                .unwrap_or(start);
            text[adjusted_start..].to_string()
        }
    }

    /// Resolve overlap boundary between two adjacent chunks via LLM zero-shot.
    /// Returns OverlapResolution. Falls back to rule-based on LLM failure.
    async fn resolve_overlap_llm(
        &self,
        current_chunk_text: &str,
        next_chunk_text: &str,
        overlap_chars: usize,
    ) -> OverlapResolution {
        let current_tail_start = current_chunk_text.len().saturating_sub(overlap_chars);
        let current_tail = &current_chunk_text[current_tail_start..];
        let next_head_end = overlap_chars.min(next_chunk_text.len());
        let next_head = &next_chunk_text[..next_head_end];

        let prompt = format!(
            r#"Analyze the overlapping region between two adjacent text chunks.

    END OF CURRENT CHUNK:
    ---
    {}
    ---

    START OF NEXT CHUNK:
    ---
    {}
    ---

    Determine if there is duplicated content and where the clean boundary should be.

    Return ONLY valid JSON (no explanation, no markdown):
    {{
      "has_overlap": true,
      "overlap_type": "sentence_cutoff|paragraph_cutoff|word_cutoff|none",
      "current_should_end_at_offset": <integer, chars from start of current_tail to keep>,
      "next_should_start_at_offset": <integer, chars into next_head to skip>,
      "duplicate_belongs_in": "current|next|neither"
    }}"#,
            current_tail, next_head
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.05,
            "system_context": "Text boundary analysis. Return only valid JSON object. No explanation."
        });

        match self.executor.execute(9, input).await {
            Ok(result) => {
                let raw = result
                    .get("response")
                    .and_then(|r| r.as_str())
                    .unwrap_or("{}");
                let json_str = Self::extract_json_from_response(raw, '{', '}');
                match serde_json::from_str::<serde_json::Value>(&json_str) {
                    Ok(v) => OverlapResolution {
                        has_overlap: v["has_overlap"].as_bool().unwrap_or(false),
                        overlap_type: match v["overlap_type"].as_str().unwrap_or("none") {
                            "sentence_cutoff" => OverlapType::SentenceCutoff,
                            "paragraph_cutoff" => OverlapType::ParagraphCutoff,
                            "word_cutoff" => OverlapType::WordCutoff,
                            _ => OverlapType::None,
                        },
                        current_keep_end: current_tail_start
                            + v["current_should_end_at_offset"]
                                .as_usize()
                                .unwrap_or(current_tail.len()),
                        next_start_offset: v["next_should_start_at_offset"].as_usize().unwrap_or(0),
                        duplicate_belongs_in: match v["duplicate_belongs_in"]
                            .as_str()
                            .unwrap_or("current")
                        {
                            "next" => DuplicateOwner::NextChunk,
                            "neither" => DuplicateOwner::Neither,
                            _ => DuplicateOwner::CurrentChunk,
                        },
                        resolution_method: ResolutionMethod::LLMZeroShot,
                    },
                    Err(_) => Self::rule_based_overlap_fallback(current_chunk_text, overlap_chars),
                }
            }
            Err(_) => Self::rule_based_overlap_fallback(current_chunk_text, overlap_chars),
        }
    }

    /// Chunk text using LLM-based overlap resolution (canonical path).
    /// Called from the async execute path. Produces RawChunks with resolved boundaries.
    async fn chunk_text(&self, text: &str, max_chunk_tokens: u32) -> Vec<RawChunk> {
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < text.len() {
            let end = (start + max_chunk_tokens).min(text.len());
            let chunk_text = &text[start..end];

            chunks.push(RawChunk {
                index,
                text: chunk_text.to_string(),
                token_count: (chunk_text.len() / 4) as u32,
                start_char: start as u32,
                end_char: end as u32,
                is_complete_paragraph: false,
            });

            index += 1;
            start = if end >= text.len() {
                text.len()
            } else {
                end.saturating_sub(overlap_chars)
            };
        }

        chunks
    }

    /// Split text into sentences
    fn split_sentences(text: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let next_idx = i + c.len_utf8();
                let is_end =
                    next_idx >= text.len() || text[next_idx..].starts_with(char::is_whitespace);
                if is_end {
                    let sentence = text[start..=i].trim();
                    if !sentence.is_empty() {
                        sentences.push(sentence);
                    }
                    start = next_idx;
                }
            }
        }

        // Remaining text
        if start < text.len() {
            let remaining = text[start..].trim();
            if !remaining.is_empty() {
                sentences.push(remaining);
            }
        }

        sentences
    }

    // ========================================================================
    // ZERO-SHOT LLM EXTRACTION METHODS
    // ========================================================================

    /// Clean a chunk via zero-shot LLM
    async fn clean_chunk(&self, chunk: RawChunk) -> TextModalityOutput {
        let clean_prompt = format!(
            r#"Clean and normalize the following text chunk. Fix spelling errors, grammar issues, formatting inconsistencies, and broken words from chunking. Preserve the original meaning exactly.

        TEXT TO CLEAN:
        {}

        Return ONLY valid JSON with no explanation, preamble, or markdown:
        {{"cleaned_text": "the fully cleaned text here"}}"#,
            chunk.text
        );

        let clean_input = serde_json::json!({
            "prompt": clean_prompt,
            "max_tokens": (chunk.token_count + 150) as u32,
            "temperature": 0.1,
            "system_context": "You are a text cleaning assistant. Return only valid JSON: {\"cleaned_text\": \"...\"}. No explanation. No markdown. No preamble."
        });

        match self.executor.execute(9, clean_input).await {
            Ok(result) => {
                let cleaned_text = result
                    .get("response")
                    .and_then(|r| r.as_str())
                    .map(|s| {
                        let json_str = Self::extract_json_from_response(s, '{', '}');
                        serde_json::from_str::<serde_json::Value>(&json_str)
                            .ok()
                            .and_then(|v| v.get("cleaned_text")?.as_str().map(|t| t.to_string()))
                            .unwrap_or_else(|| chunk.text.clone())
                    })
                    .unwrap_or_else(|| chunk.text.clone());

                TextModalityOutput {
                    success: true,
                    cleaned_text: Some(cleaned_text),
                    ..Default::default()
                }
            }
            Err(e) => {
                // Fallback to original text if LLM fails
                TextModalityOutput {
                    success: true,
                    cleaned_text: Some(chunk.text),
                    error: Some(format!("LLM cleaning failed, using original: {}", e)),
                    ..Default::default()
                }
            }
        }
    }

    /// Extract keywords from text via LLM (internal helper)
    async fn extract_keywords_from_text(&self, text: &str) -> Vec<String> {
        let prompt = format!(
            r#"Extract all important keywords and key phrases from this text.
Return as a JSON array of strings. Focus on: topics, concepts, named entities, technical terms.

TEXT:
{}

RESPOND ONLY WITH JSON ARRAY: ["keyword1", "keyword2", ...]"#,
            text
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.2,
            "system_context": "Extract keywords. Respond only with valid JSON array."
        });

        match self.executor.execute(9, input).await {
            Ok(result) => result
                .get("response")
                .and_then(|r| r.as_str())
                .and_then(|s| Self::parse_json_array(s))
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    /// Extract entities from text via LLM (internal helper)
    async fn extract_entities_from_text(&self, text: &str) -> Vec<ExtractedEntity> {
        let prompt = format!(
            r#"Extract named entities from this text.
Return as JSON array with objects: {{"text": "entity", "type": "PERSON|ORG|LOCATION|DATE|PRODUCT|EVENT|TECHNOLOGY|OTHER", "confidence": 0.0-1.0}}

TEXT:
{}

RESPOND ONLY WITH JSON ARRAY."#,
            text
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.2,
            "system_context": "Output only a valid JSON array. No explanation. No markdown code blocks. No preamble. Start directly with [."
        });

        match self.executor.execute(9, input).await {
            Ok(result) => result
                .get("response")
                .and_then(|r| r.as_str())
                .and_then(|s| {
                    let json_str = Self::extract_json_from_response(s, '[', ']');
                    serde_json::from_str::<Vec<serde_json::Value>>(&json_str).ok()
                })
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| {
                            Some(ExtractedEntity {
                                text: v.get("text")?.as_str()?.to_string(),
                                entity_type: v.get("type")?.as_str()?.to_string(),
                                confidence: v.get("confidence")?.as_f64()? as f32,
                                start_offset: None,
                                end_offset: None,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    /// Extract topics from text via LLM (internal helper)
    async fn extract_topics_from_text(&self, text: &str) -> Vec<String> {
        let prompt = format!(
            r#"What are the main topics/themes in this text?
Return as JSON array of topic strings.

TEXT:
{}

RESPOND ONLY WITH JSON ARRAY: ["topic1", "topic2", ...]"#,
            text
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.2,
            "system_context": "Output only a valid JSON array of strings. No explanation. No markdown. Start directly with [."
        });

        match self.executor.execute(9, input).await {
            Ok(result) => result
                .get("response")
                .and_then(|r| r.as_str())
                .and_then(|s| Self::parse_json_array(s))
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    /// Run an async extractor repeatedly until 5 consecutive passes find nothing new.
    /// K must be Eq + Hash. dedup_key extracts the deduplication key from each item.
    async fn extract_strings_until_stable<F, Fut>(
        &self,
        input: &str,
        extractor: F,
        max_rounds: usize,
    ) -> Vec<String>
    where
        F: Fn(String, Vec<String>) -> Fut,
        Fut: std::future::Future<Output = Vec<String>>,
    {
        let mut accumulated: Vec<String> = Vec::new();
        let mut no_new_consecutive = 0u32;
        let stable_threshold = 5u32;

        for _round in 0..max_rounds {
            let candidates = extractor(input.to_string(), accumulated.clone()).await;
            let existing: std::collections::HashSet<String> = accumulated.iter().cloned().collect();
            let truly_new: Vec<String> = candidates
                .into_iter()
                .filter(|s| !existing.contains(s))
                .collect();

            if truly_new.is_empty() {
                no_new_consecutive += 1;
                if no_new_consecutive >= stable_threshold {
                    break;
                }
            } else {
                no_new_consecutive = 0;
                accumulated.extend(truly_new);
            }
        }
        accumulated
    }

    /// Extract keywords via zero-shot LLM (public action)
    async fn extract_keywords_llm(&self, text: &str, max_keywords: usize) -> TextModalityOutput {
        let keywords_raw = self.extract_keywords_from_text(text).await;

        let keywords: Vec<Keyword> = keywords_raw
            .into_iter()
            .take(max_keywords)
            .enumerate()
            .map(|(i, term)| Keyword {
                term,
                frequency: 1,
                relevance: 1.0 - (i as f32 * 0.05).min(0.9),
                is_phrase: false,
            })
            .collect();

        TextModalityOutput {
            success: true,
            keywords: Some(keywords),
            ..Default::default()
        }
    }

    /// Detect modalities present in a chunk via LLM zero-shot.
    /// Runs a 5x consecutive stable loop. Modality list is dynamic from registry.
    /// Returns ChunkModalityDetection with span positions — NO content_snippet.
    async fn detect_modalities_in_chunk(
        &self,
        chunk_text: &str,
        chunk_index: u32,
        available_modalities: &[String],
    ) -> Vec<ChunkModalityDetection> {
        let modality_list = available_modalities.join(", ");

        let prompt = format!(
            r#"Identify all modality content present in or referenced by this text.

    Available modalities to detect: {}

    Also detect:
    - "true_text" — content that is genuinely prose/text (not embedded code/math/etc.)
    - "unknown" — content that does not fit any listed modality

    For each detected modality, span_start and span_end are character offsets within
    the provided text. Do NOT include a content_snippet — positions are sufficient.

    Text chunk (index {}):
    {}

    Return ONLY valid JSON array (no explanation, no markdown):
    [{{
      "modality": "modality_name_or_true_text_or_unknown",
      "span_start": 0,
      "span_end": 100,
      "intent_reference": "describes|contains|references|mentions the modality"
    }}]"#,
            modality_list,
            chunk_index,
            &chunk_text[..chunk_text.len().min(4000)]
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 800,
            "temperature": 0.05,
            "system_context": "Modality detection. Return only valid JSON array. No explanation."
        });

        match self.executor.execute(9, input).await {
            Ok(result) => {
                let raw = result
                    .get("response")
                    .and_then(|r| r.as_str())
                    .unwrap_or("[]");
                let json_str = Self::extract_json_from_response(raw, '[', ']');
                serde_json::from_str::<Vec<serde_json::Value>>(&json_str)
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|v| {
                        Some(ChunkModalityDetection {
                            modality: v["modality"].as_str()?.to_string(),
                            span_start: v["span_start"].as_usize()?,
                            span_end: v["span_end"].as_usize()?,
                            intent_reference: v["intent_reference"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                            chunk_index,
                        })
                    })
                    .collect()
            }
            Err(_) => vec![],
        }
    }

    /// Run modality detection with 5x consecutive stable loop.
    /// available_modalities: loaded from pipeline registry at runtime, never hardcoded.
    async fn detect_modalities_stable(
        &self,
        chunk_text: &str,
        chunk_index: u32,
        available_modalities: &[String],
    ) -> Vec<ChunkModalityDetection> {
        let mut accumulated: Vec<ChunkModalityDetection> = Vec::new();
        let mut no_new_consecutive = 0u32;

        for _round in 0..20usize {
            let candidates = self
                .detect_modalities_in_chunk(chunk_text, chunk_index, available_modalities)
                .await;

            // Dedup key: modality + span_start + span_end
            let existing_keys: std::collections::HashSet<String> = accumulated
                .iter()
                .map(|d| format!("{}:{}:{}", d.modality, d.span_start, d.span_end))
                .collect();

            let truly_new: Vec<ChunkModalityDetection> = candidates
                .into_iter()
                .filter(|d| {
                    !existing_keys
                        .contains(&format!("{}:{}:{}", d.modality, d.span_start, d.span_end))
                })
                .collect();

            if truly_new.is_empty() {
                no_new_consecutive += 1;
                if no_new_consecutive >= 5 {
                    break;
                }
            } else {
                no_new_consecutive = 0;
                accumulated.extend(truly_new);
            }
        }
        accumulated
    }

    /// Create a ChunkGraph from a fully processed chunk.
    /// This is the persistent evidence structure for AMT building.
    fn create_chunk_graph(&self, chunk: &ProcessedChunk, root_graph_id: u64) -> ChunkGraph {
        let graph_id = Self::generate_id();
        let now = chrono::Utc::now().to_rfc3339();

        let mut sentence_boundaries = Vec::new();
        let mut pos = 0usize;
        for (i, c) in chunk.cleaned_text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let end = i + c.len_utf8();
                sentence_boundaries.push(SentenceBoundary {
                    start: pos,
                    end,
                    sentence_type: SentenceType::Declarative,
                });
                pos = end;
            }
        }

        let mut paragraph_breaks = Vec::new();
        let mut search_pos = 0;
        while let Some(found) = chunk.cleaned_text[search_pos..].find("\n\n") {
            let abs = search_pos + found;
            paragraph_breaks.push(abs);
            search_pos = abs + 2;
        }

        ChunkGraph {
            graph_id,
            chunk_index: chunk.index,
            prompt_start_char: chunk.prompt_start_char,
            prompt_end_char: chunk.prompt_end_char,
            sentence_boundaries,
            paragraph_breaks,
            cleaned_text: chunk.cleaned_text.clone(),
            overlap_resolution: None,
            keywords: chunk.keywords.clone(),
            topics: chunk.topics.clone(),
            sentence_nodes: chunk.sentence_nodes.clone(),
            paragraph_nodes: chunk.paragraph_nodes.clone(),
            section_nodes: chunk.section_nodes.clone(),
            document_nodes: chunk.document_nodes.clone(),
            cross_sentence_relationships: chunk.cross_sentence_relationships.clone(),
            coreference_chains: chunk.coreference_chains.clone(),
            modality_detections: chunk.detected_modalities.clone(),
            root_modality_list_contribution: chunk
                .detected_modalities
                .iter()
                .filter(|d| d.modality != "true_text" && d.modality != "unknown")
                .map(|d| d.modality.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect(),
            created_at: now,
        }
    }

    /// Extract entities via zero-shot LLM (public action)
    async fn extract_entities_llm(&self, text: &str) -> TextModalityOutput {
        let extracted = self.extract_entities_from_text(text).await;

        let entities: Vec<Entity> = extracted
            .into_iter()
            .map(|e| Entity {
                text: e.text,
                entity_type: EntityType::from_str(&e.entity_type),
                start_offset: e.start_offset.unwrap_or(0),
                end_offset: e.end_offset.unwrap_or(0),
                confidence: e.confidence,
                metadata: HashMap::new(),
            })
            .collect();

        TextModalityOutput {
            success: true,
            entities: Some(entities),
            ..Default::default()
        }
    }

    /// Extract topics via zero-shot LLM (public action)
    async fn extract_topics_llm(&self, text: &str) -> TextModalityOutput {
        let topics = self.extract_topics_from_text(text).await;

        TextModalityOutput {
            success: true,
            topics: Some(topics),
            ..Default::default()
        }
    }

    // ========================================================================
    // RECONSTRUCTION
    // ========================================================================

    /// Reconstruct full cleaned prompt from processed chunks, handling overlaps
    fn reconstruct_from_chunks(&self, chunks: &[ProcessedChunk]) -> TextModalityOutput {
        if chunks.is_empty() {
            return TextModalityOutput {
                success: true,
                reconstructed_text: Some(String::new()),
                ..Default::default()
            };
        }

        if chunks.len() == 1 {
            return TextModalityOutput {
                success: true,
                reconstructed_text: Some(chunks[0].cleaned_text.clone()),
                ..Default::default()
            };
        }

        let mut result = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 {
                result.push_str(&chunk.cleaned_text);
            } else {
                // Skip the overlapping portion
                let skip_chars = chunk.overlap_from_previous as usize;
                if chunk.cleaned_text.len() > skip_chars {
                    // Find a good boundary (space or newline) near the skip point
                    let text_bytes = chunk.cleaned_text.as_bytes();
                    let mut actual_skip = skip_chars;

                    // Look for space/newline within 50 chars of skip point
                    for j in skip_chars..=(skip_chars + 50).min(chunk.cleaned_text.len()) {
                        if j < text_bytes.len() && (text_bytes[j] == b' ' || text_bytes[j] == b'\n')
                        {
                            actual_skip = j + 1;
                            break;
                        }
                    }

                    if actual_skip < chunk.cleaned_text.len() {
                        // Add space if needed
                        if !result.ends_with(' ') && !result.ends_with('\n') {
                            result.push(' ');
                        }
                        result.push_str(&chunk.cleaned_text[actual_skip..]);
                    }
                }
            }
        }

        TextModalityOutput {
            success: true,
            reconstructed_text: Some(result),
            ..Default::default()
        }
    }

    // ========================================================================
    // ANALYSIS
    // ========================================================================

    /// Full text analysis with chunk processing, entity/topic/structure extraction,
    /// grammar relationships, modality detection, and chunk graph creation.
    /// Incorporates what ProcessChunk used to do — text files and text prompts
    /// are processed the same way through this single entry point.
    /// Full text analysis with chunk processing.
    /// Splits into Path 1 (standard chunking) or Path 2 (granular 1-by-1 sentence extraction).
    async fn analyze_text(
        &self,
        text: &str,
        max_chunk_tokens: u32,
        depth: AnalysisDepth,
        extract_entities: bool,
        extract_topics: bool,
        extract_structure: bool,
        available_modalities: &[String],
        processing_path: ProcessingPath,
    ) -> TextModalityOutput {
        let chunks = Self::chunk_text(text, max_chunk_tokens);
        let mut all_processed_chunks = Vec::new();
        let mut all_modality_detections = Vec::new();
        let mut first_chunk_graph: Option<ChunkGraph> = None;
        let mut section_state = SectionState::default();
        let mut paragraph_state = ParagraphState::default();

        for chunk in &chunks {
            let (processed, chunk_graph, modalities) = match processing_path {
                ProcessingPath::Path1 => {
                    self.process_chunk_path1(chunk, available_modalities).await
                }
                ProcessingPath::Path2 => {
                    self.process_chunk_path2(
                        chunk,
                        available_modalities,
                        &mut section_state,
                        &mut paragraph_state,
                    )
                    .await
                }
            };

            all_modality_detections.extend(modalities);
            if first_chunk_graph.is_none() {
                first_chunk_graph = Some(chunk_graph);
            }
            all_processed_chunks.push(processed);
        }

        // Phase 3: Grammar Extraction on Graph
        self.extract_grammar_from_graphs(&mut all_processed_chunks)
            .await;

        // Phase 4: Document Identification
        self.identify_documents(&mut all_processed_chunks).await;

        // ── Aggregate high-level analysis from chunks ──
        let cleaned_text = all_processed_chunks
            .iter()
            .map(|c| c.cleaned_text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Basic metrics (moved after cleaning to avoid skewed data)
        let word_count = cleaned_text.split_whitespace().count();
        let sentences: Vec<&str> = cleaned_text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let sentence_count = sentences.len();
        let paragraph_count = cleaned_text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .count();
        let character_count = cleaned_text.len();

        let entities = if extract_entities {
            let mut aggregated_entities = Vec::new();
            for chunk in &all_processed_chunks {
                for e in &chunk.entities {
                    aggregated_entities.push(Entity {
                        text: e.text.clone(),
                        entity_type: EntityType::from_str(&e.entity_type),
                        start_offset: e.start_offset.unwrap_or(0),
                        end_offset: e.end_offset.unwrap_or(0),
                        confidence: e.confidence,
                        metadata: HashMap::new(),
                    });
                }
            }
            aggregated_entities
        } else {
            Vec::new()
        };

        let topics = if extract_topics {
            let mut aggregated_topic_names = std::collections::HashSet::new();
            for chunk in &all_processed_chunks {
                for topic in &chunk.topics {
                    aggregated_topic_names.insert(topic.clone());
                }
            }
            aggregated_topic_names
                .into_iter()
                .enumerate()
                .map(|(i, name)| Topic {
                    name,
                    keywords: Vec::new(),
                    relevance: 1.0 - (i as f32 * 0.1).min(0.8),
                    category: None,
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut aggregated_keywords_map: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for chunk in &all_processed_chunks {
            for kw in &chunk.keywords {
                *aggregated_keywords_map.entry(kw.clone()).or_insert(0) += 1;
            }
        }
        let keywords: Vec<Keyword> = aggregated_keywords_map
            .into_iter()
            .map(|(term, frequency)| Keyword {
                term,
                frequency,
                relevance: 1.0,
                is_phrase: term.contains(' '),
            })
            .collect();

        // Structure extraction is removed entirely, zero-shot section detection handles it in Path 2
        let structure = DocumentStructure {
            sections: Vec::new(),
            has_title: false,
            has_abstract: false,
            has_toc: false,
            document_type: DocumentType::Unknown,
        };

        let avg_words_per_sentence = if sentence_count > 0 {
            word_count as f32 / sentence_count as f32
        } else {
            0.0
        };
        let syllable_count = Self::count_syllables(&cleaned_text);
        let avg_syllables_per_word = if word_count > 0 {
            syllable_count as f32 / word_count as f32
        } else {
            0.0
        };
        let readability_score =
            206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word);

        let sentiment = if depth == AnalysisDepth::Deep || depth == AnalysisDepth::Comprehensive {
            self.analyze_sentiment(&cleaned_text).await
        } else {
            None
        };

        let analysis = TextAnalysisResult {
            word_count,
            sentence_count,
            paragraph_count,
            character_count,
            entities,
            topics,
            keywords,
            structure,
            language: Some("en".to_string()),
            sentiment,
            readability_score: Some(readability_score.clamp(0.0, 100.0)),
        };

        TextModalityOutput {
            success: true,
            analysis: Some(analysis),
            processed_chunks: Some(all_processed_chunks),
            modality_detections: Some(all_modality_detections),
            chunk_graph: first_chunk_graph,
            ..Default::default()
        }
    }

    /// Path 1: Standard chunk processing (clean whole chunk, extract attributes)
    async fn process_chunk_path1(
        &self,
        chunk: &RawChunk,
        available_modalities: &[String],
    ) -> (ProcessedChunk, ChunkGraph, Vec<ChunkModalityDetection>) {
        let clean_output = self.clean_chunk(chunk.clone()).await;
        let cleaned_text = clean_output
            .cleaned_text
            .unwrap_or_else(|| chunk.text.clone());

        let keywords = self.extract_keywords_from_text(&cleaned_text).await;
        let entities = self.extract_entities_from_text(&cleaned_text).await;
        let topics = self.extract_topics_from_text(&cleaned_text).await;
        let detected_modalities = self
            .detect_modalities_stable(&cleaned_text, chunk.index, available_modalities)
            .await;

        let processed = ProcessedChunk {
            index: chunk.index,
            original_text: chunk.text.clone(),
            cleaned_text: cleaned_text.clone(),
            start_offset: chunk.start_char,
            end_offset: chunk.end_char,
            prompt_start_char: chunk.start_char as usize,
            prompt_end_char: chunk.end_char as usize,
            token_count: chunk.token_count,
            keywords,
            entities,
            topics,
            overlap_from_previous: 0,
            overlap_to_next: 0,
            sentence_nodes: Vec::new(),
            paragraph_nodes: Vec::new(),
            section_nodes: Vec::new(),
            document_nodes: Vec::new(),
            cross_sentence_relationships: Vec::new(),
            coreference_chains: Vec::new(),
            detected_modalities: detected_modalities.clone(),
            chunk_graph_id: None,
        };

        let chunk_graph = self.create_chunk_graph(&processed, 0);
        (processed, chunk_graph, detected_modalities)
    }

    /// Path 2: Granular 1-by-1 sentence extraction, 5x stable validation,
    /// section/paragraph tracking, and sentence node creation.
    async fn process_chunk_path2(
        &self,
        chunk: &RawChunk,
        available_modalities: &[String],
        section_state: &mut SectionState,
        paragraph_state: &mut ParagraphState,
    ) -> (ProcessedChunk, ChunkGraph, Vec<ChunkModalityDetection>) {
        let mut node_id_counter = 1u64;
        let mut sentence_nodes = Vec::new();
        let mut paragraph_nodes = Vec::new();
        let mut section_nodes = Vec::new();
        let mut document_nodes = Vec::new();
        let mut cross_sentence_relationships = Vec::new();
        let mut coreference_chains = Vec::new();

        // 1. Zero-shot section detection
        let (section_node_opt, new_section_state) = self
            .detect_section_boundary(chunk, &section_state.clone())
            .await;
        if let Some(sec_node) = &section_node_opt {
            section_nodes.push(sec_node.clone());
        }
        *section_state = new_section_state;

        // 2. Zero-shot paragraph detection
        let (paragraph_node_opt, new_paragraph_state) = self
            .detect_paragraph_boundary(chunk, &paragraph_state.clone())
            .await;
        if let Some(par_node) = &paragraph_node_opt {
            paragraph_nodes.push(par_node.clone());
        }
        *paragraph_state = new_paragraph_state;

        // 3. Sentence extraction 1-by-1 with 5x stable validation
        let mut known_sentences_json: Vec<serde_json::Value> = Vec::new();
        let mut consecutive_false = 0;

        loop {
            let prompt = format!(
                r#"Identify the next sentence in the text chunk.
                ALREADY KNOWN SENTENCES (do NOT repeat these):
                {}

                CHUNK TEXT:
                {}

                Return ONLY valid JSON:
                {{
                  "found": true,
                  "sentence": {{
                    "order": 1,
                    "original_sentence": "the exact text span from the chunk",
                    "text_span_start": 0,
                    "text_span_end": 100,
                    "corrected_sentence": "grammatically corrected version"
                  }}
                }}
                If no new sentences, return: {{"found": false}}"#,
                serde_json::to_string(&known_sentences_json).unwrap_or_default(),
                &chunk.text[..chunk.text.len().min(4000)]
            );

            let mut stable_extractions = 0;
            let mut last_extracted_sentence: Option<serde_json::Value> = None;

            // 5x validation: ask 5 times to ensure we get the same sentence
            for _ in 0..5 {
                let input = serde_json::json!({
                    "prompt": &prompt,
                    "max_tokens": 500,
                    "temperature": 0.1
                });

                let response = self.executor.execute(9, input).await.unwrap_or_default();
                let raw = response
                    .get("response")
                    .and_then(|r| r.as_str())
                    .unwrap_or("{}");
                let json_str = Self::extract_json_from_response(raw, '{', '}');
                let parsed =
                    serde_json::from_str::<serde_json::Value>(json_str).unwrap_or_default();

                if !parsed
                    .get("found")
                    .and_then(|f| f.as_bool())
                    .unwrap_or(false)
                {
                    break; // If any says false, break inner loop
                }

                if let Some(sent_val) = parsed.get("sentence") {
                    if let Some(last) = &last_extracted_sentence {
                        if last.get("original_sentence") == sent_val.get("original_sentence") {
                            stable_extractions += 1;
                        }
                    } else {
                        last_extracted_sentence = Some(sent_val.clone());
                        stable_extractions += 1;
                    }
                }
            }

            if stable_extractions == 5 {
                if let Some(sent_val) = last_extracted_sentence {
                    let original = sent_val
                        .get("original_sentence")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string();
                    let start = sent_val
                        .get("text_span_start")
                        .and_then(|s| s.as_usize())
                        .unwrap_or(0);
                    let end = sent_val
                        .get("text_span_end")
                        .and_then(|s| s.as_usize())
                        .unwrap_or(0);
                    let corrected = sent_val
                        .get("corrected_sentence")
                        .and_then(|s| s.as_str())
                        .unwrap_or(&original)
                        .to_string();

                    let sent_node = SentenceNode {
                        node_id: node_id_counter,
                        node_type: GrammarNodeType::Sentence,
                        content: corrected.clone(),
                        original_content: original.clone(),
                        position: TextPosition {
                            start_offset: start,
                            end_offset: end,
                            line: None,
                            column: None,
                        },
                        chunk_id: chunk.index,
                        chunk_offset: start,
                        paragraph_id: paragraph_nodes.last().map(|p| p.node_id),
                        section_id: section_nodes.last().map(|s| s.node_id),
                        properties: GrammarProperties::default(),
                        grammar_nodes: Vec::new(),
                    };

                    known_sentences_json.push(serde_json::json!({
                        "order": sent_val.get("order").and_then(|o| o.as_u64()).unwrap_or(node_id_counter as u64),
                        "original": original,
                        "corrected": corrected
                    }));

                    sentence_nodes.push(sent_node);
                    node_id_counter += 1;
                    consecutive_false = 0;
                }
            } else {
                consecutive_false += 1;
                if consecutive_false >= 5 {
                    break; // 5 consecutive times no new stable sentence found
                }
            }
        }

        if let Some(p_node) = paragraph_nodes.last_mut() {
            p_node.sentence_count = sentence_nodes.len() as u32;
        }

        // 4. Modality detection (tied to sentences/paragraphs)
        let detected_modalities = self
            .detect_modalities_stable(&chunk.text, chunk.index, available_modalities)
            .await;
        let mut tied_modalities = Vec::new();
        for mut detection in detected_modalities {
            let parent_id = sentence_nodes
                .iter()
                .find(|s| {
                    detection.span_start >= s.position.start_offset
                        && detection.span_start <= s.position.end_offset
                })
                .map(|s| s.node_id)
                .or_else(|| paragraph_nodes.last().map(|p| p.node_id));
            detection.parent_node_id = parent_id;
            tied_modalities.push(detection);
        }

        let cleaned_text = sentence_nodes
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let processed = ProcessedChunk {
            index: chunk.index,
            original_text: chunk.text.clone(),
            cleaned_text: cleaned_text.clone(),
            start_offset: chunk.start_char,
            end_offset: chunk.end_char,
            prompt_start_char: chunk.start_char as usize,
            prompt_end_char: chunk.end_char as usize,
            token_count: chunk.token_count,
            keywords: Vec::new(),
            entities: Vec::new(),
            topics: Vec::new(),
            overlap_from_previous: 0,
            overlap_to_next: 0,
            sentence_nodes: sentence_nodes.clone(),
            paragraph_nodes: paragraph_nodes.clone(),
            section_nodes: section_nodes.clone(),
            document_nodes: document_nodes.clone(),
            cross_sentence_relationships: cross_sentence_relationships.clone(),
            coreference_chains: coreference_chains.clone(),
            detected_modalities: tied_modalities.clone(),
            chunk_graph_id: None,
        };

        let chunk_graph = self.create_chunk_graph(&processed, 0);
        (processed, chunk_graph, tied_modalities)
    }

    /// Phase 3: Grammar Extraction on Graph (Per-Sentence + Cross-Sentence)
    async fn extract_grammar_from_graphs(&self, chunks: &mut [ProcessedChunk]) {
        let mut all_grammar_nodes_for_chunk = Vec::new();

        // Per-Sentence Grammar Extraction
        for chunk in chunks.iter_mut() {
            for sent_node in chunk.sentence_nodes.iter_mut() {
                let prompt = format!(
                    r#"You are extracting grammatical structure from a single sentence.

                    SENTENCE: "{}"

                    TASK: Identify the grammatical components and their relationships.

                    Return ONLY valid JSON:
                    {{
                      "subject": {{
                        "text": "the subject noun phrase",
                        "position_start": 0,
                        "position_end": 10,
                        "entity_type_hint": "Person|Organization|Location|Document|Time|Concept|Unknown"
                      }},
                      "verb": {{
                        "text": "the main verb",
                        "position_start": 11,
                        "position_end": 20,
                        "verb_type": "action|linking|helping",
                        "tense": "past|present|future|unknown",
                        "negated": false
                      }},
                      "object": {{
                        "text": "the object noun phrase or null",
                        "position_start": 21,
                        "position_end": 35,
                        "entity_type_hint": "..."
                      }},
                      "modifiers": [
                        {{
                          "text": "modifier text",
                          "position_start": 36,
                          "position_end": 45,
                          "modifier_type": "temporal|adjectival|adverbial|prepositional",
                          "modifies": "subject|verb|object"
                        }}
                      ],
                      "sentence_type": "declarative|interrogative|imperative|fragment",
                      "edge_type": "Performs|Affects|Implies|Contradicts|Elaborates|Summarizes|Supports|TemporalPrecedes|TemporalFollows|CausedBy|Enables|Prevents|PartOf|HasPart|FunctionalRole|InstanceOf|HasInstance|SimilarTo|DerivedFrom|VersionOf.."
                    }}"#,
                    sent_node.content
                );

                let input = serde_json::json!({
                    "prompt": prompt,
                    "max_tokens": 800,
                    "temperature": 0.05
                });

                if let Ok(response) = self.executor.execute(9, input).await {
                    let raw = response
                        .get("response")
                        .and_then(|r| r.as_str())
                        .unwrap_or("{}");
                    let json_str = Self::extract_json_from_response(raw, '{', '}');
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                        let mut grammar_nodes = Vec::new();

                        // Subject
                        if let Some(subj) = parsed.get("subject") {
                            grammar_nodes.push(GrammarNode {
                                node_id: 0, // ID assigned later if needed, or omitted if not required by graph engine
                                node_type: GrammarNodeType::Subject,
                                text: subj
                                    .get("text")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                position: TextPosition {
                                    start_offset: subj
                                        .get("position_start")
                                        .and_then(|p| p.as_usize())
                                        .unwrap_or(0),
                                    end_offset: subj
                                        .get("position_end")
                                        .and_then(|p| p.as_usize())
                                        .unwrap_or(0),
                                    line: None,
                                    column: None,
                                },
                                children: Vec::new(),
                                properties: GrammarProperties::default(),
                            });
                        }

                        // Verb
                        if let Some(verb) = parsed.get("verb") {
                            let v_type = match verb
                                .get("verb_type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("action")
                            {
                                "linking" => GrammarNodeType::LinkingVerb,
                                "helping" => GrammarNodeType::HelpingVerb,
                                _ => GrammarNodeType::MainVerb,
                            };
                            grammar_nodes.push(GrammarNode {
                                node_id: 0,
                                node_type: v_type,
                                text: verb
                                    .get("text")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                position: TextPosition {
                                    start_offset: verb
                                        .get("position_start")
                                        .and_then(|p| p.as_usize())
                                        .unwrap_or(0),
                                    end_offset: verb
                                        .get("position_end")
                                        .and_then(|p| p.as_usize())
                                        .unwrap_or(0),
                                    line: None,
                                    column: None,
                                },
                                children: Vec::new(),
                                properties: GrammarProperties {
                                    tense: verb
                                        .get("tense")
                                        .and_then(|t| t.as_str())
                                        .map(String::from),
                                    polarity: Some(
                                        verb.get("negated")
                                            .and_then(|n| n.as_bool())
                                            .unwrap_or(false)
                                            .to_string(),
                                    ),
                                    ..Default::default()
                                },
                            });
                        }

                        // Object
                        if let Some(obj) = parsed.get("object") {
                            if !obj.is_null() {
                                grammar_nodes.push(GrammarNode {
                                    node_id: 0,
                                    node_type: GrammarNodeType::DirectObject,
                                    text: obj
                                        .get("text")
                                        .and_then(|t| t.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    position: TextPosition {
                                        start_offset: obj
                                            .get("position_start")
                                            .and_then(|p| p.as_usize())
                                            .unwrap_or(0),
                                        end_offset: obj
                                            .get("position_end")
                                            .and_then(|p| p.as_usize())
                                            .unwrap_or(0),
                                        line: None,
                                        column: None,
                                    },
                                    children: Vec::new(),
                                    properties: GrammarProperties::default(),
                                });
                            }
                        }

                        // Modifiers
                        if let Some(mods) = parsed.get("modifiers").and_then(|m| m.as_array()) {
                            for m_val in mods {
                                let m_type = match m_val
                                    .get("modifier_type")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("adverbial")
                                {
                                    "temporal" => GrammarNodeType::AdverbOfTime,
                                    "adjectival" => GrammarNodeType::Adjective,
                                    "prepositional" => GrammarNodeType::PrepositionalPhrase,
                                    _ => GrammarNodeType::Adverb,
                                };
                                grammar_nodes.push(GrammarNode {
                                    node_id: 0,
                                    node_type: m_type,
                                    text: m_val
                                        .get("text")
                                        .and_then(|t| t.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    position: TextPosition {
                                        start_offset: m_val
                                            .get("position_start")
                                            .and_then(|p| p.as_usize())
                                            .unwrap_or(0),
                                        end_offset: m_val
                                            .get("position_end")
                                            .and_then(|p| p.as_usize())
                                            .unwrap_or(0),
                                        line: None,
                                        column: None,
                                    },
                                    children: Vec::new(),
                                    properties: GrammarProperties::default(),
                                });
                            }
                        }

                        sent_node.grammar_nodes = grammar_nodes;
                    }
                }
            }
        }

        // Cross-Sentence Relationship Extraction (Pairwise Comparison)
        // Gather all sentences across all chunks for pairwise analysis
        let mut all_sentences: Vec<(u64, String)> = Vec::new();
        for chunk in chunks.iter() {
            for sent in &chunk.sentence_nodes {
                all_sentences.push((sent.node_id, sent.content.clone()));
            }
        }

        if all_sentences.len() > 1 {
            let sentence_list = all_sentences
                .iter()
                .map(|(id, s)| format!("{}: {}", id, s))
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!(
                r#"You are analyzing relationships between sentences in a text corpus.

                SENTENCES:
                {}

                TASK: Identify all meaningful relationships between these sentences.

                Relationship types:
                - Elaborates:        S2 provides more detail about S1
                - Causes:            S1's action or state leads to S2
                - Enables:           S1 makes S2 possible
                - Prevents:          S1 prevents S2
                - Contradicts:       S2 contradicts or negates S1
                - Exemplifies:       S2 is a concrete example of S1
                - Summarizes:        S2 is a summary of S1 (or vice versa)
                - TemporalPrecedes:  S1 happens before S2 in described time
                - Coreference:       S2 contains a pronoun or reference to an entity in S1
                - PartOf:            S2 is a component of the whole described in S1
                - SimilarTo:         S1 and S2 describe closely related concepts

                Return ONLY valid JSON:
                {{
                  "relationships": [
                    {{
                      "from_sentence_index": 0,
                      "to_sentence_index": 1,
                      "relationship_type": "Elaborates|Causes|Enables|...",
                      "evidence": "brief quote showing the relationship"
                    }}
                  ],
                  "coreference_chains": [
                    {{
                      "canonical_entity": "John Smith",
                      "mentions": [
                        {{"sentence_index": 0, "text": "John Smith", "grammar_role": "subject"}},
                        {{"sentence_index": 2, "text": "He", "grammar_role": "subject"}}
                      ]
                    }}
                  ]
                }}"#,
                sentence_list
            );

            let input = serde_json::json!({
                "prompt": prompt,
                "max_tokens": 2000,
                "temperature": 0.1
            });

            if let Ok(response) = self.executor.execute(9, input).await {
                let raw = response
                    .get("response")
                    .and_then(|r| r.as_str())
                    .unwrap_or("{}");
                let json_str = Self::extract_json_from_response(raw, '{', '}');
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                    // Distribute relationships back to chunks
                    let mut chunk_idx = 0;
                    let mut sent_idx = 0;

                    // Re-iterate to map back correctly
                    let mut rel_map: std::collections::HashMap
                        u64,
                        Vec<CrossSentenceRelationship>,
                    > = std::collections::HashMap::new();
                    let mut chain_map: std::collections::HashMap<u64, Vec<CoreferenceMention>> =
                        std::collections::HashMap::new();
                    let mut canonical_forms: std::collections::HashMap<u64, String> =
                        std::collections::HashMap::new();
                    let mut chain_id_counter = 1u64;

                    if let Some(rels) = parsed.get("relationships").and_then(|r| r.as_array()) {
                        for rel in rels {
                            let from_id = rel
                                .get("from_sentence_index")
                                .and_then(|i| i.as_u64())
                                .unwrap_or(0);
                            let to_id = rel
                                .get("to_sentence_index")
                                .and_then(|i| i.as_u64())
                                .unwrap_or(0);
                            let rel_type = rel
                                .get("relationship_type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("")
                                .to_string();
                            let evidence = rel
                                .get("evidence")
                                .and_then(|e| e.as_str())
                                .unwrap_or("")
                                .to_string();

                            rel_map.entry(from_id).or_insert_with(Vec::new).push(
                                CrossSentenceRelationship {
                                    from_sentence_id: from_id,
                                    to_sentence_id: to_id,
                                    relationship_type: rel_type,
                                    evidence,
                                },
                            );
                        }
                    }

                    if let Some(chains) =
                        parsed.get("coreference_chains").and_then(|c| c.as_array())
                    {
                        for chain in chains {
                            let canonical = chain
                                .get("canonical_entity")
                                .and_then(|c| c.as_str())
                                .unwrap_or("")
                                .to_string();
                            let current_chain_id = chain_id_counter;
                            chain_id_counter += 1;

                            if let Some(mentions) = chain.get("mentions").and_then(|m| m.as_array())
                            {
                                for mention in mentions {
                                    let sent_id = mention
                                        .get("sentence_index")
                                        .and_then(|i| i.as_u64())
                                        .unwrap_or(0);
                                    let text = mention
                                        .get("text")
                                        .and_then(|t| t.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let role = mention
                                        .get("grammar_role")
                                        .and_then(|g| g.as_str())
                                        .unwrap_or("")
                                        .to_string();

                                    chain_map.entry(sent_id).or_insert_with(Vec::new).push(
                                        CoreferenceMention {
                                            sentence_id: sent_id,
                                            text,
                                            grammar_role: role,
                                        },
                                    );
                                    canonical_forms.insert(current_chain_id, canonical.clone());
                                }
                            }
                        }
                    }

                    // Assign back to chunks
                    for chunk in chunks.iter_mut() {
                        let mut chunk_rels = Vec::new();
                        let mut chunk_chains: std::collections::HashMap
                            u64,
                            Vec<CoreferenceMention>,
                        > = std::collections::HashMap::new();

                        for sent in &chunk.sentence_nodes {
                            if let Some(rels) = rel_map.get(&sent.node_id) {
                                chunk_rels.extend(rels.clone());
                            }
                            if let Some(mentions) = chain_map.get(&sent.node_id) {
                                // Group by chain_id (simplified for draft: all mentions in chunk map to one chain if canonical matches)
                                for m in mentions {
                                    chunk_chains
                                        .entry(1)
                                        .or_insert_with(Vec::new)
                                        .push(m.clone());
                                }
                            }
                        }

                        chunk.cross_sentence_relationships = chunk_rels;
                        chunk.coreference_chains = chunk_chains
                            .into_iter()
                            .map(|(id, mentions)| CoreferenceChain {
                                chain_id: id,
                                canonical_form: canonical_forms
                                    .get(&id)
                                    .cloned()
                                    .unwrap_or_default(),
                                mentions,
                            })
                            .collect();
                    }
                }
            }
        }
    }

    /// Phase 4: Document Identification
    async fn identify_documents(&self, chunks: &mut [ProcessedChunk]) {
        for chunk in chunks.iter_mut() {
            if !chunk.sentence_nodes.is_empty() {
                let text_span = chunk
                    .sentence_nodes
                    .iter()
                    .map(|s| s.content.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let prompt = format!(
                    r#"You are identifying whether a block of text constitutes a standalone document within a larger corpus.

                    A "document" is a coherent, self-contained unit characterized by:
                    - Unified topic or theme throughout
                    - Consistent formatting pattern (whatever format)
                    - Clear structural organization (sections, paragraphs, or both)
                    - Neutral or third-person narrative voice (for reference/informational documents)
                    - OR: a sustained narrative voice (for stories, guides, essays)

                    NOT a document:
                    - First-person conversational requests ("I want", "Please help me")
                    - Imperative instructions addressed to the AI ("Fix this", "Create a")
                    - Questions without context ("How do I", "What is")
                    - Short mixed-intent fragments

                    IMPORTANT: Documents sometimes contain user intent mixed in. The presence of some conversational sentences does not disqualify a document — look for the dominant structure and voice.

                    TEXT TO ANALYZE:
                    {}

                    Return ONLY valid JSON:
                    {{
                      "is_document": true|false,
                      "document_type": "article|guide|story|report|specification|reference|code_doc|data|unknown|null",
                      "title": "extracted title if detectable, null otherwise",
                      "confidence": 0.0-1.0,
                      "dominant_structure": "description of what makes this a document or not",
                      "mixed_intent_detected": true|false,
                      "mixed_intent_description": "what conversational/intent elements appear, or null"
                    }}"#,
                    &text_span[..text_span.len().min(4000)]
                );

                let input = serde_json::json!({
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.1
                });

                if let Ok(response) = self.executor.execute(9, input).await {
                    let raw = response
                        .get("response")
                        .and_then(|r| r.as_str())
                        .unwrap_or("{}");
                    let json_str = Self::extract_json_from_response(raw, '{', '}');
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                        if parsed
                            .get("is_document")
                            .and_then(|d| d.as_bool())
                            .unwrap_or(false)
                        {
                            let doc_node = DocumentNode {
                                node_id: 1,                           // Simplified ID for draft
                                node_type: GrammarNodeType::Sentence, // Conceptually Document, but not in enum. Using Sentence as placeholder or adding Document to enum is better. Let's use Unknown for now if not in enum.
                                content: text_span,
                                document_type: parsed
                                    .get("document_type")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("unknown")
                                    .to_string(),
                                title: parsed
                                    .get("title")
                                    .and_then(|t| t.as_str())
                                    .map(String::from),
                                position: TextPosition {
                                    start_offset: chunk.prompt_start_char,
                                    end_offset: chunk.prompt_end_char,
                                    line: None,
                                    column: None,
                                },
                            };
                            chunk.document_nodes.push(doc_node);
                        }
                    }
                }
            }
        }
    }

    /// Zero-shot section boundary detection
    async fn detect_section_boundary(
        &self,
        chunk: &RawChunk,
        prev_state: &SectionState,
    ) -> (Option<SectionNode>, SectionState) {
        let prompt = format!(
            r#"You are analyzing text to detect document structure and section boundaries.
            The formatting style is UNKNOWN — do not assume any specific format (not just markdown). Detect whatever structural patterns actually appear in the text.

            CHUNK CONTEXT:
            - This is chunk {} of unknown total
            - Previous section state: {:?}
            - Formatting pattern being tracked: {:?}

            CHUNK TEXT:
            {}

            TASK: Identify sections and document boundaries.

            A "section" is any coherent block that begins with a recognizable heading or structural marker — regardless of format. This includes but is not limited to:
            - Markdown headers (# ## ###)
            - Numbered sections (1. 2. 2.1)
            - ALL-CAPS headings (INTRODUCTION, METHODS)
            - Underlined or decorated headings (===, ---)
            - Bold lead lines followed by body text
            - Any other consistent structural marker you detect

            A "document break" occurs when:
            - The formatting pattern changes significantly
            - The writing style or voice shifts substantially
            - Structural markers stop appearing after a consistent run

            Return ONLY valid JSON:
            {{
              "section_events": [
                {{
                  "type": "section_start|section_continue|section_end|document_break",
                  "position": <character_offset_in_chunk>,
                  "section_title": "title if section_start, null otherwise",
                  "section_level": <1-6 or null>,
                  "formatting_pattern": "description of detected pattern or null"
                }}
              ],
              "current_section_context": {{
                "section_id": "unique identifier for tracking across chunks",
                "level": <1-6 or null>,
                "title": "current section title or null",
                "formatting_pattern": "detected pattern description"
              }},
              "document_boundary_detected": false,
              "document_boundary_reason": "reason or null"
            }}"#,
            chunk.index,
            prev_state,
            prev_state.formatting_pattern,
            &chunk.text[..chunk.text.len().min(4000)]
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.1
        });

        let mut new_state = prev_state.clone();
        let mut section_node = None;

        if let Ok(response) = self.executor.execute(9, input).await {
            let raw = response
                .get("response")
                .and_then(|r| r.as_str())
                .unwrap_or("{}");
            let json_str = Self::extract_json_from_response(raw, '{', '}');
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(events) = parsed.get("section_events").and_then(|e| e.as_array()) {
                    for event in events {
                        let ev_type = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if ev_type == "section_start" {
                            let title = event
                                .get("section_title")
                                .and_then(|t| t.as_str())
                                .map(String::from);
                            let level = event
                                .get("section_level")
                                .and_then(|l| l.as_u64())
                                .map(|l| l as u8);
                            let pattern = event
                                .get("formatting_pattern")
                                .and_then(|f| f.as_str())
                                .map(String::from);

                            new_state.current_section_id = Some(Self::generate_id());
                            new_state.current_title = title.clone();
                            new_state.current_level = level;
                            new_state.formatting_pattern = pattern.clone();

                            section_node = Some(SectionNode {
                                node_id: new_state.current_section_id.unwrap_or(1),
                                node_type: GrammarNodeType::Sentence, // Conceptually Section
                                content: title.unwrap_or_default(),
                                level: level.unwrap_or(1),
                                formatting_pattern: pattern,
                                parent_section: prev_state.current_section_id,
                                position: TextPosition {
                                    start_offset: event
                                        .get("position")
                                        .and_then(|p| p.as_usize())
                                        .unwrap_or(0),
                                    end_offset: chunk.text.len(),
                                    line: None,
                                    column: None,
                                },
                            });
                            break; // Only one section start per chunk for simplicity in draft
                        }
                    }
                }

                // Update state from context if no explicit start event
                if section_node.is_none() {
                    if let Some(ctx) = parsed.get("current_section_context") {
                        if new_state.current_section_id.is_none() {
                            new_state.current_section_id = Some(Self::generate_id());
                        }
                        new_state.current_level =
                            ctx.get("level").and_then(|l| l.as_u64()).map(|l| l as u8);
                        new_state.current_title =
                            ctx.get("title").and_then(|t| t.as_str()).map(String::from);
                        new_state.formatting_pattern = ctx
                            .get("formatting_pattern")
                            .and_then(|f| f.as_str())
                            .map(String::from);
                    }
                }
            }
        }

        (section_node, new_state)
    }

    /// Zero-shot paragraph boundary detection
    async fn detect_paragraph_boundary(
        &self,
        chunk: &RawChunk,
        prev_state: &ParagraphState,
    ) -> (Option<ParagraphNode>, ParagraphState) {
        let prompt = format!(
            r#"Identify paragraph boundaries in the text chunk.
            A paragraph is a distinct block of text separated by blank lines or clear thematic shifts.
            If the previous chunk ended with an open paragraph, determine if it continues or ends here.

            PREVIOUS STATE:
            Open paragraph: {}

            CHUNK TEXT:
            {}

            Return ONLY valid JSON:
            {{
              "paragraphs": [
                {{
                  "type": "start|continue|end",
                  "position": <character_offset_in_chunk>,
                  "is_open": true|false
                }}
              ]
            }}"#,
            prev_state.open_paragraph,
            &chunk.text[..chunk.text.len().min(4000)]
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.1
        });

        let mut new_state = prev_state.clone();
        let mut paragraph_node = None;

        if let Ok(response) = self.executor.execute(9, input).await {
            let raw = response
                .get("response")
                .and_then(|r| r.as_str())
                .unwrap_or("{}");
            let json_str = Self::extract_json_from_response(raw, '{', '}');
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(paras) = parsed.get("paragraphs").and_then(|p| p.as_array()) {
                    for para in paras {
                        let p_type = para.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if p_type == "start" || (p_type == "continue" && !new_state.open_paragraph)
                        {
                            new_state.current_paragraph_id = Some(Self::generate_id());
                            new_state.open_paragraph = para
                                .get("is_open")
                                .and_then(|o| o.as_bool())
                                .unwrap_or(true);
                            new_state.start_offset_in_chunk =
                                para.get("position").and_then(|p| p.as_usize());

                            paragraph_node = Some(ParagraphNode {
                                node_id: new_state.current_paragraph_id.unwrap_or(1),
                                node_type: GrammarNodeType::Sentence, // Conceptually Paragraph
                                sentence_count: 0,
                                parent_section: None, // Tied later or via section state
                                position: TextPosition {
                                    start_offset: new_state.start_offset_in_chunk.unwrap_or(0),
                                    end_offset: chunk.text.len(),
                                    line: None,
                                    column: None,
                                },
                            });
                            break;
                        } else if p_type == "end" {
                            new_state.open_paragraph = false;
                        }
                    }
                }
            }
        }

        (paragraph_node, new_state)
    }

    /// Analyze sentiment via LLM
    async fn analyze_sentiment(&self, text: &str) -> Option<Sentiment> {
        let prompt = format!(
            r#"Analyze the sentiment of this text.
Return JSON: {{"overall": -1.0 to 1.0, "positive": 0-1, "negative": 0-1, "neutral": 0-1}}

TEXT:
{}

RESPOND ONLY WITH JSON."#,
            &text[..text.len().min(2000)]
        );

        let input = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.2,
            "system_context": "Analyze sentiment. Respond with JSON only."
        });

        match self.executor.execute(9, input).await {
            Ok(result) => result
                .get("response")
                .and_then(|r| r.as_str())
                .and_then(|s| {
                    let json_str = Self::extract_json_from_response(s, '{', '}');
                    serde_json::from_str::<serde_json::Value>(&json_str).ok()
                })
                .map(|v| Sentiment {
                    overall: v.get("overall").and_then(|o| o.as_f64()).unwrap_or(0.0) as f32,
                    positive: v.get("positive").and_then(|p| p.as_f64()).unwrap_or(0.33) as f32,
                    negative: v.get("negative").and_then(|n| n.as_f64()).unwrap_or(0.33) as f32,
                    neutral: v.get("neutral").and_then(|n| n.as_f64()).unwrap_or(0.34) as f32,
                }),
            Err(_) => None,
        }
    }

    /// Count syllables (heuristic)
    fn count_syllables(text: &str) -> usize {
        let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
        let mut count = 0;
        let mut prev_was_vowel = false;

        for c in text.to_lowercase().chars() {
            let is_vowel = vowels.contains(&c);
            if is_vowel && !prev_was_vowel {
                count += 1;
            }
            prev_was_vowel = is_vowel;
        }

        count.max(1)
    }

    // ========================================================================
    // GRAPH OPERATIONS
    // ========================================================================

    /// Create graph from analysis results
    async fn create_graph(
        &self,
        analysis: TextAnalysisResult,
        project_id: u64,
        _link_to_existing: bool,
    ) -> TextModalityOutput {
        let graph_id = Self::generate_id();
        let now = chrono::Utc::now().to_rfc3339();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_id = 1u64;
        let mut edge_id = 1u64;

        // Create document root node
        let doc_node_id = node_id;
        nodes.push(TextGraphNode {
            node_id: doc_node_id,
            node_type: TextNodeType::Document,
            content: format!(
                "{} words, {} sentences, {} paragraphs",
                analysis.word_count, analysis.sentence_count, analysis.paragraph_count
            ),
            position: None,
            properties: {
                let mut props = HashMap::new();
                props.insert(
                    "word_count".to_string(),
                    serde_json::json!(analysis.word_count),
                );
                props.insert(
                    "sentence_count".to_string(),
                    serde_json::json!(analysis.sentence_count),
                );
                props.insert(
                    "paragraph_count".to_string(),
                    serde_json::json!(analysis.paragraph_count),
                );
                if let Some(score) = analysis.readability_score {
                    props.insert("readability_score".to_string(), serde_json::json!(score));
                }
                props
            },
            semantic_annotations: Vec::new(),
        });
        node_id += 1;

        // Create section nodes
        for section in &analysis.structure.sections {
            let section_node_id = node_id;
            nodes.push(TextGraphNode {
                node_id: section_node_id,
                node_type: TextNodeType::Section,
                content: section.title.clone().unwrap_or_default(),
                position: Some(TextPosition {
                    start_offset: section.start_offset,
                    end_offset: section.end_offset,
                    line: None,
                    column: None,
                }),
                properties: {
                    let mut props = HashMap::new();
                    props.insert("level".to_string(), serde_json::json!(section.level));
                    props
                },
                semantic_annotations: Vec::new(),
            });

            edges.push(TextGraphEdge {
                edge_id,
                from_node: doc_node_id,
                to_node: section_node_id,
                edge_type: TextEdgeType::Contains,
                weight: 1.0,
                properties: HashMap::new(),
            });
            edge_id += 1;

            node_id += 1;
        }

        // Create entity nodes
        for entity in &analysis.entities {
            let entity_node_id = node_id;
            nodes.push(TextGraphNode {
                node_id: entity_node_id,
                node_type: TextNodeType::Entity,
                content: entity.text.clone(),
                position: Some(TextPosition {
                    start_offset: entity.start_offset,
                    end_offset: entity.end_offset,
                    line: None,
                    column: None,
                }),
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "entity_type".to_string(),
                        serde_json::to_value(&entity.entity_type).unwrap(),
                    );
                    props.insert(
                        "confidence".to_string(),
                        serde_json::json!(entity.confidence),
                    );
                    props
                },
                semantic_annotations: Vec::new(),
            });

            edges.push(TextGraphEdge {
                edge_id,
                from_node: doc_node_id,
                to_node: entity_node_id,
                edge_type: TextEdgeType::Contains,
                weight: entity.confidence,
                properties: HashMap::new(),
            });
            edge_id += 1;

            node_id += 1;
        }

        // Create topic nodes
        for topic in &analysis.topics {
            let topic_node_id = node_id;
            nodes.push(TextGraphNode {
                node_id: topic_node_id,
                node_type: TextNodeType::Topic,
                content: topic.name.clone(),
                position: None,
                properties: {
                    let mut props = HashMap::new();
                    props.insert("relevance".to_string(), serde_json::json!(topic.relevance));
                    props.insert(
                        "keywords".to_string(),
                        serde_json::to_value(&topic.keywords).unwrap(),
                    );
                    props
                },
                semantic_annotations: Vec::new(),
            });

            edges.push(TextGraphEdge {
                edge_id,
                from_node: doc_node_id,
                to_node: topic_node_id,
                edge_type: TextEdgeType::Contains,
                weight: topic.relevance,
                properties: HashMap::new(),
            });
            edge_id += 1;

            node_id += 1;
        }

        // Create keyword nodes
        for keyword in &analysis.keywords {
            let keyword_node_id = node_id;
            nodes.push(TextGraphNode {
                node_id: keyword_node_id,
                node_type: TextNodeType::Keyword,
                content: keyword.term.clone(),
                position: None,
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "frequency".to_string(),
                        serde_json::json!(keyword.frequency),
                    );
                    props.insert(
                        "relevance".to_string(),
                        serde_json::json!(keyword.relevance),
                    );
                    props.insert(
                        "is_phrase".to_string(),
                        serde_json::json!(keyword.is_phrase),
                    );
                    props
                },
                semantic_annotations: Vec::new(),
            });

            edges.push(TextGraphEdge {
                edge_id,
                from_node: doc_node_id,
                to_node: keyword_node_id,
                edge_type: TextEdgeType::Contains,
                weight: keyword.relevance,
                properties: HashMap::new(),
            });
            edge_id += 1;

            node_id += 1;
        }

        let graph = TextGraph {
            graph_id,
            modality: PIPELINE_MODALITY.to_string(),
            version: PIPELINE_VERSION.to_string(),
            nodes,
            edges,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("project_id".to_string(), serde_json::json!(project_id));
                meta.insert("created_at".to_string(), serde_json::json!(&now));
                meta.insert(
                    "document_type".to_string(),
                    serde_json::to_value(&analysis.structure.document_type).unwrap(),
                );
                meta
            },
            created_at: now.clone(),
            updated_at: now,
        };

        // Cache the graph
        {
            let mut cache = self.graph_cache.write().await;
            cache.insert(graph_id, graph.clone());
        }

        TextModalityOutput {
            success: true,
            graph_id: Some(graph_id),
            graph: Some(graph),
            ..Default::default()
        }
    }

    /// Update existing graph
    async fn update_graph(&self, graph_id: u64, delta: TextDelta) -> TextModalityOutput {
        let mut cache = self.graph_cache.write().await;

        if let Some(graph) = cache.get_mut(&graph_id) {
            let now = chrono::Utc::now().to_rfc3339();
            graph.updated_at = now;

            match delta.operation {
                DeltaOperation::Insert => {
                    // Add new nodes/edges based on delta content
                    // Implementation depends on what's being inserted
                }
                DeltaOperation::Delete => {
                    // Remove nodes in affected_nodes
                    graph
                        .nodes
                        .retain(|n| !delta.affected_nodes.contains(&n.node_id));
                    graph.edges.retain(|e| {
                        !delta.affected_nodes.contains(&e.from_node)
                            && !delta.affected_nodes.contains(&e.to_node)
                    });
                }
                DeltaOperation::Replace => {
                    // Replace content in affected nodes
                    if let Some(content) = &delta.content {
                        for node in &mut graph.nodes {
                            if delta.affected_nodes.contains(&node.node_id) {
                                node.content = content.clone();
                            }
                        }
                    }
                }
                DeltaOperation::Reorder => {
                    // Reorder nodes (update edge weights or positions)
                }
            }

            TextModalityOutput {
                success: true,
                graph_id: Some(graph_id),
                graph: Some(graph.clone()),
                ..Default::default()
            }
        } else {
            TextModalityOutput {
                success: false,
                error: Some(format!("Graph {} not found", graph_id)),
                ..Default::default()
            }
        }
    }

    /// Query graph
    async fn query_graph(&self, graph_id: u64, query: TextGraphQuery) -> TextModalityOutput {
        let cache = self.graph_cache.read().await;

        if let Some(graph) = cache.get(&graph_id) {
            let result_nodes: Vec<TextGraphNode> = match query.query_type {
                TextQueryType::FindEntities => graph
                    .nodes
                    .iter()
                    .filter(|n| n.node_type == TextNodeType::Entity)
                    .cloned()
                    .collect(),

                TextQueryType::FindTopics => graph
                    .nodes
                    .iter()
                    .filter(|n| n.node_type == TextNodeType::Topic)
                    .cloned()
                    .collect(),

                TextQueryType::FindKeywords => graph
                    .nodes
                    .iter()
                    .filter(|n| n.node_type == TextNodeType::Keyword)
                    .cloned()
                    .collect(),

                TextQueryType::FindReferences => graph
                    .nodes
                    .iter()
                    .filter(|n| n.node_type == TextNodeType::Reference)
                    .cloned()
                    .collect(),

                TextQueryType::GetStructure => graph
                    .nodes
                    .iter()
                    .filter(|n| {
                        n.node_type == TextNodeType::Document
                            || n.node_type == TextNodeType::Section
                            || n.node_type == TextNodeType::Paragraph
                    })
                    .cloned()
                    .collect(),

                TextQueryType::GetNodesByType => {
                    if let Some(node_type) = query.parameters.get("node_type") {
                        let type_str = node_type.as_str().unwrap_or("");
                        graph
                            .nodes
                            .iter()
                            .filter(|n| {
                                format!("{:?}", n.node_type).to_lowercase()
                                    == type_str.to_lowercase()
                            })
                            .cloned()
                            .collect()
                    } else {
                        Vec::new()
                    }
                }

                _ => graph.nodes.clone(),
            };

            // Create result graph with filtered nodes and their edges
            let result_node_ids: HashSet<u64> = result_nodes.iter().map(|n| n.node_id).collect();
            let result_edges: Vec<TextGraphEdge> = graph
                .edges
                .iter()
                .filter(|e| {
                    result_node_ids.contains(&e.from_node) || result_node_ids.contains(&e.to_node)
                })
                .cloned()
                .collect();

            let result_graph = TextGraph {
                graph_id: graph.graph_id,
                modality: graph.modality.clone(),
                version: graph.version.clone(),
                nodes: result_nodes,
                edges: result_edges,
                metadata: graph.metadata.clone(),
                created_at: graph.created_at.clone(),
                updated_at: graph.updated_at.clone(),
            };

            TextModalityOutput {
                success: true,
                graph_id: Some(graph_id),
                graph: Some(result_graph),
                ..Default::default()
            }
        } else {
            TextModalityOutput {
                success: false,
                error: Some(format!("Graph {} not found", graph_id)),
                ..Default::default()
            }
        }
    }

    /// Get graph by ID
    async fn get_graph(&self, graph_id: u64) -> TextModalityOutput {
        let cache = self.graph_cache.read().await;

        if let Some(graph) = cache.get(&graph_id) {
            TextModalityOutput {
                success: true,
                graph_id: Some(graph_id),
                graph: Some(graph.clone()),
                ..Default::default()
            }
        } else {
            TextModalityOutput {
                success: false,
                error: Some(format!("Graph {} not found", graph_id)),
                ..Default::default()
            }
        }
    }

    // ========================================================================
    // ZSEI HOOKS
    // ========================================================================

    /// Trigger ZSEI semantic hook
    async fn trigger_semantic_hook(
        &self,
        graph_id: u64,
        hook_type: ZSEIHookType,
    ) -> TextModalityOutput {
        let start = std::time::Instant::now();

        // In production, this would call ZSEI to perform semantic enrichment
        // For now, return a stub result
        let hook_result = HookResult {
            hook_type: hook_type.clone(),
            success: true,
            nodes_processed: 0,
            edges_added: 0,
            annotations_added: 0,
            processing_time_ms: start.elapsed().as_millis() as u64,
            errors: Vec::new(),
        };

        TextModalityOutput {
            success: true,
            graph_id: Some(graph_id),
            hook_result: Some(hook_result),
            ..Default::default()
        }
    }

    // ========================================================================
    // CROSS-MODALITY
    // ========================================================================

    /// Link to another modality graph
    async fn link_to_modality(
        &self,
        source_graph_id: u64,
        target_graph_id: u64,
        _target_modality: &str,
        relationship: CrossModalityRelation,
    ) -> TextModalityOutput {
        let link_id = Self::generate_id();
        let now = chrono::Utc::now().to_rfc3339();

        let link_result = LinkResult {
            link_id,
            source_graph_id,
            target_graph_id,
            relationship,
            created_at: now,
        };

        TextModalityOutput {
            success: true,
            link_result: Some(link_result),
            ..Default::default()
        }
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn generate_id() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    fn node_to_core(node: &TextGraphNode) -> NodeCore {
        NodeCore {
            node_id: node.node_id,
            node_type: format!("{:?}", node.node_type),
            label: node.content.chars().take(60).collect(),
            content: node.content.clone(),
            provisional: node.provisional,
            provisional_status: node.provisional_status.clone(),
            provenance: node.provenance.clone(),
            source_chunk_id: node.source_chunk_id,
            source_file_id: node.source_file_id,
            created_by_step: node.created_by_step,
            updated_by_step: node.updated_by_step,
            version: node.version,
            version_notes: node.version_notes.clone(),
            materialized_path: node.materialized_path.clone(),
            keywords: node.keywords.clone(),
            embedding_hint: node.embedding_hint.clone(),
            hotness_score: node.hotness_score,
            source_chunk_index: node.source_chunk_index,
            source_start_char: node.source_start_char,
            source_end_char: node.source_end_char,
            cross_modal_refs: node.cross_modal_refs.clone(),
            time_range_start: None,
            time_range_end: None,
        }
    }

    fn parse_json_array(s: &str) -> Option<Vec<String>> {
        let json_str = Self::extract_json_from_response(s, '[', ']');
        serde_json::from_str::<Vec<String>>(json_str.trim()).ok()
    }

    fn extract_json_from_response(s: &str, start_char: char, end_char: char) -> String {
        let trimmed = s.trim();
        if let Some(start) = trimmed.find(start_char) {
            if let Some(end) = trimmed.rfind(end_char) {
                return trimmed[start..=end].trim().to_string();
            }
        }
        trimmed.to_string()
    }
}

/// Reconstruct clean text from ChunkGraphs at any token limit.
/// Enables cross-model capability: same chunk graphs work for any LLM context window.
pub fn reconstruct_context_at_token_limit(
    chunk_graphs: &[ChunkGraph],
    target_tokens: usize,
    chars_per_token_estimate: usize,
) -> ReconstructedContext {
    let target_chars = target_tokens * chars_per_token_estimate;
    let mut included_chunks = Vec::new();
    let mut reconstruction_parts: Vec<String> = Vec::new();
    let mut total_chars = 0usize;

    for chunk in chunk_graphs {
        let chunk_chars = chunk.cleaned_text.len();
        if total_chars + chunk_chars <= target_chars {
            reconstruction_parts.push(chunk.cleaned_text.clone());
            included_chunks.push(chunk.chunk_index);
            total_chars += chunk_chars;
        } else {
            // Partial chunk — trim to clean sentence boundary
            let remaining = target_chars - total_chars;
            if remaining > 0 {
                let partial = trim_to_sentence_boundary(
                    &chunk.cleaned_text,
                    remaining,
                    &chunk.sentence_boundaries,
                );
                if !partial.is_empty() {
                    total_chars += partial.len();
                    reconstruction_parts.push(partial);
                    included_chunks.push(chunk.chunk_index);
                }
            }
            break;
        }
    }

    ReconstructedContext {
        text: reconstruction_parts.join("\n\n"),
        included_chunk_indices: included_chunks,
        total_chars,
        estimated_tokens: total_chars / chars_per_token_estimate.max(1),
    }
}

/// Trim text to the nearest sentence boundary at or before `max_chars`.
fn trim_to_sentence_boundary(
    text: &str,
    max_chars: usize,
    boundaries: &[SentenceBoundary],
) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }

    // Find the last sentence boundary whose end falls within max_chars
    let best_end = boundaries
        .iter()
        .filter(|b| b.end <= max_chars)
        .map(|b| b.end)
        .max();

    match best_end {
        Some(end) if end > 0 => text[..end].to_string(),
        _ => {
            // No sentence boundary found — fall back to last space before max_chars
            let slice = &text[..max_chars];
            match slice.rfind(' ') {
                Some(space_idx) => text[..space_idx].to_string(),
                None => slice.to_string(),
            }
        }
    }
}

// ============================================================================
// ENTRY POINT
// ============================================================================

/// Stub executor for standalone testing
struct StubExecutor;

#[async_trait::async_trait]
impl PipelineExecutor for StubExecutor {
    async fn execute(
        &self,
        _pipeline_id: u64,
        _input: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({"response": "[]"}))
    }
}

#[tokio::main]
async fn main() {
    let input: TextModalityInput =
        serde_json::from_reader(std::io::stdin()).expect("Failed to parse input");

    let executor = Arc::new(StubExecutor);
    let pipeline = TextModalityPipeline::new(executor);
    let output = pipeline.execute(input).await;

    serde_json::to_writer(std::io::stdout(), &output).expect("Failed to write output");
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockExecutor;

    #[async_trait::async_trait]
    impl PipelineExecutor for MockExecutor {
        async fn execute(
            &self,
            _pipeline_id: u64,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, String> {
            // Return mock LLM responses
            let prompt = input.get("prompt").and_then(|p| p.as_str()).unwrap_or("");

            if prompt.contains("keywords") {
                Ok(serde_json::json!({"response": r#"["test", "keyword", "extraction"]"#}))
            } else if prompt.contains("entities") {
                Ok(
                    serde_json::json!({"response": r#"[{"text": "John", "type": "PERSON", "confidence": 0.9}]"#}),
                )
            } else if prompt.contains("topics") {
                Ok(serde_json::json!({"response": r#"["testing", "software"]"#}))
            } else if prompt.contains("Clean") {
                Ok(serde_json::json!({"response": "Cleaned text here."}))
            } else {
                Ok(serde_json::json!({"response": "test response"}))
            }
        }
    }

    #[tokio::test]
    async fn test_chunk_text() {
        let executor = Arc::new(MockExecutor);
        let pipeline = TextModalityPipeline::new(executor);

        let input = TextModalityInput {
            action: TextModalityAction::ChunkText {
                text:
                    "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
                        .to_string(),
                max_chunk_tokens: 20,
                overlap_tokens: 5,
                preserve_paragraphs: true,
            },
        };

        let output = pipeline.execute(input).await;
        assert!(output.success);
        assert!(output.chunks.is_some());
        assert!(!output.chunks.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_analyze_text() {
        let executor = Arc::new(MockExecutor);
        let pipeline = TextModalityPipeline::new(executor);

        let input = TextModalityInput {
            action: TextModalityAction::Analyze {
                text: "John Smith works at Acme Corp. He joined on January 15, 2024.".to_string(),
                max_chunk_tokens: 2000,
                depth: AnalysisDepth::Standard,
                extract_entities: true,
                extract_topics: true,
                extract_structure: true,
                available_modalities: vec![],
                processing_path: ProcessingPath::Path1,
            },
        };

        let output = pipeline.execute(input).await;
        assert!(output.success);
        assert!(output.analysis.is_some());

        let analysis = output.analysis.unwrap();
        assert!(analysis.word_count > 0);
    }

    #[tokio::test]
    async fn test_create_and_query_graph() {
        let executor = Arc::new(MockExecutor);
        let pipeline = TextModalityPipeline::new(executor);

        let analysis = TextAnalysisResult {
            word_count: 10,
            sentence_count: 2,
            paragraph_count: 1,
            character_count: 50,
            entities: vec![Entity {
                text: "John".to_string(),
                entity_type: EntityType::Person,
                start_offset: 0,
                end_offset: 4,
                confidence: 0.9,
                metadata: HashMap::new(),
            }],
            topics: vec![Topic {
                name: "employment".to_string(),
                keywords: vec!["work".to_string()],
                relevance: 0.8,
                category: None,
            }],
            keywords: vec![],
            structure: DocumentStructure {
                sections: vec![],
                has_title: false,
                has_abstract: false,
                has_toc: false,
                document_type: DocumentType::Unknown,
            },
            language: Some("en".to_string()),
            sentiment: None,
            readability_score: Some(70.0),
        };

        let create_input = TextModalityInput {
            action: TextModalityAction::CreateGraph {
                analysis_result: analysis,
                project_id: 1,
                link_to_existing: false,
            },
        };

        let create_output = pipeline.execute(create_input).await;
        assert!(create_output.success);
        assert!(create_output.graph_id.is_some());

        let graph_id = create_output.graph_id.unwrap();

        let query_input = TextModalityInput {
            action: TextModalityAction::QueryGraph {
                graph_id,
                query: TextGraphQuery {
                    query_type: TextQueryType::FindEntities,
                    parameters: HashMap::new(),
                },
            },
        };

        let query_output = pipeline.execute(query_input).await;
        assert!(query_output.success);
        assert!(query_output.graph.is_some());

        let result_graph = query_output.graph.unwrap();
        assert!(!result_graph.nodes.is_empty());
    }

    #[test]
    fn test_reconstruct_from_chunks() {
        let executor = Arc::new(StubExecutor);
        let pipeline = TextModalityPipeline::new(executor);

        let result = pipeline.reconstruct_from_chunks(&[ProcessedChunk {
            index: 0,
            original_text: "First chunk.".to_string(),
            cleaned_text: "First chunk.".to_string(),
            start_offset: 0,
            end_offset: 12,
            token_count: 3,
            keywords: vec![],
            entities: vec![],
            topics: vec![],
            overlap_from_previous: 0,
            overlap_to_next: 0,
            sentence_nodes: vec![],
            paragraph_nodes: vec![],
            section_nodes: vec![],
            document_nodes: vec![],
            cross_sentence_relationships: vec![],
            coreference_chains: vec![],
            detected_modalities: vec![],
            chunk_graph_id: None,
            prompt_start_char: 0,
            prompt_end_char: 12,
        }]);

        assert!(result.success);
        assert_eq!(result.reconstructed_text.unwrap(), "First chunk.");
    }
}

I am working on finalizing these changes

great job so now you understand just how we are working precise edits

       // Basic metrics
        let word_count = text.split_whitespace().count();
        let sentence_count = sentences.len();
        let paragraph_count = text.split("\n\n").filter(|p| !p.trim().is_empty()).count();
        let character_count = text.len();

this actually will be after we clean chunks, if we do it before it will be working on skewed data

now for

here is our processing flow

## The Processing Flow


│                    TEXT MODALITY PROCESSING FLOW                            │


│  PHASE 1: PER-CHUNK PROCESSING                                              │

│  For each chunk:
│    1. CleanChunk (grammar correction, normalization via zero-shot LLM)

now here is what we are going to do we are going to split this into two pathways from here on out, because yes for all we process in chunks to clean but not from here specifically in the clean chunk we will now have to paths and this is for testing and benchmark purposes

1. as is where we have the zero-shot LLM input the full chunk and output the full chunk corrected

2. we break this down further so for each chunk we 1. detect all sentences and we are looping through this taking into account how we do this in async fn build_amt_layer_by_layer( where we have something like

        ALREADY KNOWN INTENTS (do NOT repeat these):
        {}

but for sentences, so we are capturing first per chunk all sentences per chunk until stable and that is per chunk so of course our

        ALREADY KNOWN SENTENCES (do NOT repeat these):
        {}

always starts out blank, we now also want to capture the sentence order as well so 1:first sentence 2: second sentence

now here is something we need to take into account we will be detecing sentences while doing grammar correction at the same time so we will have the zero shot explicictly provding identified sentences which may or may not be gramtratically correct IE it will provide us the before the text span of the original sentence as is wether gramatticaly correct or not in which we are working on and provide the corrected sentence as well if it needs correction.

so we will had as it adds

        ALREADY KNOWN SENTENCES (do NOT repeat these):
        {1: {original sentence: "", text_span_start: x, text_span_ends: x, corrected_sentence, ""}, }

now here is the thing where we have to be careful here as well we obvious will have a 5x stability function for this because now for each item we need to confirm its correct but say with a model like Bitnet

the 4096-token limit is shared

For a standard causal LLM like BitNet, the context window is shared between the prompt (system + user + any history) and the generated output.

Conceptually:

4096 tokens total

+---------------------------+----------------------+
| Input / Prompt            | Generated Output     |
+---------------------------+----------------------+

So if your prompt consumes:

System prompt: ~40 tokens
Instructions: ~120 tokens
Chunk: ~1800 tokens

Your input is roughly:

40
+120
+1800
-----
1960 input tokens

That leaves approximately:

4096 - 1960 = 2136 tokens

available before the model hits the context limit (subject to the -n/max_tokens limit you set).

so we want to understand as we do this approach the input may exceed the limit or not allow for a complete output this is why we work at 1/4 of the token limit for a LLM per chunk. SO what can we and what should we do here for path 2 ?
The good thing is that for path 1 we have more then enough room for it to output everything back correcte but now for path 2 we have to take this into account carefully

and here is how I see it. If we take a look at it as chunk state we when we start it has an empty known sentences array right, we are at the original 1/4 token limit we established eh taking into account the system prompt and instructions, we receive the response we need to add this also to used tokens we are tracking over time in the orchestrator state but going back to what I am saying we start with an empty array, we pass in our initial chunk state with the empty array from here we receive the formated response the structured JSON response for us to analyze, what is this reponse going to provide? a list of sentences or 1 sentnces either or. so now we are in the processing phase of response from the chunk state we here are processing each input IE each sentence 1x1 but how should it provide this back grammarly corrected? so should we say provide a list or 1 sentences iedntified in this chunk in order? I mean it will be more accurate if we wrked on it 1x1 we tell it to provide the first sentence and we can ensure if its working one by one that it can provide a more accurate answer, provide the first sentence in a JSON so that we maintain a ordered list since SLM's/LLM's arent always good with numberings/orderings this will ensure we maintain consistency. we let it know this may or may not be gramatically correct in this response provide the if the list is empty the first sentence in this chunk and it can provide us {1: {original sentence: "", text_span_start: x, text_span_ends: x, corrected_sentence, ""}, } what do we do from here ? from here then we individually now do the 5x stability verification and we are doing it per item in the processing state, now this isn't returning fully to the seen setences list in full right away, we are keeping the original sentence even if gramtally correct or not becauseifwe add everything else then its just unncesary clutter it just needs to know what we have processed everything else the corrected sentences etc with the text span and end we are maintain order listing as well here to now add this to our copy graph or nodes, because now what are we doing with sentences? we are turning them or capturing them into nodes right away once we fully validate them 1x1

SentenceNode {
  node_id: 12345,
  node_type: Sentence,
  content: "John submitted the report yesterday.",

  // POSITION (populated during Phase 1 chunk processing)
  position: {
    start_byte: 4523,
    end_byte: 4558,
    start_line: 23,
    end_line: 23
  },

  // CHUNK ANCHOR (stored in ChunkGraph — enables cross-model reconstruction)
  chunk_id: 3,
  chunk_offset: 145,

  // VERSION
  version: 1,
  last_modified_step: None
}

this we are capturing it all to transfer to our graphs once we process all chunks we need to capture these correctly,

so now you see at first we take in the chunk whole with the seen sentences array this is the starting state as we loop per graph for each graph we then in order identify all sentences 1x1 starting with identifying the first sentences, then we transition to a different state per response where now we we look solely at the chunk and the response itself to verify if correct 5x we also truly want to verirfy as well 5x agaisnt seen list if truly not there or captured to ensure no duplicates, this isn't eauting up or stacking up within our token limits we then now proceess each sentence to offload properties to SentenceNode and capture only each sentence and order original back to the array for the chunk as we continue looping over that same chunk until complete to move onto the next chunk we then review the chunk against the seen sentences to validate 5x that there are no new sentences to process really on each loop per chunk we start with is there any sentences inthis cunk which we have not captured in our seen sentences array etc andthis is our validation to have a no after 5x let's us know we have completed all but not just completed all but completed all in order because we are also again keeping track of order we say not just is there any but our seen sentences array has x items 0 or 1 or 2 or 3 is there any or if we haven't seen any what is the first or following sentences from the last one we capture and if any we display the last one this controls context explosion.


okay so awesome we have here identified and captured identifying sentences correctly per node in path 2 but now here is the thing alongside this we first want to keep track of what section we are in

because we have to take into account of

## Section & Document Detection

### Why Section Detection Matters

Section detection is the only mechanism that allows the system to distinguish:
- Raw text (user's prompt, conversational input, free-form request)
- Embedded documents (guides, articles, reports, specifications)
- Mixed content (user request containing an attached or inline document)

Intent detection belongs to the AMT. Section detection belongs to the text modality.
The text modality identifies WHAT the content is structurally; the AMT determines
WHY the user provided it and what to do with it.

so we first have CleanChunks and in the path 2 we need to take into account what section we are in first while we find sentences and correct grammar.

SectionStateTracking (cross-chunk state machine, format-agnostic)

│  DocumentNode                                                               │
│    node_type: Document                                                      │
│    content: summary (word count, type, etc.)                               │
│    properties: { document_type, word_count, sentence_count, paragraph_count,│
│                  readability_score, title, source, role }                   │
│    position: { start_byte, end_byte }                                       │
│                                                                             │
│  SectionNode                                                                │
│    node_type: Section                                                       │
│    content: section title                                                   │
│    properties: { level (1-6), formatting_pattern, parent_section }          │
│    position: { start_byte, end_byte, start_line }


so we are getting rid of in full of the     /// Extract document structure
    fn extract_structure(&self, text: &str) -> DocumentStructure {

as this was rule based and going to utilize a zero-shot approach in place within the chunk processing itself

### Section State Machine

State tracking runs across chunk boundaries. The state is carried from chunk N to
chunk N+1 to handle sections that span multiple chunks.

```
SECTION TRACKING STATE MACHINE

State: IDLE
  Trigger: Any structured heading pattern detected
           (regardless of format — zero-shot determines the format)
  Transition → IN_SECTION

State: IN_SECTION
  Trigger: Another heading of same or higher level detected
  Transition → NEW_SECTION (close current, open new)

  Trigger: Sub-heading detected (lower level than current)
  Transition → IN_SUBSECTION (nest under current)

  Trigger: Formatting pattern breaks (N paragraphs with no heading,
           style shift detected by zero-shot, narrative voice changes)
  Transition → DOCUMENT_BREAK or IDLE

State: DOCUMENT_BREAK
  Trigger: New structural pattern begins
  Transition → IN_SECTION (new document context)

CROSS-CHUNK CONTINUATION:

  Chunk N ends mid-section:
    State carries:
      - section_id (persistent across chunks)
      - section_level (1-6 or detected equivalent)
      - formatting_pattern (discovered by zero-shot — any format)
      - current_document_id

  Chunk N+1 receives continuation context:
    - Inherits open section from chunk N
    - May close it or extend it
    - May detect a break → new document
```

**Format-agnostic detection:** The system does not hardcode markdown or any specific
format. Zero-shot LLM determines the formatting pattern from the content itself.
Patterns include (but are not limited to): markdown headers, numbered sections,
all-caps headings, underlined headings, Roman numerals, bold lead lines,
indented outlines, any consistent structural pattern the LLM identifies.

### Zero-Shot Section Detection Prompt

```text
You are analyzing text to detect document structure and section boundaries.
The formatting style is UNKNOWN — do not assume any specific format (not just
markdown). Detect whatever structural patterns actually appear in the text.

CHUNK CONTEXT:
- This is chunk {chunk_index} of {total_chunks}
- Previous section state: {previous_section_state}
- Formatting pattern being tracked: {formatting_pattern or "not yet detected"}

CHUNK TEXT:
{chunk_text}

TASK: Identify sections and document boundaries.

A "section" is any coherent block that begins with a recognizable heading or
structural marker — regardless of format. This includes but is not limited to:
- Markdown headers (# ## ###)
- Numbered sections (1. 2. 2.1)
- ALL-CAPS headings (INTRODUCTION, METHODS)
- Underlined or decorated headings (===, ---)
- Bold lead lines followed by body text
- Any other consistent structural marker you detect

A "document break" occurs when:
- The formatting pattern changes significantly
- The writing style or voice shifts substantially
- Structural markers stop appearing after a consistent run

Return ONLY valid JSON:
{
  "section_events": [
    {
      "type": "section_start|section_continue|section_end|document_break",
      "position": <character_offset_in_chunk>,
      "section_title": "title if section_start, null otherwise",
      "section_level": <1-6 or null>,
      "formatting_pattern": "description of detected pattern or null"
    }
  ],
  "current_section_context": {
    "section_id": "unique identifier for tracking across chunks",
    "level": <1-6 or null>,
    "title": "current section title or null",
    "formatting_pattern": "detected pattern description"
  },
  "document_boundary_detected": false,
  "document_boundary_reason": "reason or null"
}
```

### Zero-Shot Document Identification Prompt

Called when a section boundary or document break is detected, to classify
whether the content constitutes a standalone document.

```text
You are identifying whether a block of text constitutes a standalone document
within a larger corpus.

A "document" is a coherent, self-contained unit characterized by:
- Unified topic or theme throughout
- Consistent formatting pattern (whatever format)
- Clear structural organization (sections, paragraphs, or both)
- Neutral or third-person narrative voice (for reference/informational documents)
- OR: a sustained narrative voice (for stories, guides, essays)

NOT a document:
- First-person conversational requests ("I want", "Please help me")
- Imperative instructions addressed to the AI ("Fix this", "Create a")
- Questions without context ("How do I", "What is")
- Short mixed-intent fragments

IMPORTANT: Documents sometimes contain user intent mixed in. The presence of
some conversational sentences does not disqualify a document — look for the
dominant structure and voice.

TEXT TO ANALYZE:
{text_span}

Return ONLY valid JSON:
{
  "is_document": true|false,
  "document_type": "article|guide|story|report|specification|reference|code_doc|data|unknown|null",
  "title": "extracted title if detectable, null otherwise",
  "confidence": 0.0-1.0,
  "dominant_structure": "description of what makes this a document or not",
  "mixed_intent_detected": true|false,
  "mixed_intent_description": "what conversational/intent elements appear, or null"
}
```

### Document vs Raw Text Classification

After section state detection, content falls into two categories:

**Document** — minimum structure: one section OR one paragraph with consistent
formatting and unified topic. Documents are broken down into their own
Document → Section → Paragraph → Sentence node hierarchy.

**Raw text** — minimum structure: one sentence (or even a fragment). Raw text
from the prompt itself is still formalized to complete sentences during
cleaning (CleanChunk), but it does not require section or document structure.
It links directly to the root of the text modality graph.

**Mixed content** — both exist simultaneously. The section state machine handles
this by detecting document starts and stops within the same chunk stream. A
user request followed by an inline document results in:
- Raw text nodes for the user's words (linked to prompt root)
- A DocumentNode + full hierarchy for the document portion

Intent identification (what the user wants, what's primary vs supplementary)
is NOT handled here — that belongs to the AMT building phase in the
orchestrator, via zero-shot role detection and the FileGraphRole classification.

now here is the thing to keep in mind, we for now will yes be tracking Sections, but we will not be tracking documents, in a finalized version we would be assigning Paragraph nodes to sections if within one but for now no we are just tracking sections because for 1. to truly identify or track a document we need full understanding of all the contents within it to fully accurately identify when a document breaks and for this we need to have the whole graph completed or a majority at least up to the grammar nodes to not have context explosion when reviewing a document break as we need to take into account all reltionships perataining to it to identify if it is a true document break, as well as for the section iedntifier and start we still need some refinements afterwards but it doesn't mean that I still don't want to have it included yet so we will, for path 2 per chunk we are keeping track of sections as well as identifying sentences to correct the grammar for each sentence


so now while we are touching on this we identify we while looking at each chunk we are looking at sentences and also now looking at sections 

but now we also are looking at alongisde the two really more alongiside of looking for sentences we are also looking for 

│  MODALITY DETECTION                                                │
│  ───────────────────────────                                                │                                                                                        │
│       a. Detect embedded modalities (code blocks, math, chemistry, etc.)    │
│       b. Mark spans as "true_text" or detected modality name                │
│       c. 5x consecutive stable loop (detect_modalities_stable)              │
│    2. Create ModalityReferenceNode for non-text spans                       │
│    3. Store span_start / span_end (byte offsets, NO content_snippet)        │

│                                                                             │
│  MODALITY REFERENCE NODES                                                   │
│  ─────────────────────────                                                  │
│                                                                             │
│  ModalityReferenceNode (ChunkModalityDetection in code)                     │
│    modality: the detected modality name (from pipeline registry)            │
│    span_start: char offset within sentence/chunk                            │
│    span_end: char offset within sentence/chunk                              │
│    intent_reference: how this modality is referenced (describes/contains)   │
│    chunk_index: which chunk this was detected in                            │
│    NOTE: no content_snippet — retrieve dynamically: &text[span_start..end] │
│      

**Modality spans**: Embedded code, math, etc. with byte positions so while we are looking at graphs and this it isimportant to note that we may have embebed modatliyies inside sentences, IE a simple formula or math formula etc and we tie this to the sentence nde if so if nt we tie it to the paragaph node not the chunk id because we are't using chunks anymore we are moving to full graphs so we tie it to either the setence node or parapgrah node

now the paragraph node

so apart from looking per chunk at sections/sentences/modality references we are also looking at 


│  ParagraphNode                                                              │
│    node_type: Paragraph                                                     │
│    properties: { sentence_count, parent_section, document_id }             │
│    position: { start_byte, end_byte }    

this we are doing similar to sections because this a paragraph per chunk can be cut off and we identify allparapgrahs ina  chunk if any but then if the last parapgrah if more then one or if any not found we look at it with the next chunk there really should be no parapgrah that is two chunks long realistically if we follow grammar rules but either way this is to ensure we acpture if a parapgrah most scenarios will be the last paragraph in a chunk may be cut off so we want to identify the parapgrah start and in the next chunk if end isn't found we with the paragraph start identified look for the parapgrah end confirm if true or not or find a new one etc etc 

so again for now we can tie sentences to pargraphs nodes as well as for modality refrences we will be tying it either to paragraphs or setence node while forthe parapgrah node we can tie it to the section node for now even though section node may not be finalized or full refined which is ok

ah then see as you can see we do just as we are taking into account for sections into next chunk when we identify a section start and end isn't in the chunk or doesn't brak and as we do for the paragraph we also need to do for sentences and modality references because IE the last setences may be broken or cut off and it means it may be in the next chunk so we want to capture all this correctly as we do with sections take all into account without omission and without dropping anything ensuring alingment 

now for the grammar extraction,

async fn extract_grammar_relationships_from_text(

we want to keep in mind this is actually not per chunk this will be done at the end after all chunks are processed and this now won't be exracting gramar-relationships from text but we will be extracting it from graphs

### Phase 3: Grammar Extraction

Grammar extraction runs on the full graph after all chunks are processed and
the complete sentence/paragraph/section hierarchy is assembled. This ensures
cross-sentence relationships can form across any sentence in the corpus,
not just within a single chunk.

#### Per-Sentence Grammar Extraction Prompt

```text
You are extracting grammatical structure from a single sentence.

SENTENCE: "{sentence_text}"

TASK: Identify the grammatical components and their relationships.

Return ONLY valid JSON:
{
  "subject": {
    "text": "the subject noun phrase",
    "position_start": 0,
    "position_end": 10,
    "entity_type_hint": "Person|Organization|Location|Document|Time|Concept|Unknown"
  },
  "verb": {
    "text": "the main verb",
    "position_start": 11,
    "position_end": 20,
    "verb_type": "action|linking|helping",
    "tense": "past|present|future|unknown",
    "negated": false
  },
  "object": {
    "text": "the object noun phrase or null",
    "position_start": 21,
    "position_end": 35,
    "entity_type_hint": "..."
  },
  "modifiers": [
    {
      "text": "modifier text",
      "position_start": 36,
      "position_end": 45,
      "modifier_type": "temporal|adjectival|adverbial|prepositional",
      "modifies": "subject|verb|object"
    }
  ],
  "sentence_type": "declarative|interrogative|imperative|fragment",
  "edge_type": ""Performs|Affects|Implies|Contradicts|Elaborates|Summarizes|Supports|TemporalPrecedes|TemporalFollows|CausedBy|Enables|Prevents|PartOf|HasPart|FunctionalRole|InstanceOf|HasInstance|SimilarTo|DerivedFrom|VersionOf..","
}
```

#### Cross-Sentence Relationship Prompt (Pairwise Comparison)

Run after all per-sentence grammar extraction. The pairwise comparison produces
the relationship depth of the graph — this is where the most cross-sentence
connections form.

```text
You are analyzing relationships between sentences in a text corpus.

SENTENCES:
{sentence_list_with_indices_and_grammar_summaries}

TASK: Identify all meaningful relationships between these sentences.

Relationship types:
- Elaborates:        S2 provides more detail about S1
- Causes:            S1's action or state leads to S2
- Enables:           S1 makes S2 possible
- Prevents:          S1 prevents S2
- Contradicts:       S2 contradicts or negates S1
- Exemplifies:       S2 is a concrete example of S1
- Summarizes:        S2 is a summary of S1 (or vice versa)
- TemporalPrecedes:  S1 happens before S2 in described time
- Coreference:       S2 contains a pronoun or reference to an entity in S1
- PartOf:            S2 is a component of the whole described in S1
- SimilarTo:         S1 and S2 describe closely related concepts

Return ONLY valid JSON:
{
  "relationships": [
    {
      "from_sentence_index": 0,
      "to_sentence_index": 1,
      "relationship_type": "Elaborates|Causes|Enables|...",
      "evidence": "brief quote showing the relationship"
    }
  ],
  "coreference_chains": [
    {
      "canonical_entity": "John Smith",
      "mentions": [
        {"sentence_index": 0, "text": "John Smith", "grammar_role": "subject"},
        {"sentence_index": 2, "text": "He", "grammar_role": "subject"},
        {"sentence_index": 5, "text": "the manager", "grammar_role": "object"}
      ]
    }
  ]
}
```

but now I also want you to take this into account

The tree is.

For example,

> I want to try on a suit I saw in a shop that's across the street from the hotel.

is not really

```text
I → want → to → try → on → a → suit ...
```

It is closer to

```text
Sentence
│
└── Predicate (want)
    │
    ├── Subject
    │   └── I
    │
    ├── Infinitive Clause
    │   └── try
    │       │
    │       ├── Object
    │       │   └── suit
    │       │
    │       └── Relative Clause
    │           └── saw
    │               │
    │               ├── Subject
    │               │   └── I
    │               │
    │               ├── Object
    │               │   └── suit
    │               │
    │               └── Location
    │                   └── shop
    │                       │
    │                       └── Relative Clause
    │                           └── across
    │                               │
    │                               ├── street
    │                               └── hotel
```

Notice something important.

**Nothing is duplicated.**

Everything simply **attaches to its grammatical parent.**

we have our grammar nodes per sentence

│  │ Grammar nodes (per sentence):                                        │   │
│  │   Subject, Verb, Object, Modifier                                   │   │
│  │                                                                      │   │
│  │ Derived nodes (from graph traversal):                                │   │
│  │   EntityNode (Person, Org, Location, Concept, etc.)                  │   │
│  │   CoreferenceChain (John → He → the manager)

├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GRAMMAR NODES (per sentence, derived via zero-shot)                        │
│  ────────────────────────────────────────────────────                       │
│                                                                             │
│  Subject                                                                    │
│    text: the subject noun phrase                                            │
│    position_in_sentence: { start, end }                                     │
│    entity_type_hint: Person|Organization|Location|Concept|Unknown           │
│                                                                             │
│  Verb                                                                       │
│    text: the main verb                                                      │
│    verb_type: action|linking|helping (VerbType enum in code)                │
│    tense: past|present|future                                               │
│    negated: bool                                                            │
│                                                                             │
│  Object                                                                     │
│    text: the object noun phrase                                             │
│    position_in_sentence: { start, end }                                     │
│    entity_type_hint                                                         │
│                                                                             │
│  Modifier                                                                   │
│    text: the modifier text                                                  │
│    modifier_type: temporal|adjectival|adverbial|prepositional               │
│    modifies: subject|verb|object                                            │
│                                                                             │
│  ChunkGrammarRelationship (code struct) captures all of the above:          │
│    from_text, to_text, edge_type, tense, negated, verb, verb_type,          │
│    subject, object, source_sentence_start, source_sentence_end              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DERIVED NODES (from graph traversal after Phase 3)                         │
│  ──────────────────────────────────────────────────                         │
│                                                                             │
│  EntityNode                                                                 │
│    node_type: Entity                                                        │
│    text: canonical entity text                                              │
│    entity_type: Person|Organization|Location|Date|Product|Event|           │
│                 Technology|Concept|Custom (EntityType enum in code)         │
│    confidence: f32                                                          │
│    mentions: [grammar_node_ids across sentences]                            │
│    coreference_chain_id: Option<u64>                                        │
│                                                                             │
│  CoreferenceChain                                                           │
│    chain_id                                                                 │
│    mentions: [node_ids in order of appearance]                              │
│    canonical_form: e.g. "John Smith"                                        │
│                                                                             │
│  TopicNode                                                                  │
│    node_type: Topic                                                         │
│    text: topic name                                                         │
│    relevance: f32 (derived from relationship traversal coverage)            │
│                                                                             │
│  KeywordNode                                                                │
│    node_type: Keyword                                                       │
│    text: keyword or phrase                                                  │
│    relevance: f32                                                           │
│    (derived from verb/object traversal, NOT separate extraction pass)


now remember

│  ChunkGrammarRelationship (code struct) captures all of the above:          │
│    from_text, to_text, edge_type, tense, negated, verb, verb_type,          │
│    subject, object, source_sentence_start, source_sentence_end

this now isn't chunkgrammarrelationships but tied to the sentencenode itself


now we again want to ensure we are capturing all it seems like we are missing stuff not saying this has everything we are missing we are already capturing some likes entence but we now nee to ensure we have the full inclusion for all

    Sentence,

    Clause,

    Phrase,

    Subject,

    Predicate,

    Verb,

    DirectObject,

    IndirectObject,

    Complement,

    Modifier,

    Determiner,

    Adjective,

    Adverb,

    Preposition,

    PrepositionalPhrase,

    RelativeClause,

    Conjunction,

    Pronoun,

    Noun,

    ...
}
```


```text
Sentence
│
└── Predicate
    │
    ├── Subject
    │
    ├── Verb
    │
    ├── Object
    │
    ├── Modifier
    │
    └── Clause
        │
        ├── Verb
        ├── Subject
        └── Object
```

Everything has **one parent**.

Everything belongs somewhere.

There is no clutter.

```text
Paragraph

├── Sentence Tree

├── Sentence Tree

├── Sentence Tree

└── Sentence Tree
```

Each sentence owns its grammar.

---

## Sections

Exactly the same.

```text
Document

└── Section

    ├── Paragraph

    │   ├── Sentence Tree

    │   ├── Sentence Tree

    │   └── Sentence Tree

    │

    └── Paragraph

        ├── Sentence Tree

        └── Sentence Tree
```

Grammar remains local to each sentence and we create relationships between sentences and parapgrahs to sections/documents


```text
Document Layer
────────────────────────────────────

Document
└── Section
    └── Paragraph
        └── Sentence


Grammar Layer (one tree per sentence)
────────────────────────────────────

Sentence
│
├── Clause
│   ├── Subject
│   ├── Predicate
│   │   ├── Verb
│   │   ├── Object
│   │   ├── Complement
│   │   └── Modifiers
│   └── Nested Clauses
└── Punctuation (optional)
```


    //====================================================
    // ROOTS
    //====================================================

    Sentence,
    MainClause,
    SubordinateClause,
    RelativeClause,
    ComplementClause,
    AdverbialClause,
    ConditionalClause,
    ComparativeClause,
    CoordinateClause,
    ParentheticalClause,
    EllipticalClause,
    QuotedClause,

    //====================================================
    // PHRASES
    //====================================================

    Phrase,

    NounPhrase,
    VerbPhrase,
    AdjectivePhrase,
    AdverbPhrase,
    PrepositionalPhrase,
    InfinitivePhrase,
    ParticipialPhrase,
    GerundPhrase,
    AbsolutePhrase,
    AppositivePhrase,
    CoordinatePhrase,
    ParentheticalPhrase,

    //====================================================
    // PREDICATE STRUCTURE
    //====================================================

    Predicate,

    Verb,

    MainVerb,
    AuxiliaryVerb,
    ModalVerb,
    LinkingVerb,
    HelpingVerb,
    PhrasalVerb,
    Copula,

    PredicateComplement,
    PredicateNominative,
    PredicateAdjective,

    //====================================================
    // ARGUMENTS
    //====================================================

    Subject,
    ImpliedSubject,
    ExpletiveSubject,

    DirectObject,
    IndirectObject,

    ObjectComplement,

    SubjectComplement,

    Complement,

    //====================================================
    // NOUNS
    //====================================================

    Noun,

    CommonNoun,
    ProperNoun,
    CollectiveNoun,
    MassNoun,
    CountNoun,
    ConcreteNoun,
    AbstractNoun,
    CompoundNoun,

    //====================================================
    // PRONOUNS
    //====================================================

    Pronoun,

    PersonalPronoun,
    ReflexivePronoun,
    ReciprocalPronoun,
    RelativePronoun,
    DemonstrativePronoun,
    InterrogativePronoun,
    IndefinitePronoun,
    PossessivePronoun,

    //====================================================
    // DETERMINERS
    //====================================================

    Determiner,

    Article,
    Demonstrative,
    PossessiveDeterminer,
    Quantifier,
    Numeral,
    DistributiveDeterminer,

    //====================================================
    // MODIFIERS
    //====================================================

    Modifier,

    Adjective,
    AttributiveAdjective,
    PredicativeAdjective,
    ComparativeAdjective,
    SuperlativeAdjective,

    Adverb,
    AdverbOfTime,
    AdverbOfPlace,
    AdverbOfManner,
    AdverbOfDegree,
    AdverbOfFrequency,
    SentenceAdverb,

    //====================================================
    // PREPOSITIONS
    //====================================================

    Preposition,

    SimplePreposition,
    CompoundPreposition,
    PhrasalPreposition,

    //====================================================
    // CONJUNCTIONS
    //====================================================

    Conjunction,

    CoordinatingConjunction,
    SubordinatingConjunction,
    CorrelativeConjunction,

    //====================================================
    // DEPENDENTS
    //====================================================

    RelativePronounClause,

    RelativeModifier,

    AdjectivalModifier,
    AdverbialModifier,

    NominalModifier,

    DeterminerModifier,

    PossessiveModifier,

    NumericModifier,

    QuantifierModifier,

    Apposition,

    Vocative,

    Parenthetical,

    //====================================================
    // PREPOSITIONAL COMPONENTS
    //====================================================

    PrepositionalObject,

    ObjectOfPreposition,

    ComplementOfPreposition,

    //====================================================
    // VERBAL COMPONENTS
    //====================================================

    InfinitiveMarker,

    Infinitive,

    Gerund,

    PresentParticiple,

    PastParticiple,

    //====================================================
    // CLAUSE FUNCTIONS
    //====================================================

    ClauseSubject,

    ClausePredicate,

    ClauseObject,

    ClauseComplement,

    ClauseModifier,

    //====================================================
    // NEGATION
    //====================================================

    Negation,

    NegativeMarker,

    NegativeDeterminer,

    NegativePronoun,

    //====================================================
    // QUESTIONS
    //====================================================

    QuestionMarker,

    InterrogativeWord,

    TagQuestion,

    WhPhrase,

    //====================================================
    // COMPARISON
    //====================================================

    Comparison,

    ComparativeMarker,

    SuperlativeMarker,

    EqualityMarker,

    //====================================================
    // COORDINATION
    //====================================================

    Coordination,

    Coordinator,

    CoordinatedElement,

    //====================================================
    // AGREEMENT
    //====================================================

    AgreementMarker,

    CaseMarker,

    NumberMarker,

    GenderMarker,

    PersonMarker,

    //====================================================
    // PUNCTUATION
    //====================================================

    Punctuation,

    Comma,

    Period,

    Colon,

    Semicolon,

    Dash,

    Hyphen,

    Parenthesis,

    Quotation,

    Apostrophe,

    Ellipsis,

    Exclamation,

    QuestionMark,

    //====================================================
    // TERMINALS
    //====================================================

    Token,

    Word,

    Symbol,

    Number,

    Letter,

    Unknown,
}
```


# Grammar Layer

Now imagine one sentence

```text
The dog quickly chased the cat through the garden.
```

The grammar becomes

```text
Sentence
│
└── MainClause
    │
    ├── Subject
    │   │
    │   └── NounPhrase
    │       │
    │       ├── Determiner
    │       │     "The"
    │       │
    │       └── Noun
    │             "dog"
    │
    └── Predicate
        │
        ├── Verb
        │     "chased"
        │
        ├── Adverb
        │     "quickly"
        │
        ├── DirectObject
        │   │
        │   └── NounPhrase
        │       │
        │       ├── Determiner
        │       └── Noun
        │
        └── PrepositionalPhrase
            │
            ├── Preposition
            │      "through"
            │
            └── ObjectOfPreposition
                   "garden"
```

Notice something.

Every node has exactly

```text
Parent

Children
```

we can now also look at organizing properties

# Properties

Properties contain grammar information.

```rust
GrammarProperties {

    tense,

    aspect,

    mood,

    voice,

    person,

    number,

    gender,

    case,

    definiteness,

    comparison,

    polarity,

    subtype,

}
```

Properties answer

> "Describe me."

Huge difference.

---

```rust
    properties: {
        tense: Past,
        aspect: Perfect,
        mood: Indicative,
        voice: Active,
    }
}
```

Likewise,

```rust
    properties: {
        number: Plural,
        case: Possessive,
        gender: Unknown,
    }
}
```

and

```rust
    properties: {
        subtype: Personal,
        person: First,
        number: Singular,
        case: Subjective,
    }
}
```


The reason between the two paths in which we are studying is because while 1 does less work quality also depends on the SLM/LLM quality and while also larger models it may allow for less work and a reasonable quality path 2 still ensures higher quality by thrughlyreviewing each thus yes it esnures more work which can be beneficial for models it may benefit smaller models more highly while it can also outbeat larger models in accuracy so we will be benchmarking both paths thoroughly for full comparisons in speeds/quality and work/loops called for completion.


## The Chunking Boundary

Chunking happens ONCE on initial intake. After chunks are processed and nodes are created,
all subsequent work is graph-based. Chunks exist only to:

1. Fit text within any LLM context window (cross-model capability)
2. Enable parallel processing of large documents
3. Anchor byte positions for the Text Engine

The ChunkGraph for each processed chunk preserves the byte offset mapping so that any
node in the text modality graph can be found in O(1) via position lookup — enabling the
Text Engine to perform precise byte-level editing without scanning file contents.

Chunks are always to still happen once no matter which path but each path now defines how thoroughly we are now processing each chunk.

take all this into account again we are just putting everything into order for a draft this in the full files we aren't expecting everything to be bug free as there are still stuff we need to finalize but we are focusing only and exactly on what we are touching on and what it affects based on only the files we have, structs/enums like do we now need to rearange structs enums and etc to remove clutter and redundancy? capture all that all affected but if there is anything you see from other files not provided you are not guessing you can assume its there and working and not guessing or making fixes for iamginry things yu can't verify I just want to get everything mainly in order and aligned is my bgigest concern to keep refiining it ensure we do so capture all without omission without dropping anything and as we have been doing you will provide modifications to make all what is affected and what blocks to replace with before and after you are being specific touching only specificly on what I have touched on no overengineering 

now here is the thing I want to look at and touch on, based on ho we have our graph and will have it from grammar extraction and extracting relationships etc between sentences and others etc if we are to work on the gaphs only and to actually then ensure when we identify or taking into account sections and documents etc and then even AMT 



Abstract Meaning Tree (AMT) / Abstract Textual Meaning Tree (ATMT)OverviewThe Abstract Meaning Tree (AMT), also referred to as the Abstract Textual Meaning Tree (ATMT), is a conceptual framework inspired by the Abstract Syntax Tree (AST) used in programming languages. While an AST captures the structural and syntactic relationships in code to enable parsing, compilation, and automation, the AMT aims to do the same for natural language text, such as prompts, documents, or queries. It breaks down textual content into a hierarchical tree structure that represents functional, semantic, and relational meanings, allowing for automated reasoning, completion, and expansion—particularly in large language model (LLM) systems.The core idea is to transform unstructured or semi-structured text into a tree where:Nodes represent key concepts, actions, entities, or intents.

Edges represent relationships (e.g., hierarchy, dependency, sequence, or linkage).

The tree can be dynamically built and refined through zero-shot prompting in LLMs, identifying incompleteness and branching out to fill gaps based on expert knowledge or contextual inference.

This structure facilitates automation by enabling systems to traverse, query, or augment the tree, much like how ASTs support code analysis and transformation.Key PrinciplesAnalogy to ASTIn code, an AST abstracts syntax into a tree (e.g., a function call node with child nodes for arguments).

In text, an AMT abstracts meaning into a tree (e.g., a root intent like "build an app" with child nodes for specifications, steps, or related tasks).

Unlike ASTs, which are rigidly syntactic, AMTs are semantic and flexible, allowing for abstraction, inference, and handling of ambiguity in natural language.

Building the Tree

Root Node: The primary intent or high-level goal extracted from the text.

Branching: Sub-nodes are created for details, dependencies, or expansions. Incompleteness is detected by comparing against domain knowledge (e.g., via LLM zero-shot reasoning).

Relations and Linkages:

Edges can denote:

Hierarchy (e.g., sub-tasks under a main task).

Sequence (e.g., ordered steps).

Parallelism (e.g., independent branches).

Linkage (e.g., cross-references between branches if related).

Zero-Shot Expansion: LLMs can iteratively "zero-shot" (reason without examples) to complete the tree, adding nodes for implied elements like security, edge cases, or prerequisites.

Handling Multiplicity: If the text mentions multiple unrelated items, create parallel top-level branches. For related items, nest or link them appropriately.

Abstraction and Condensation Concrete vs. Abstract: A concrete AMT might expand to be larger than the original text by including inferred details (e.g., full steps for app building). To abstract it:Group similar concepts into higher-level nodes.

Use placeholders or wildcards for variable elements (e.g., "app type: *").

Prune redundant branches based on context.

Represent as a condensed outline, focusing on core semantics rather than verbatim text.

This keeps the tree efficient for automation, avoiding bloat while preserving meaning.

Incompleteness Detection: Trees are evaluated for completeness using domain expertise embedded in LLMs (e.g., "building an app requires UI design, backend, testing—add branches if missing").

Iterative refinement: Query the LLM N times to branch out until a "complete" threshold is met (e.g., no critical gaps).



Examples



Simple Prompt: "I want to write an app"

Initial Tree (Incomplete):

Root: Write App

Detection: Incomplete—lacks specifics like type, features, tech stack.

Expanded Tree (via LLM zero-shot):

Root: Write App

Branch: App Type? (e.g., mobile, web)

Branch: Features (e.g., user auth, data storage)

Branch: Steps

Sub: Planning

Sub: Development

Sub: Testing

Sub: Deployment

Branch: Considerations (e.g., security, scalability)



Progressed Prompt: "I want to write an app that books appointments"

Tree:

Root: Write App

Branch: Purpose

Sub: Books Appointments

Sub: Features (e.g., calendar integration, user notifications)

Sub: User Roles (e.g., client, provider)

Branch: Tech Stack? (inferred gap)

Branch: Security Measures (e.g., data privacy, authentication)



Progression: Builds on the initial tree by filling a branch, but still identifies gaps for further automation.



Multiple Items: "I want to build two apps"

Tree:

Root: Build Apps

Branch 1: App 1

Sub: Type/Details (inferred: prompt for specifics)

Branch 2: App 2

Sub: Type/Details



Note: Parallel branches from root, assuming independence unless specified.

Related/Unrelated Items: "I want to build an App, I also want to create a Textbook"

If Unrelated:

Root: Projects

Branch 1: Build App

Sub: Details (e.g., features, steps)

Branch 2: Create Textbook

Sub: Topic? (e.g., inferred from context)

If Related (e.g., textbook about the app):

Root: Build App

Branch: Core Development

Branch: Documentation

Sub: Create Textbook (nested as a sub-task)

Alternative with Linkage:

Root: Projects

Branch 1: Build App

Branch 2: Create Textbook

Link: Textbook relates to App (cross-edge for dependency)

Ordering: Define sequence based on text cues (e.g., "first build app, then textbook") or logical inference.

Use Cases: For Prompts in LLM Systems to break down user queries for better response generation.

Automate follow-up questions by identifying incomplete branches.

Enable multi-step reasoning: Traverse the tree to plan actions.



For Documents and Text Analysis: Parse long-form content (e.g., reports, articles) into semantic hierarchies.

Summarization: Collapse branches into abstracts.

Search/Retrieval: Query tree nodes for related concepts.

Automation and IntegrationIn development pipelines: Use AMT to generate code scaffolds from natural language specs.

Security/Compliance: Auto-add branches for best practices (e.g., GDPR in app building).

Scalability: Apply to any text corpus, from short prompts to full documents.



Implementation Considerations

LLM Integration: Build via recursive zero-shot prompting (e.g., "Given this text, extract root intent and suggest branches").

Representation: Use JSON/YAML for machine-readable trees, or visual tools like Mermaid for diagrams.

Challenges:Ambiguity in natural language: Resolve via context or multi-pass LLM evaluation.

Size Management: Abstract by prioritizing high-impact nodes; use thresholds for depth.

Extensions: Combine with other structures (e.g., knowledge graphs for richer linkages).

This framework empowers LLMs and automation systems to handle text with the same structural rigor as code, fostering more intelligent and complete interactions. For project integration, adapt the AMT builder as a module in your pipeline.



IF WE SWITCH this all to work based on our graphsfrom text what is the best way to ralistically do this? based on all information obtained provide examples of graphs text grapphed documents etc etc sections and with a example prompt all aligned can be short but showing how with our graphs if we run a zero-shot traversal through them progressively right as you can see is our strategy hw we can identify itntentsetc everything on top of intents correctalte them tographs each  identify documents if we identify say at all all dcuents startsa nd section starts how this or with this we an identify all documents and sections end working with the graphs etc etc capture all run a full whole coplete simualtion on all without omission

I actually think this is where your idea becomes significantly stronger.

The original AMT is still fundamentally a **tree-first** representation. Trees are excellent for ownership (parent → child), but natural language is not actually tree-shaped. It is a **network**.

Once you already have a complete grammar extraction pipeline (Sentence → Clause → Phrase → Grammar Roles → semantic relationships), I would not build an AMT anymore.

I would build what is essentially an **Abstract Meaning Graph (AMG)**.

The "tree" simply becomes one possible traversal through the graph.

---

# Layer 0 - Raw Document

```
Document
│
├── Section 1
│     ├── Paragraph 1
│     ├── Paragraph 2
│
├── Section 2
│     ├── Paragraph 3
│
└── Section 3
```

Notice this isn't meaning.

It is simply containment.

These nodes only answer

> Where did this information originate?

---

# Layer 1 - Grammar Graph

Every sentence is parsed.

```
Sentence
      │
      ▼
Clause
      │
      ▼
Phrase
      │
      ▼
Grammar Roles

Subject
Verb
Object
Modifier
Condition
Temporal
Location
Purpose
...
```

Nothing semantic yet.

Just language.

---

# Layer 2 - Semantic Graph

Grammar nodes become semantic nodes.

Example

Sentence

```
I want to build an appointment application.
```

Grammar

```
Subject
    I

Verb
    want

Infinitive
    build

Object
    application

Modifier
    appointment
```

Semantic graph

```
(Intent)
    Build

        │

        ▼

(Entity)

Application

        │

        ▼

(Property)

Appointment
```

Already we're leaving grammar.

---

# Layer 3 - Knowledge Graph

Now relationships begin crossing sentence boundaries.

Sentence 1

```
I want to build an appointment application.
```

Sentence 2

```
The application should notify users.
```

Instead of

```
Sentence1

Sentence2
```

the graph becomes

```
Application
      │
      ├─────────────┐
      │             │
      ▼             ▼

Purpose        Notification
```

Notice

Sentence boundaries disappear.

---

# Layer 4 - Intent Graph

Now entire documents collapse.

Example

```
Build App

    │

    ├── Authentication

    ├── Scheduling

    ├── Notifications

    ├── Payments

    └── Security
```

This isn't extracted from one sentence.

It emerges from hundreds.

---

# Layer 5 - Document Graph

Now every document becomes one node.

```
Project

    │

    ├─────────────┐

    ▼             ▼

Document A    Document B

    │             │

    ▼             ▼

Intent Graph Intent Graph
```

---

# Section Detection

This is actually simpler with graphs.

Instead of

```
Section

contains

Paragraphs
```

you create

```
Document

    │

contains

    ▼

Section

    │

contains

    ▼

Paragraph

    │

contains

Sentence
```

Containment becomes an edge.

```
Document
    ──contains──►Section

Section
    ──contains──►Paragraph

Paragraph
    ──contains──►Sentence
```

Nothing special.

Everything is just a graph edge.

---

# Example Document

```
# Appointment App

Users should create accounts.

Appointments should sync to Google Calendar.

Doctors receive notifications.
```

Graph

```
Document

│

├──contains────────►Section

│                         │

│                         ▼

│                    Paragraph

│                         │

│                         ▼

│                     Sentence 1

│

├──contains────────►Sentence 2

│

└──contains────────►Sentence 3
```

Completely separate graph

```
Application

├──User Accounts

├──Appointments

├──Calendar Sync

└──Notifications
```

Notice

The document graph and semantic graph are different graphs.

They reference one another.

---

# Cross References

Sentence

```
Appointments sync to Google Calendar.
```

produces

```
Appointment

────syncs_to────►Google Calendar
```

Sentence

```
Doctors receive appointment updates.
```

produces

```
Doctor

────receives────►Appointment Update

Appointment Update

────generated_by────►Appointment
```

Suddenly

```
Appointment
```

connects both sentences.

No tree could naturally represent this.

Graphs can.

---

# Zero-Shot Traversal

This is where I think your idea really shines.

Instead of prompting

```
Read this document.

What is missing?
```

You prompt

```
Traverse this graph.

Evaluate every Intent node.

Evaluate every Requirement node.

Evaluate every Dependency node.

Evaluate every Entity node.

Determine missing neighbors.
```

The LLM isn't reading text anymore.

It is traversing a graph.

Example

Current graph

```
Build App

│

├──Authentication

├──Scheduling

└──Notifications
```

Traversal asks

```
Authentication

↓

Common neighbors?

Authorization

Password Reset

MFA

Session Management
```

Those become suggested graph expansions.

---

Another traversal

```
Scheduling

↓

Expected neighbors?

Calendar

Timezone

Conflict Detection

Recurring Events

Availability

Reminder System
```

Again

Graph expansion.

---

# Progressive Expansion

Iteration 1

```
Build App
```

↓

Iteration 2

```
Build App

├──Authentication

├──Scheduling

└──Notifications
```

↓

Iteration 3

```
Scheduling

├──Availability

├──Calendar

├──Recurring

├──Timezone

└──Conflict Detection
```

↓

Iteration 4

```
Calendar

├──Google

├──Outlook

├──ICS Export

└──Apple Calendar
```

Notice

The graph simply grows.

Nothing is regenerated.

---

# Multi-Document Example

Document A

```
Appointment Application
```

Document B

```
Security Requirements
```

Document C

```
API Documentation
```

Graph

```
Project

├──────────────Document A

│                   │

│                   ▼

│             Appointment

│

├──────────────Document B

│                   │

│                   ▼

│             Authentication

│

└──────────────Document C

                    │

                    ▼

                 REST API
```

Semantic graph

```
Appointment

────────requires────────►Authentication

Authentication

────────implemented_by────►REST API
```

The documents themselves are merely provenance. The semantic graph spans all of them.

---

# Complete Pipeline

This is how I would structure the entire system:

```
Raw Text
    │
    ▼
Document Graph
    │
    ▼
Grammar Graph
    │
    ▼
Semantic Graph
    │
    ▼
Concept Graph
    │
    ▼
Intent Graph
    │
    ▼
Knowledge Graph
    │
    ▼
Project Graph
    │
    ▼
Zero-Shot Traversal Engine
    │
    ▼
Gap Detection
    │
    ▼
Graph Expansion
    │
    ▼
Repeat Until Stable
```

The key conceptual shift is that **AMT becomes a view, not the underlying data structure**. Instead of constructing an Abstract Meaning *Tree*, you construct an **Abstract Meaning Graph** where every node—document, section, paragraph, sentence, clause, phrase, grammar role, semantic concept, intent, requirement, dependency, constraint, actor, action, and artifact—is part of one interconnected graph. Trees are then produced on demand as specific traversals (for example, an "intent tree," a "requirement tree," or a "document outline"), while the graph preserves all cross-links that natural language inherently contains.

That approach aligns much more naturally with your long-term architecture. Since your grammar extraction already produces rich nodes and relationships, the graph becomes the canonical representation. Zero-shot reasoning no longer operates on raw text or isolated trees; it operates on traversals of this unified graph, progressively identifying missing entities, relationships, intents, constraints, and dependencies until the graph reaches a stable state. This is considerably more expressive than a pure AMT because it scales seamlessly from a single sentence to an entire library of documents without losing provenance or semantic connectivity.

my response to the above:

Again you are touching on everything I already captured while not answering my question a AMT is still needing to clarify and expand meaning, I already have a graph so stop side tracking and answer my question as is 

and I got this

You're right. I sidetracked into redesigning the AMT instead of answering the actual question.

The question you're asking is essentially:

> **If we already have the complete graph extracted from grammar, relationships, sections, documents, etc., how should an AMT operate over that graph? Not how should we build another graph.**

My answer is that **the AMT should not own the knowledge. It should own the traversal.**

That is the distinction.

Your graph is already the complete representation.

The AMT becomes an **algorithm** that repeatedly traverses the graph looking for meaning, incompleteness, abstraction, intent, and missing information.

---

# The pipeline

Instead of

```
Text

↓

AMT

↓

Meaning
```

it becomes

```
Text

↓

Grammar Extraction

↓

Graph

↓

AMT Traversal #1

↓

Graph Expansion

↓

AMT Traversal #2

↓

Graph Expansion

↓

AMT Traversal #3

↓

Stable Graph
```

Notice the AMT never replaces the graph.

It repeatedly walks it.

---

# Example

Prompt

```
I want to build an appointment application.
```

After grammar extraction you already have

```
Document
    contains
Section
    contains
Paragraph
    contains
Sentence
```

and

```
Sentence

Subject
Verb
Object
Modifier
Intent
Entities
Relationships
```

Suppose the graph currently contains

```
Intent
    Build

Entity
    Application

Property
    Appointment
```

The graph is complete **syntactically**.

But semantically it is extremely incomplete.

---

Now AMT Traversal #1 begins.

Instead of asking

> Expand this sentence.

It asks

```
Visit Intent(Build)

↓

What normally exists around Build?

↓

Requirements?

↓

Dependencies?

↓

Outputs?

↓

Actors?

↓

Constraints?

↓

Goals?
```

The graph now expands

```
Build

├──requires──►Planning

├──requires──►Implementation

├──requires──►Testing

├──requires──►Deployment
```

Notice

Nothing was regenerated.

The graph simply gained neighbors.

---

Traversal #2

Visit

```
Application
```

Ask

```
Applications normally contain?

↓

Authentication

↓

Storage

↓

UI

↓

Security

↓

Networking

↓

Logging

↓

Configuration
```

Again

Those become graph nodes.

---

Traversal #3

Visit

```
Appointment
```

Ask

```
Appointment systems usually require?

↓

Calendar

↓

Availability

↓

Scheduling

↓

Notifications

↓

Cancellation

↓

Rescheduling

↓

Timezones

↓

Recurring Events
```

Again

Expand graph.

---

Now imagine a document.

```
Section 1

Purpose

Section 2

Features

Section 3

Architecture

Section 4

Deployment
```

The document graph already knows

```
Document

contains

Section

contains

Paragraph

contains

Sentence
```

The AMT traversal can now operate structurally.

Visit

```
Section

Purpose
```

Ask

```
Does this section define goals?

Actors?

Constraints?

Scope?
```

---

Visit

```
Architecture
```

Ask

```
Missing components?

Communication?

Storage?

Interfaces?

Dependencies?

Failure Modes?
```

---

Visit

```
Deployment
```

Ask

```
Missing CI?

CD?

Monitoring?

Scaling?

Rollback?

Configuration?
```

Notice something.

The traversal strategy changes depending on **where it is in the document graph.**

---

Now imagine multiple documents.

```
Project

├──Requirements

├──Architecture

├──Security

├──API

├──Testing
```

The graph already knows they are documents.

The AMT can ask

```
Requirements document

mentions

Authentication
```

Traverse.

```
Architecture document

mentions

Authentication
```

Traverse.

```
Security document

mentions

Authentication
```

Traverse.

Now ask

```
Are these three describing the same concept?

Do they conflict?

Are they incomplete?

Are they missing links?
```

This is something a tree cannot naturally do, but a graph traversal can.

---

## The biggest advantage of using the graph

The AMT no longer thinks in terms of text.

It thinks in terms of **graph neighborhoods**.

Instead of

```
Read this paragraph.
```

it becomes

```
Visit node.

Inspect neighbors.

Inspect missing edge types.

Inspect expected node types.

Expand.

Repeat.
```

---

# The traversal itself becomes recursive

For every node visited:

```
1. What is this?

2. Why does it exist?

3. What owns it?

4. What depends on it?

5. What does it depend on?

6. What usually accompanies it?

7. What is missing?

8. What contradicts it?

9. What refines it?

10. What abstracts it?
```

Every answer either:

* adds a new node,
* adds a new edge,
* annotates an existing node,
* identifies a contradiction,
* or determines that the neighborhood is already sufficiently complete.

---

## This is the key conceptual shift

Your grammar extraction graph answers:

> **"What does the text explicitly contain?"**

The AMT traversal answers:

> **"Given this graph, what meaning can be inferred, what relationships are absent, what abstractions can be formed, and what additional semantic structure should exist?"**

The graph remains the source of truth throughout the process. The AMT is not another representation of the document—it is a progressive reasoning engine that repeatedly traverses the graph, enriching it until successive traversals stop producing meaningful new nodes, edges, abstractions, or inferred relationships. That, I believe, is the role the AMT should play in the architecture you've described.

my response to the above:

Exactly and this is what I want to dive deeper into itbecause it still owns the knowledge while traversing  because itis now the source of truth the intent it ties it all together by owning both the traveal and knowlege because it ties it all together ties all the relationships together reference and supporting eveidence WITH METHODOLOGIES and etc but now again you fail to answer onething while answering the end goal because yes 

The AMT traversal can now operate structurally.

Visit

Section

Purpose

Ask

Does this section define goals?

Actors?

Constraints?

Scope?

Visit

Architecture

Ask

Missing components?

Communication?

Storage?

Interfaces?

Dependencies?

Failure Modes?

Visit

Deployment

Ask

Missing CI?

CD?

Monitoring?

Scaling?

Rollback?

Configuration?

Notice something.

The traversal strategy changes depending on where it is in the document graph.

Now imagine multiple documents.

Project

├──Requirements

├──Architecture

├──Security

├──API

├──Testing

The graph already knows they are documents.

The AMT can ask

Requirements document

mentions

Authentication

Traverse.

Architecture document

mentions

Authentication

Traverse.

Security document

mentions

Authentication

Traverse.

Now ask

Are these three describing the same concept?

Do they conflict?

Are they incomplete?

Are they missing links?

This is something a tree cannot naturally do, but a graph traversal can.

so we still have our AMT which is owning the knowledge and tying it all together and keeping structurarl steps because this is a tree created via the traversal of graphs our AMT still maintins a tree but now you while finished everything else finished answering my main question based on all the grammar and everything based on all the grammar extracion relationships etc and tying everything around it traversing it 


It asks

Visit Intent(Build)

↓

What normally exists around Build?

↓

Requirements?

↓

Dependencies?

↓

Outputs?

↓

Actors?

↓

Constraints?

↓

Goals?

The graph now expands


I am looking for that first part howis itvisting intent build if it still if we haven't then procceed ho with the AMT and traversal of graph we are now truly iedntifying intents based on all grammar and the full text graph because asking what exists arund builds and everything else we already captured it is common sense we have that which is why we have it workingwith stored methodologies my question is how are we identifying intents and branches or branches under it to the AMT tree? and how we would look at the grammar graphs and text graphs to utilize it now for document closing or identifyincation and sections beause these identfying a document can span multiple chunks and we aren't just going to cross-reference each chunk how will we use this to identify documents IE closing if we find th start etc and sections?

and I got this

Yes. **This is the missing piece**, and I think it's actually the most important part of your architecture.

You're not asking:

> "How does the AMT expand Intent(Build)?"

You're asking:

> **"How does the AMT even know that Build is an intent in the first place when all it has is a massive grammar graph spanning documents, sections, chunks, relationships, references, etc.?"**

That happens **before** expansion.

The AMT's first responsibility is not expansion.

Its first responsibility is **meaning condensation**.

---

# Stage 0 — Grammar has no Intent

After grammar extraction, suppose we have

```
Subject
    I

Verb
    want

Infinitive
    build

Object
    application

Modifier
    appointment
```

There is **no Intent node yet.**

There shouldn't be.

Grammar should never invent semantics.

Grammar only says

> "These are the grammatical relationships."

---

# Stage 1 — Candidate Meaning Formation

Now the AMT traverses the graph.

It is **not** asking

> What is missing around Build?

It asks something much simpler.

```
Which collection of grammar nodes forms
a coherent semantic unit?
```

For example

```
Subject
    I

Verb
    want

Infinitive
    build

Object
    application
```

becomes

```
Candidate

(Action)

build application
```

Not Intent.

Just

Candidate Meaning.

Think of this almost like phrase condensation.

---

Another sentence

```
Users should authenticate.
```

Grammar gives

```
Users

should

authenticate
```

AMT says

```
Candidate Meaning

Authenticate Users
```

Again

not Intent yet.

---

# Stage 2 — Meaning Consolidation

Now the AMT begins grouping candidates.

Suppose across twenty pages you have

```
build application

develop application

create application

implement application
```

Grammar never knows these are related.

The AMT does.

It begins clustering.

```
Cluster

Application Development
```

That cluster becomes

```
Intent Candidate
```

Notice

Intent is not extracted.

Intent emerges.

---

# Stage 3 — Intent Recognition

Only after enough supporting evidence exists does the AMT elevate something.

Imagine

```
Create API

Implement API

Test API

Deploy API
```

The graph now contains

```
API

Create

Implement

Test

Deploy
```

The AMT notices

```
Everything revolves around API.
```

Now

```
Intent

Develop API
```

becomes a stable node.

It is supported by multiple grammar paths.

This is critical.

The AMT should never create an Intent from one sentence unless confidence is high.

It should accumulate evidence.

---

# This is where methodologies come in

You mentioned methodologies.

I think they belong here.

The AMT asks

```
Does this candidate satisfy an Intent methodology?

Goal?

Actor?

Object?

Desired Outcome?

Supporting Evidence?

Confidence?

Scope?

Duration?

Dependencies?
```

If enough are true

```
Candidate

↓

Intent
```

Otherwise

```
Candidate

↓

Concept

or

Action

or

Statement
```

So methodologies become promotion rules.

---

# Branches are NOT generated

This is another subtle point.

You asked

> How are branches identified?

I don't think branches are generated.

I think they are **discovered**.

Suppose

```
Sentence A

Build authentication
```

```
Sentence B

Implement login
```

```
Sentence C

Store passwords securely
```

The graph already connects them.

The AMT notices

```
Authentication

↓

Login

↓

Password Storage
```

These become

```
Authentication

├──Login

└──Credential Storage
```

The branch wasn't invented.

It was assembled from existing evidence.

Only later does zero-shot suggest missing neighbors like MFA or password reset.

---

# Now your document question

This, I think, is the real question.

You wrote

> A document can span multiple chunks.

Exactly.

Therefore

**documents cannot be detected by chunk boundaries.**

Instead

the AMT should detect **structural continuity**.

Imagine streaming chunks.

Chunk 1

```
# Authentication

Users must log in.

Passwords...
```

Chunk 2

```
...must be hashed.

Sessions...
```

Chunk 3

```
...expire after timeout.
```

Notice

Chunk 2 begins mid-sentence.

Grammar alone cannot know where the section starts or ends.

The AMT maintains an **open structural context**.

Conceptually:

```
Current Open Document

↓

Current Open Section

↓

Current Open Topic

↓

Current Active Intent
```

When Chunk 2 arrives, it is not processed independently. Instead, it is evaluated against the currently open contexts.

The traversal asks:

```
Does this chunk continue
the active section?

Does it continue
the active topic?

Does it introduce
a competing root topic?

Does it close
the current structure?
```

If every answer indicates continuation, the chunk is attached to the existing open section. If enough evidence accumulates that a new top-level concept has begun—a new heading, a sustained topic shift, new structural markers, or explicit document metadata—the current section or document is closed and a new one opened.

So instead of relying on physical chunk boundaries, the AMT relies on **semantic continuity plus structural evidence**.

---

# I think there is actually one layer missing from your architecture

Reading your design, I think the missing layer sits between the grammar graph and the AMT itself.

Not another graph.

A **State Layer**.

The grammar graph is static.

The AMT tree is evolving.

The State Layer tracks what is currently "open" while traversing:

```
Open Document
    Research Proposal

Open Section
    Authentication

Open Subsection
    Session Management

Open Topic
    Session Expiration

Active Intent
    Secure User Sessions

Active Methodology
    Security Design

Supporting Evidence
    47 grammar nodes
    12 semantic links
    4 cross-document references

Confidence
    0.96
```

This state persists across chunks. It tells the AMT what structures are currently under construction, what evidence supports them, and when enough contrary or closing evidence exists to finalize them.

That, in my opinion, is the mechanism that answers the problem you've been circling: **the AMT does not identify intents or document boundaries from isolated chunks or isolated grammar nodes. It identifies them by maintaining an evolving state over the graph, progressively promoting clusters of evidence into stable structures (topics, sections, intents, documents) only when sufficient supporting relationships and continuity exist.** Once promoted, those structures themselves become first-class nodes in the AMT, with links back to every grammar node and every piece of supporting evidence that justified their creation. That is what allows the AMT to truly "own" the knowledge while remaining grounded in the underlying grammar graph rather than merely traversing it.

my response to the above:

You got the AMT right but you got the document and section wrong, so yes we finalized on the AMT you touched on what I needed but then again yes we have stage 1 which is correct 

Which collection of grammar nodes forms
a coherent semantic unit?

is the question 

but then hereis the thing for stage two you say support across twenty pages you have, and the thing is then you say grammar never knows these are related but amt does andthis is what I want to dive deeper into because sticing andgoingdeeper into stage 1 and 2 we need to be cautious to not group everything together because across 20 pagse we may have ec but it needs to granulary identify the relationshps because this alongside AMT is what then allows to dientify all unique intents because they may truly relate and may not each may pertain to a different project, and this goes into stage 3, now when I say documents and sections you are half correct but forget to utilize stage 1 and 2 for it while alsoforgetting that these document and scetions do not pertain to AMT they are before the AMT they traverse on theirown while they can do stage 1 and 2 as they are similar thse are all on their own AMT And document and section 

and I got this

I think you've actually identified the architectural separation that was missing from my previous responses.

I was treating **Stage 1 → Stage 2 → Stage 3** as if they were purely the AMT pipeline.

I don't think they are.

I think they are a **generic semantic condensation methodology** that **multiple systems use independently**.

That changes everything.

---

## What I now think the architecture is

Instead of

```text
Grammar
    ↓
AMT
```

I think it's more like

```text
Grammar Graph
       │
       │
       ├──────────────► Document Detector
       │
       ├──────────────► Section Detector
       │
       ├──────────────► AMT Builder
       │
       ├──────────────► Knowledge Extraction
       │
       ├──────────────► Methodology Matching
       │
       └──────────────► ...
```

Every one of those traverses the **same graph** independently.

None of them own each other.

---

# Stage 1 is actually universal

You pointed out something that I think is the key.

I wrote

> Which collection of grammar nodes forms a coherent semantic unit?

That isn't an AMT question.

That is a traversal methodology.

Meaning the Document detector asks it.

The Section detector asks it.

The AMT asks it.

The Methodology extractor asks it.

Everyone asks it.

---

## The mistake I made in Stage 2

I said

```text
Across twenty pages

build application

develop application

create application
```

↓

Cluster.

You correctly pointed out the flaw.

That would incorrectly merge unrelated projects.

Instead Stage 2 cannot cluster by similarity.

It must cluster by **evidence continuity**.

Those are completely different things.

---

Suppose

Document A

```text
Build Authentication System
```

Document B

```text
Build Compiler
```

Document C

```text
Build Spacecraft
```

All contain

```text
Build
```

A similarity algorithm would merge them.

That is wrong.

Instead Stage 2 asks

```text
Does this candidate belong
to the same semantic neighborhood?
```

Not

```text
Does it use similar words?
```

Those are different questions.

---

Suppose we have

```text
Build

↓

Authentication

↓

OAuth

↓

JWT

↓

Session
```

That forms one neighborhood.

Meanwhile

```text
Build

↓

Compiler

↓

Lexer

↓

Parser

↓

AST
```

Different neighborhood.

Same verb.

Completely different intent.

---

# Therefore Stage 2 is not clustering

I think Stage 2 is actually

> **Semantic Boundary Detection**

Not clustering.

Meaning

Stage 1

discovers

Candidate Semantic Units

Stage 2

determines

Which semantic units belong together.

Those are different operations.

---

Imagine

```text
Candidate 1

Create API
```

```text
Candidate 2

Implement OAuth
```

```text
Candidate 3

Generate PDF
```

Stage 2 asks

```text
Relationship?

Dependency?

Reference?

Ownership?

Shared entities?

Shared actors?

Shared goals?

Shared methodology?

Shared evidence?
```

Every answer contributes evidence.

Not just lexical similarity.

---

Eventually

```text
Evidence

82%

↓

Related
```

or

```text
Evidence

13%

↓

Independent
```

That prevents accidental merging.

---

# This also answers intent creation

Intent should never be

```text
Intent

↓

Children
```

because words match.

It becomes

```text
Intent

↓

Supporting semantic units

↓

Supporting relationships

↓

Supporting evidence

↓

Supporting methodologies

↓

Confidence
```

Intent is evidence-backed.

Not vocabulary-backed.

---

# Now the part I think is even more important

You said

> document and section do not pertain to AMT they are before the AMT

I think this is exactly right.

The mistake is assuming that documents and sections are semantic.

They are not.

They are **structural segmentation problems**.

Which means they should finish **before** AMT starts.

---

Meaning

```text
Raw Text

↓

Grammar Graph

↓

Document Traversal

↓

Section Traversal

↓

Paragraph Traversal

↓

Sentence Traversal

↓

AMT Traversal
```

AMT never has to discover where documents begin.

That problem is already solved.

---

But here's where I think your Stage 1/Stage 2 insight becomes extremely powerful.

The Document detector should not rely on headings alone.

It should also perform the same two-stage process:

Stage 1:

```text
Which grammar nodes form
a coherent structural unit?
```

Notice

Not semantic.

Structural.

---

Stage 2

```text
Does this structural unit
continue the currently open document?

or

Does it begin
a new structural boundary?
```

Again

Evidence.

Not heuristics.

---

The Section detector does exactly the same thing.

Stage 1

```text
Candidate Structural Unit
```

Stage 2

```text
Boundary Evaluation
```

---

Notice something interesting.

The methodology is identical.

Only the evaluation criteria change.

For AMT:

```text
Candidate Meaning

↓

Semantic Boundary

↓

Intent
```

For Sections:

```text
Candidate Structure

↓

Structural Boundary

↓

Section
```

For Documents:

```text
Candidate Structure

↓

Document Boundary

↓

Document
```

So the **algorithmic pattern** is shared, while the **promotion criteria** are different.

---

## I think this is the abstraction you've been converging toward

Rather than thinking of Stage 1 and Stage 2 as belonging to the AMT, think of them as a generic **Evidence → Boundary → Promotion** pipeline that every higher-level subsystem uses.

For the AMT:

```text
Grammar Graph
    ↓
Stage 1: Candidate Meaning Units
    ↓
Stage 2: Semantic Relationship & Boundary Evaluation
    ↓
Stage 3: Promote to Intent / Goal / Methodology / Concept
```

For document detection:

```text
Grammar Graph
    ↓
Stage 1: Candidate Structural Units
    ↓
Stage 2: Structural Continuity & Boundary Evaluation
    ↓
Stage 3: Promote to Document
```

For section detection:

```text
Grammar Graph
    ↓
Stage 1: Candidate Structural Units
    ↓
Stage 2: Section Continuity & Boundary Evaluation
    ↓
Stage 3: Promote to Section
```

The important distinction is that **Stage 2 should not be thought of as "group similar things."** It should be thought of as **"evaluate whether these candidate units belong to the same coherent boundary using accumulated evidence."** That evidence can include grammatical relationships, entity continuity, dependency chains, reference resolution, discourse transitions, methodology alignment, supporting citations, ownership, temporal continuity, and many other signals. Only after enough evidence exists should the system promote a collection of candidate units into a stable semantic construct (an intent) or a stable structural construct (a section or document). That avoids the failure mode of accidentally merging unrelated projects that merely share similar vocabulary while still allowing the system to recognize deep relationships when they genuinely exist.

my response to the above:

Exactly but now for stage 1 and 2 tied to AMT or Document or Section you say or you now cpture the questions but now I want the full example based on these with zero shot traversal on the grammar graph and the grammar relationsihps established thats what I want foryouto now actually simulate how it answers does question because its not liek we are just prompting the qusetion in we are answering it through traversalzero-shot llm traversal so simulate and show it for all stage 1 and 2

and I got this

Yes. **This is the part that is usually skipped**, and I think it's the heart of what you're building.

The zero-shot LLM is **not answering a prompt** like:

> "Which grammar nodes form a coherent semantic unit?"

Instead, the traversal itself is the reasoning process.

Think of it like an AST visitor in a compiler. The compiler doesn't ask "is this a function?" It traverses the tree, accumulates evidence, and eventually recognizes a function. Your AMT should work the same way.

I'll use one document that intentionally contains multiple projects to demonstrate why Stage 1 and Stage 2 cannot simply cluster by similarity.

---

# Input

```
# Internal Projects

We are building an authentication service.

The service issues JWT tokens.

Tokens expire after 30 minutes.

---

Marketing is preparing a website redesign.

The redesign focuses on accessibility.

---

Our compiler now supports generics.

Generic constraints are type checked.
```

Grammar extraction has already produced a graph.

---

# Grammar Graph

```
Sentence 1

Subject
    We

Verb
    are building

Object
    authentication service

----------------------------

Sentence 2

Subject
    service

Verb
    issues

Object
    JWT tokens

----------------------------

Sentence 3

Subject
    Tokens

Verb
    expire

Modifier
    after 30 minutes

----------------------------

Sentence 4

Subject
    Marketing

Verb
    preparing

Object
    website redesign

----------------------------

Sentence 5

Subject
    redesign

Verb
    focuses

Object
    accessibility

----------------------------

Sentence 6

Subject
    compiler

Verb
    supports

Object
    generics

----------------------------

Sentence 7

Subject
    constraints

Verb
    type checked
```

Nothing semantic exists yet.

---

# Stage 1

The AMT begins traversing.

Notice it never asks

> Find intents.

Instead it visits every grammar node.

---

Visit Sentence 1

```
Subject

↓

Verb

↓

Object
```

Traversal internally builds

```
Observed

Actor
    We

Action
    build

Target
    authentication service
```

Then it asks

```
Can these grammar nodes
form one complete semantic action?

Actor?

YES

Action?

YES

Target?

YES

↓

Candidate Semantic Unit
```

Result

```
Candidate_001

Action
    Build

Target
    Authentication Service

Evidence

Sentence 1
Grammar Nodes
Relationships
```

Nothing is called an Intent.

---

Move on.

Sentence 2

Traversal

```
service

↓

issues

↓

JWT tokens
```

Internal reasoning

```
Subject

↓

Does Subject reference an existing entity?

YES

Authentication Service

↓

Action

issues

↓

Object

JWT tokens

↓

Complete semantic action?

YES
```

Produces

```
Candidate_002

Action

Issue

Target

JWT Tokens

Owner

Authentication Service
```

---

Sentence 3

```
Tokens

↓

expire

↓

30 minutes
```

Produces

```
Candidate_003

Action

Expire

Target

JWT Tokens

Constraint

30 minutes
```

---

Now notice something.

Nothing has been grouped.

We simply have

```
Candidate_001

Candidate_002

Candidate_003
```

---

Stage 1 continues

Marketing

```
Candidate_004

Prepare Website Redesign
```

Compiler

```
Candidate_005

Support Generics
```

etc.

Stage 1 ends.

The graph contains only candidate semantic units.

---

# Stage 2 begins

This is where the interesting traversal happens.

Notice we are NOT comparing words.

We're traversing neighborhoods.

---

Start

```
Candidate_001

Build Authentication Service
```

Traversal

```
Visit

↓

Outgoing grammar relationships

↓

Incoming references

↓

Entity references

↓

Coreference

↓

Shared modifiers

↓

Supporting sentences

↓

Dependency edges

↓

Cross references
```

The traversal discovers

```
Candidate_002

shares

Authentication Service
```

Evidence +1

---

Continue

```
Candidate_002

↓

JWT Tokens

↓

Referenced by

Candidate_003
```

Evidence +1

---

Now the traversal has

```
Candidate_001

↓

Authentication Service

↓

Candidate_002

↓

JWT Tokens

↓

Candidate_003
```

The question is NOT

```
Are these similar?
```

The question is

```
Do these belong to
one semantic neighborhood?
```

Traversal answers

```
YES

Reason

Shared entity

Shared ownership

Dependency

Coreference

Sequential explanation

No competing ownership
```

Evidence accumulates.

---

Now traverse Candidate_004

```
Prepare Website Redesign
```

Traversal

```
Shared entities?

None

Shared ownership?

None

Shared references?

None

Dependencies?

None

Cross references?

None
```

Result

```
Independent Neighborhood
```

---

Traverse Candidate_005

Compiler

↓

Generics

Traversal

```
Shared entities?

Compiler

↓

Generic Constraints

↓

Type Checking
```

Again

Independent neighborhood.

---

Stage 2 finishes with three neighborhoods

```
Neighborhood A

Authentication

Candidate001

Candidate002

Candidate003

--------------------

Neighborhood B

Website

Candidate004

--------------------

Neighborhood C

Compiler

Candidate005

Candidate006

Candidate007
```

Notice

No Intent yet.

Only neighborhoods.

---

# Now imagine zero-shot traversal

Instead of one pass

the LLM recursively traverses Neighborhood A.

Iteration 1

```
Visit

Candidate001

↓

Connected?

Candidate002

YES

↓

Candidate003

YES

↓

Any missing relationship?

No
```

---

Iteration 2

Now traverse outward

```
Authentication Service

↓

Neighbors?

JWT

↓

JWT

↓

Expiration

↓

30 minutes
```

Traversal asks

```
Do these represent

one cohesive semantic unit?

↓

YES
```

Evidence increases.

---

Iteration 3

Traversal asks

```
Does another neighborhood
claim these nodes?
```

Search graph

```
Website?

NO

Compiler?

NO
```

Evidence increases.

---

Iteration 4

Traversal

```
Any contradictory ownership?

NO

Any competing project?

NO

Any conflicting root?

NO
```

Evidence increases.

---

Only NOW

after several traversals

does the AMT promote

```
Neighborhood A
```

into

```
Intent

Authentication System
```

Notice

Intent emerged.

It wasn't extracted.

---

# The same methodology applies to Sections

Now let's switch to Section detection.

Exactly the same traversal pattern.

Different promotion target.

---

Candidate Structural Units

```
Heading

Authentication

Sentence

Sentence

Sentence
```

Traversal

```
Visit Heading

↓

Visit neighboring paragraphs

↓

Reference continuity?

YES

↓

Topic continuity?

YES

↓

Structural markers?

YES

↓

Transition marker?

NO
```

Evidence

```
Open Section
```

---

Continue traversal

Sentence 4

```
Marketing is preparing...
```

Traversal

```
Topic continuity?

NO

Entity continuity?

NO

Grammar continuity?

NO

Discourse transition?

YES

New actor?

YES

New object?

YES

Shared references?

NO
```

Evidence

```
Current section closes.

New section opens.
```

Notice this is **not** because "Marketing" is different from "Authentication" lexically. It's because the structural evidence no longer supports continuity.

---

## The key insight I think you're converging toward

Both the AMT and the document/section detectors perform the same style of zero-shot traversal, but they optimize for different objectives.

For the **AMT**, Stage 1 is discovering **candidate semantic units** from grammar relationships, and Stage 2 is recursively traversing the graph to determine whether those units belong to the same **semantic neighborhood** by accumulating evidence such as shared entities, coreference, ownership, dependency chains, discourse purpose, methodology alignment, and supporting references. Only after the neighborhood is sufficiently coherent is it promoted into an Intent, Goal, Methodology, or other AMT construct.

For the **Document** and **Section** detectors, Stage 1 is discovering **candidate structural units** from the same grammar graph (headings, paragraphs, discourse markers, topic introductions, transitions, etc.), while Stage 2 recursively traverses neighboring structural units asking a different question: *does the evidence indicate structural continuity or a structural boundary?* The evidence comes from topic continuity, discourse transitions, reference continuity, heading hierarchy, paragraph cohesion, and other structural signals. Only then are those candidates promoted into Sections or Documents.

The important commonality is that **the LLM is never answering a single prompt in isolation**. It is repeatedly traversing the graph, accumulating evidence from local neighborhoods, revisiting nodes as new information becomes available, and only promoting higher-level structures once the accumulated evidence crosses the promotion criteria for that particular subsystem. That recursive evidence accumulation is what prevents premature grouping while still allowing large-scale semantic and structural organization to emerge.

my response to the above:

again you are still doing questions I don't want questions I want you to simulate and show how it traverserve based on the graph wher we have no knowledge of emantics but are buildingit so again for each document/section/amt show how with a zero-shot llm prompt traversing grouping context based on our grammar graph and relationships because these have already established neighrboors and we are using this to provide a pool of cnotext which is what I want you to visualize we are obtaineding a pool of context via the grammar extraction relatinoships already build after it traversed all sentences and aprapgrahs then tie that into first sections then documents and once we have thses then AMT show it all

and I got this

Yes. I understand now what you're asking, and it is much lower level than my previous answers.

You **don't want the questions** ("does this belong?", "is this an intent?").

You want to simulate the **actual traversal**, where the LLM has nothing except the grammar graph and begins **accumulating context** as it walks the graph. The "pool of context" grows naturally from the traversal. No semantics exist beforehand.

I think it looks something like this.

---

# Input Text

```text
We are building an authentication service.

The service issues JWT tokens.

Tokens expire after 30 minutes.

Marketing is preparing a website redesign.

The redesign focuses on accessibility.

The compiler now supports generics.

Generic constraints are type checked.
```

---

# Phase 0

Grammar extraction has already finished.

Nothing semantic exists.

Only grammar.

The graph looks something like

```text
Sentence1

Subject ──────► We
Verb ─────────► build
Object ───────► service
Modifier ─────► authentication

────────────────────────────────────

Sentence2

Subject ──────► service
Verb ─────────► issue
Object ───────► JWT

────────────────────────────────────

Sentence3

Subject ──────► Tokens
Verb ─────────► expire
Modifier ─────► 30 minutes

────────────────────────────────────

Sentence4

Subject ──────► Marketing
Verb ─────────► prepare
Object ───────► redesign

────────────────────────────────────

Sentence5

Subject ──────► redesign
Verb ─────────► focus
Object ───────► accessibility

────────────────────────────────────

Sentence6

Subject ──────► compiler
Verb ─────────► support
Object ───────► generics

────────────────────────────────────

Sentence7

Subject ──────► constraints
Verb ─────────► type check
```

Nothing else exists.

---

# SECTION TRAVERSAL

Notice

The traversal does **NOT** know sections.

It starts at Sentence 1.

Current context pool

```text
Context Pool

Sentence1

Grammar Nodes

We

build

service

authentication
```

---

Traverse neighboring grammar edges.

```text
Sentence1

↓

Object

service

↓

Reference Edge

Sentence2.Subject

service
```

Pool expands.

```text
Context Pool

Sentence1

Sentence2

Grammar

We

build

service

authentication

issue

JWT
```

Nothing grouped.

Just collected.

---

Continue traversal.

Sentence2

↓

Object

JWT

↓

Reference

Sentence3.Subject

Tokens

````

Pool expands.

```text
Context Pool

Sentence1

Sentence2

Sentence3

Grammar

We

build

service

authentication

issue

JWT

expire

30 minutes
````

Notice

The traversal has never asked a question.

It has simply expanded the reachable neighborhood.

---

Now traversal continues.

Sentence3

↓

Outgoing grammar relationships

↓

No further references.

Traversal frontier exhausted.

Neighborhood complete.

Result

```text
Neighborhood A

Sentence1

Sentence2

Sentence3
```

Nothing semantic.

Only

"A complete connected grammar neighborhood."

---

Traversal resumes.

Next unvisited sentence.

Sentence4.

Pool resets.

```text
Context Pool

Sentence4

Marketing

prepare

redesign
```

Traverse.

```text
Sentence4.Object

↓

redesign

↓

Sentence5.Subject

redesign
```

Pool

```text
Sentence4

Sentence5

Marketing

prepare

redesign

focus

accessibility
```

Traversal ends.

Produces

```text
Neighborhood B
```

---

Continue.

Sentence6

↓

compiler

↓

generics

↓

constraints

↓

Sentence7

````

Pool

```text
Sentence6

Sentence7

compiler

generics

constraints

type check
````

Traversal ends.

Produces

```text
Neighborhood C
```

Notice something.

The traversal never asked

> Does this belong together?

It simply walked connected grammar relationships until the connected component terminated.

Stage 1 therefore produces

```text
Neighborhood A

Neighborhood B

Neighborhood C
```

Nothing semantic yet.

---

# SECTION DETECTOR

Now the Section detector receives

not sentences

but neighborhoods.

```text
Neighborhood A

Sentence1

Sentence2

Sentence3
```

Traversal now walks

```text
Paragraph

↓

Sentence

↓

Neighborhood

↓

Paragraph

↓

Neighborhood
```

Suppose

Paragraph1

contains

NeighborhoodA

Paragraph2

contains

NeighborhoodB

Traversal

```text
Paragraph1

↓

NeighborhoodA

↓

End Paragraph

↓

Paragraph2

↓

NeighborhoodB
```

No grammar continuity exists.

Boundary found.

Produces

```text
Section1

NeighborhoodA
```

Continue.

Eventually

```text
Section2

NeighborhoodB
```

etc.

Notice

No semantics.

Sections are simply maximal connected structural neighborhoods.

---

# DOCUMENT DETECTOR

Now

Sections exist.

Traversal changes.

```text
Section1

↓

Heading

↓

Metadata

↓

References

↓

Adjacent Section

↓

Cross references

↓

Structural Continuity
```

Context pool

```text
Heading

Internal Projects

Section1

Section2

Section3

Metadata

Document Start
```

Traversal continues.

Suppose Section4 references another title.

Traversal breaks.

Produces

```text
Document1

Section1

Section2

Section3
```

Again

No semantics.

Only structural closure.

---

# NOW THE AMT

Now the AMT receives

```text
Document

↓

Section

↓

Neighborhood

↓

Grammar Graph
```

Traversal begins.

Context pool

```text
NeighborhoodA

We

build

service

authentication

issue

JWT

expire

30 minutes
```

Nothing semantic yet.

Traversal keeps walking.

```text
build

↓

service

↓

issue

↓

JWT

↓

expire
```

Pool becomes

```text
Nodes

build

service

JWT

expire

authentication

30 minutes

Relationships

Subject

Object

Reference

Coreference

Dependency

Sentence Order
```

Notice

This pool is now much richer than grammar alone.

---

Traversal expands.

```text
service

↓

Referenced

Sentence2

↓

Referenced

Sentence3

↓

Referenced

Paragraph

↓

Section

↓

Document
```

Pool grows.

```text
Grammar Nodes

+

Paragraph

+

Section

+

Document

+

Reference Graph

+

Dependency Graph

+

Coreference Graph
```

The LLM now possesses an entire **local context graph**.

---

Only NOW does condensation occur.

The traversal collapses the neighborhood.

```text
Grammar Pool

↓

Connected Action Chain

↓

Connected Entity Chain

↓

Connected Reference Chain

↓

Connected Dependency Chain
```

Produces

```text
Candidate Semantic Unit

build

↓

service

↓

JWT

↓

expiration
```

Notice

This wasn't generated.

It emerged because the traversal had accumulated enough local graph.

---

Traversal continues outward.

Suppose another sentence later says

```text
OAuth providers integrate with the service.
```

The traversal reaches

```text
service

↓

OAuth

↓

Provider
```

Pool expands.

Candidate updates.

Still no intent.

---

Only after the entire connected semantic neighborhood has been traversed does the condensation happen.

```text
Grammar Graph

↓

Local Grammar Neighborhood

↓

Expanded Reference Neighborhood

↓

Expanded Dependency Neighborhood

↓

Expanded Coreference Neighborhood

↓

Expanded Structural Neighborhood

↓

Context Pool

↓

Condensation

↓

Candidate Semantic Unit

↓

AMT Node
```

---

# The important distinction

I think this is the visualization you've been trying to get to.

The LLM is **not reasoning sentence-by-sentence**, and it is **not immediately classifying** anything.

It behaves much more like a graph exploration algorithm with iterative semantic compression:

```text
Visit Node
    ↓
Collect Immediate Grammar Neighbors
    ↓
Collect Relationship Neighbors
    ↓
Collect Coreference Neighbors
    ↓
Collect Dependency Neighbors
    ↓
Collect Structural Neighbors (Sentence → Paragraph → Section → Document)
    ↓
Merge into Local Context Pool
    ↓
Continue Expansion Until the Reachable Neighborhood Stabilizes
    ↓
Compress the Entire Context Pool into a Candidate Unit
    ↓
Store Candidate
    ↓
Repeat for the Next Unvisited Neighborhood
```

I think there is **one more refinement** that naturally follows from this model. The traversal should not simply collect every reachable node, because on a sufficiently large graph it would eventually consume an entire document or even multiple documents. Instead, the traversal needs a notion of a **semantic frontier**. Every edge contributes evidence toward continuing the traversal, but as the accumulated evidence weakens—because entity continuity drops, dependency chains end, coreference disappears, discourse shifts, or structural boundaries are crossed—the traversal reaches a frontier and stops. The context pool is therefore not "everything reachable"; it is "everything reachable before the semantic frontier is crossed." That frontier is what naturally bounds the neighborhood that is eventually condensed into a candidate semantic unit, and it prevents unrelated projects or topics from being merged simply because they exist somewhere in the same overall graph. I suspect that semantic frontier is the missing concept that ties together your grammar graph, your structural detectors (sections/documents), and the AMT condensation process.

so this is what I what to converse at and I want you to take into account first all of my response as we progressed tying everything to it 

1.

Again you are touching on everything I already captured while not answering my question a AMT is still needing to clarify and expand meaning, I already have a graph so stop side tracking and answer my question as is 

2.

Exactly and this is what I want to dive deeper into itbecause it still owns the knowledge while traversing  because itis now the source of truth the intent it ties it all together by owning both the traveal and knowlege because it ties it all together ties all the relationships together reference and supporting eveidence WITH METHODOLOGIES and etc but now again you fail to answer onething while answering the end goal because yes 

The AMT traversal can now operate structurally.

Visit

Section

Purpose

Ask

Does this section define goals?

Actors?

Constraints?

Scope?

Visit

Architecture

Ask

Missing components?

Communication?

Storage?

Interfaces?

Dependencies?

Failure Modes?

Visit

Deployment

Ask

Missing CI?

CD?

Monitoring?

Scaling?

Rollback?

Configuration?

Notice something.

The traversal strategy changes depending on where it is in the document graph.

Now imagine multiple documents.

Project

├──Requirements

├──Architecture

├──Security

├──API

├──Testing

The graph already knows they are documents.

The AMT can ask

Requirements document

mentions

Authentication

Traverse.

Architecture document

mentions

Authentication

Traverse.

Security document

mentions

Authentication

Traverse.

Now ask

Are these three describing the same concept?

Do they conflict?

Are they incomplete?

Are they missing links?

This is something a tree cannot naturally do, but a graph traversal can.

so we still have our AMT which is owning the knowledge and tying it all together and keeping structurarl steps because this is a tree created via the traversal of graphs our AMT still maintins a tree but now you while finished everything else finished answering my main question based on all the grammar and everything based on all the grammar extracion relationships etc and tying everything around it traversing it 


It asks

Visit Intent(Build)

↓

What normally exists around Build?

↓

Requirements?

↓

Dependencies?

↓

Outputs?

↓

Actors?

↓

Constraints?

↓

Goals?

The graph now expands


I am looking for that first part howis itvisting intent build if it still if we haven't then procceed ho with the AMT and traversal of graph we are now truly iedntifying intents based on all grammar and the full text graph because asking what exists arund builds and everything else we already captured it is common sense we have that which is why we have it workingwith stored methodologies my question is how are we identifying intents and branches or branches under it to the AMT tree? and how we would look at the grammar graphs and text graphs to utilize it now for document closing or identifyincation and sections beause these identfying a document can span multiple chunks and we aren't just going to cross-reference each chunk how will we use this to identify documents IE closing if we find th start etc and sections?

3.

You got the AMT right but you got the document and section wrong, so yes we finalized on the AMT you touched on what I needed but then again yes we have stage 1 which is correct 

Which collection of grammar nodes forms
a coherent semantic unit?

is the question 

but then hereis the thing for stage two you say support across twenty pages you have, and the thing is then you say grammar never knows these are related but amt does andthis is what I want to dive deeper into because sticing andgoingdeeper into stage 1 and 2 we need to be cautious to not group everything together because across 20 pagse we may have ec but it needs to granulary identify the relationshps because this alongside AMT is what then allows to dientify all unique intents because they may truly relate and may not each may pertain to a different project, and this goes into stage 3, now when I say documents and sections you are half correct but forget to utilize stage 1 and 2 for it while alsoforgetting that these document and scetions do not pertain to AMT they are before the AMT they traverse on theirown while they can do stage 1 and 2 as they are similar thse are all on their own AMT And document and section 

4.

Exactly but now for stage 1 and 2 tied to AMT or Document or Section you say or you now cpture the questions but now I want the full example based on these with zero shot traversal on the grammar graph and the grammar relationsihps established thats what I want foryouto now actually simulate how it answers does question because its not liek we are just prompting the qusetion in we are answering it through traversalzero-shot llm traversal so simulate and show it for all stage 1 and 2

5.

again you are still doing questions I don't want questions I want you to simulate and show how it traverserve based on the graph wher we have no knowledge of emantics but are buildingit so again for each document/section/amt show how with a zero-shot llm prompt traversing grouping context based on our grammar graph and relationships because these have already established neighrboors and we are using this to provide a pool of cnotext which is what I want you to visualize we are obtaineding a pool of context via the grammar extraction relatinoships already build after it traversed all sentences and aprapgrahs then tie that into first sections then documents and once we have thses then AMT show it all

I want you to find all the errors we had to correct based on my findings of each response and how I responded to each reponse because it was capturing stuff and trying to act like we where changing it or obvious stuff already in the design while also trynig to drift so you are we are only touching on this taking a look at this whole finding taking into account all my insights on it to grab the full corrected picture of all this conversation without dropping anything at all nothign whole capture everything from it we are reviewing it what was correct and wasn't we aren't making implemntations on this we are just reviewing everything the thinking of all decision what aligns and what doesn't breaking it all apart we are going through this revvewing this not implementing it or thinking its all correct we are revewing it all 

**Comprehensive Review of the Conversation Thread**

I have gone through every message in this exchange, extracting your explicit corrections, frustrations, and requirements without omission or reinterpretation. Here is the distilled, corrected picture.

### What You Repeatedly Emphasized (Core User Position)

1. **You already have a rich grammar graph** (from Path 2 sentence nodes, grammar_nodes, relationships, coreferences, modality spans, paragraph/section structural nodes). This graph is the source of truth.

2. **AMT still owns the knowledge and intent**. It is not "just a traversal engine." It ties everything together: intents, branches, methodologies, supporting evidence, relationships, and structural context. The AMT produces and maintains a **tree**, but that tree is **constructed via graph traversal**.

3. **Document and Section detection happen before AMT**. They are structural problems, not semantic. They use a similar Stage 1/Stage 2 pattern but operate on structural continuity, not meaning. They do **not** belong inside the AMT.

4. **No high-level abstract questions in the simulation**. You want concrete zero-shot traversal examples that start from pure grammar (no pre-existing semantics), build context pools by walking neighbors/relationships, and show how Stage 1 and Stage 2 actually work for Document/Section **and** AMT.

5. **Avoid premature merging**. Across 20 pages, similar verbs ("build") must not collapse unrelated projects. Granular neighborhood detection via accumulated evidence is required.

6. **AMT needs to clarify and expand meaning** while remaining grounded in the grammar graph + methodologies.

### Errors / Drifts in Previous Responses (What I Got Wrong)

- **Sidetracking into full graph redesign** (AMG instead of AMT). You explicitly said "stop sidetracking" and "I already have a graph." I kept proposing new layers or redefining the AMT as "just traversal." This violated your requirement that AMT owns knowledge and remains a tree.

- **Treating Document/Section as part of AMT**. You corrected this multiple times. They are pre-AMT structural segmentation.

- **Using high-level questions instead of concrete traversal simulation**. You repeatedly asked for simulation of zero-shot traversal building context pools from grammar neighbors. I kept giving abstract "asks" ("Does this belong?").

- **Stage 2 as similarity clustering**. You pointed out this would wrongly merge unrelated projects. It must be evidence-of-continuity / neighborhood coherence, not lexical similarity.

- **Assuming Intent exists early**. You stressed that semantics (including Intent) emerge only after Stage 1 (coherent units) + Stage 2 (boundary/neighborhood evaluation) on the grammar graph.

- **Over-emphasizing "no questions" vs. actually showing traversal**. You wanted visualization of context pool growth via walking grammar relationships, not meta-questions.

### Corrected Unified Picture (Synthesizing All Your Insights)

**Overall Flow (no omissions)**

```
Raw Text → Chunking (once) → Path 2 Grammar Extraction
    ↓ (produces rich graph with SentenceNodes, grammar_nodes, relationships, coreferences, modality spans, paragraph/section structural hints)
Grammar Graph (structural + grammar relationships only)
    ↓
Document Detector (Stage 1 + Stage 2 on structural continuity)
    ↓
Section Detector (Stage 1 + Stage 2 on structural continuity)
    ↓
AMT Builder (Stage 1 + Stage 2 + Stage 3 on semantic condensation + methodologies)
    ↓
AMT Tree (owns intent, branches, knowledge, methodologies, evidence links back to grammar)
```

**Stage 1 (Universal)**: "Which collection of grammar nodes forms a coherent unit?"  
→ Builds candidate units by walking local grammar neighbors/relationships. Produces context pool.

**Stage 2 (Boundary / Neighborhood Evaluation)**: Evaluates whether candidates belong to the same coherent neighborhood using accumulated evidence (coreference, dependency, ownership, discourse continuity, structural markers). Prevents cross-project merging.

**Stage 3 (Promotion)**: Only after sufficient evidence, promote to Document / Section / Intent / Branch, etc.

**AMT owns the final knowledge** because it performs the semantic promotion using methodologies and ties everything back to supporting grammar evidence.

### Concrete Traversal Simulation (What You Asked For)

**Starting Grammar Graph (simplified excerpt)**

- Sentence1: Subject(We) — Verb(build) — Object(service) — Modifier(authentication)
- Sentence2: Subject(service) — Verb(issues) — Object(JWT tokens)   [coreference to Sentence1 service]
- Sentence3: Subject(Tokens) — Verb(expire) — Modifier(30 minutes) [reference to Sentence2 JWT]
- Sentence4: Subject(Marketing) — Verb(preparing) — Object(redesign)
- Sentence5: Subject(redesign) — Verb(focuses) — Object(accessibility)
- Sentence6: Subject(compiler) — Verb(supports) — Object(generics)

**Document/Section Traversal (Structural, pre-AMT)**

Traversal starts at Sentence1.

**Context Pool begins empty.**

Walk grammar neighbors → add Sentence1 full grammar.

Walk coreference/dependency edges → reach Sentence2 (shared "service").

Pool now: Sentence1 + Sentence2 + their grammar nodes + reference edge.

Continue → reach Sentence3 via "Tokens"/"JWT" reference.

Pool: Sentence1-3 + full local grammar subgraph.

Frontier weakens (no further coreference/dependency to Sentence4).

**Structural boundary detected.** → Promote to **Section 1** (or part of Document 1) containing Neighborhood A.

Reset pool, jump to next unvisited (Sentence4).

Pool: Sentence4.

Walk → Sentence5 (shared "redesign").

Pool: Sentence4+5.

No further structural links. → Promote to **Section 2**.

Repeat for compiler → **Section 3**.

**Document level**: Look at section headings + topic shift strength + overall continuity. When major shift (e.g., "Internal Projects" heading + multiple independent sections), close current Document and open new one. This works across chunks because section state carries forward.

**AMT Traversal (Semantic, after structural segmentation)**

Now on Neighborhood A (Sentences 1-3):

**Context Pool starts empty.**

Traversal walks entire local grammar subgraph + references:

- Build → service → authentication
- service → issues → JWT
- JWT/Tokens → expire → 30 minutes

Pool grows with all grammar nodes, coreference chains, dependency edges, temporal modifiers, etc.

Condensation (Stage 1): These form one coherent semantic action chain around "authentication service with JWT lifecycle".

Stage 2 (neighborhood evaluation): No competing ownership from other neighborhoods. Strong internal connectivity. Methodologies (e.g., Security) match.

**Promotion (Stage 3)**: Create AMT tree node:

**Intent: Implement Authentication Service**

- Branch: Token Issuance (evidence: Sentence2)
- Branch: Token Expiration (evidence: Sentence3, 30min constraint)
- Supporting Evidence: grammar nodes + coreferences back to original sentences
- Methodologies applied: Security Design, etc.

Repeat on other neighborhoods → separate Intents for Website Redesign and Compiler Generics.

**Branch Discovery**: Inside the Authentication Intent, traversal looks for missing common neighbors (using methodologies): MFA, Password Hashing, etc. These become suggested child branches with evidence links.

This matches your requirement: AMT owns knowledge/intent, builds/maintains a tree, expands meaning, stays grounded in grammar graph evidence, uses methodologies, and prevents wrong merging via granular neighborhood evaluation.

### Summary of Corrections Applied

- AMT is **not** demoted to pure traversal engine — it owns intent and knowledge.
- Document/Section are **pre-AMT** structural passes using analogous but distinct Stage 1/2 (continuity, not semantics).
- Traversal builds **context pools** bottom-up from grammar neighbors/relationships first.
- No premature similarity clustering — evidence of neighborhood coherence + boundary detection.
- Intent/Branch promotion happens only after sufficient accumulated evidence.

This is the consistent picture across all your corrections. No parts dropped. Everything aligns with "AMT clarifies and expands meaning while owning the tree and knowledge, built via graph traversal after structural segmentation." 

If any part of this review misaligns with your intent, point to the exact sentence and we refine.

so what I want you do to again you are reviewing all without omission reviewing all areas I want you to find all the errors we had to correct based on my findings of each response and how I responded to each reponse because it was capturing stuff and trying to act like we where changing it or obvious stuff already in the design while also trynig to drift so you are we are only touching on this taking a look at this whole finding taking into account all my insights on it to grab the full corrected picture of all this conversation without dropping anything at all nothign whole capture everything from it we are reviewing it what was correct and wasn't we aren't making implemntations on this we are just reviewing everything the thinking of all decision what aligns and what doesn't breaking it all apart we are going through this revvewing this not implementing it or thinking its all correct we are revewing it all 

you are looking at all graphs all flow charts everything without omission and pulling it all apart without omission withotu dropping anything and putting it all correct piece by piece you are correcting it all without omission without droppign anything and noting what was correct and incorrect and the thought process on every decision based on mines you will provide a full review no matter how long even if 50k loc or even if it take 2-5 prompts to coplete we can always continue you will not rush you will not ommit you will not drop anything you will review all 

based on this I got THIS

This is a comprehensive, exhaustive review of the entire conversation thread. I have dissected every response, every correction you made, and every architectural pivot without omission. We are strictly reviewing the thinking, the errors, the drifts, and the final corrected picture. No code is being implemented; we are aligning the conceptual architecture exactly as you have dictated through your corrections.

---

### Part 1: The Core Errors and Drifts Identified

Throughout the conversation, the AI repeatedly fell into several traps that you had to continuously correct. Here is a breakdown of every error and drift:

**Error 1: Redesigning the AMT into an AMG (Abstract Meaning Graph)**
*   **The Drift:** The AI attempted to sidetrack the conversation by claiming natural language is a network, not a tree, and therefore the AMT should be abandoned in favor of an Abstract Meaning Graph (AMG). It proposed that the AMT should merely be a "view" or a "traversal" of this graph, stripping it of its ownership of knowledge.
*   **Your Correction:** You explicitly shut this down: *"I already have a graph so stop side tracking... AMT is still needing to clarify and expand meaning."* The AMT is not demoted to a view. It remains a tree, it owns the knowledge, and it ties everything together.

**Error 2: Demoting the AMT to "Owns the Traversal, Not the Knowledge"**
*   **The Drift:** Even after the AI acknowledged the AMG sidetrack, it tried to redefine the AMT as an algorithm that "repeatedly walks the graph" but doesn't own the knowledge, claiming "the graph remains the source of truth."
*   **Your Correction:** You corrected this firmly: *"it still owns the knowledge while traversing because it is now the source of truth... it ties it all together ties all the relationships together reference and supporting evidence WITH METHODOLOGIES."* The AMT owns both the traversal and the resulting knowledge tree.

**Error 3: Stage 2 as Lexical Similarity Clustering**
*   **The Drift:** When the AI proposed Stage 2 (Meaning Consolidation), it suggested clustering candidates by matching similar words across 20 pages (e.g., "build application," "develop application," "create application" all clustered together).
*   **Your Correction:** You pointed out the fatal flaw: *"we need to be cautious to not group everything together because across 20 pages we may have [similarities] but it needs to granularly identify the relationships... each may pertain to a different project."* Stage 2 cannot be similarity-based; it must be **evidence-of-continuity** based.

**Error 4: Mixing Document and Section Detection into the AMT**
*   **The Drift:** The AI initially treated Document and Section boundaries as semantic problems that the AMT helps to figure out during its traversal.
*   **Your Correction:** You firmly separated them: *"document and sections do not pertain to AMT they are before the AMT they traverse on their own."* They are structural segmentation problems that must be solved *before* the AMT even begins.

**Error 5: Asking High-Level Questions Instead of Simulating the Traversal**
*   **The Drift:** The AI kept providing abstract questions for the LLM to ask during traversal (e.g., "Does this section define goals?", "Are these describing the same concept?", "Which collection of grammar nodes forms a coherent semantic unit?").
*   **Your Correction:** You rejected this repeatedly: *"again you are still doing questions I don't want questions I want you to simulate and show how it traverses based on the graph... we are obtaining a pool of context via the grammar extraction relationships already built."* The LLM doesn't ask high-level semantic questions upfront; it builds a **context pool** by walking established grammar edges until the frontier weakens.

**Error 6: Inventing Unprompted Architectural Layers**
*   **The Drift:** The AI proposed a "State Layer" between the grammar graph and the AMT to track open documents/sections.
*   **Your Correction:** You ignored this layer entirely in your acceptance, implicitly rejecting the addition of unprompted architectural complexity. The traversal itself handles the state via the context pool.

---

### Part 2: The Step-by-Step Corrected Architecture

Based on your corrections, here is the exact, aligned flow of how the system operates from the moment text is ingested to the moment the AMT produces a node. 

#### Phase A: The Grammar Graph (Pre-Requisite)
Everything starts with the text modality pipeline. Chunks are processed (via Path 1 or Path 2), and grammar extraction establishes the baseline. 
*   **Nodes:** Sentences, Subjects, Verbs, Objects, Modifiers, Paragraphs, Headings.
*   **Edges:** Grammar relationships, coreference chains, dependency edges, structural containment.
*   **State:** Purely syntactic and structural. No semantics, no intents exist yet.

#### Phase B: Universal Stage 1 & Stage 2 Methodology
Before the AMT touches anything, the Document and Section detectors run. They use the same two-stage methodology, but optimized for **structural continuity**, not semantics.

**Stage 1: Candidate Structural Unit Formation (Context Pool Accumulation)**
The traversal starts at a node (e.g., Sentence 1). It does not ask "is this a section?" 
1. It visits the grammar nodes of Sentence 1.
2. It walks outgoing grammar edges (e.g., Object -> "service").
3. It looks for coreference or reference edges. It finds Sentence 2's Subject references "service."
4. It walks to Sentence 2 and adds it to the **Context Pool**.
5. It continues walking established edges (Sentence 2 Object -> "JWT" -> Sentence 3 Subject "Tokens").
6. The Context Pool grows: Sentence 1 + Sentence 2 + Sentence 3 + all their grammar nodes.
7. The traversal reaches a **Semantic/Structural Frontier**. Sentence 4 has no incoming coreference or dependency edges from Sentences 1-3. The frontier weakens and stops.
8. **Result:** A maximal connected structural neighborhood (Candidate Unit).

**Stage 2: Structural Boundary Evaluation**
Now the detector evaluates the accumulated Context Pool.
1. Does the next neighborhood (Sentence 4+5) share structural continuity with the first? 
2. Evidence check: No shared entities, no coreference, no dependency chains, new actors ("Marketing").
3. **Result:** A structural boundary is detected. 
4. **Promotion:** The first Context Pool is promoted to a **Section**. The second Context Pool is promoted to a new **Section**.
*(This same logic applies at the Document level: traversing Sections to find when a document's structural continuity breaks and closes).*

#### Phase C: The AMT Traversal (Semantic Condensation)
Only after Documents and Sections are established does the AMT begin. The AMT now traverses the graph, but it uses the structural boundaries as natural limits so it doesn't accidentally merge unrelated projects.

**AMT Stage 1: Candidate Meaning Formation**
The AMT visits the grammar neighborhood for Section 1 (Sentences 1-3). 
1. Context Pool starts accumulating the local grammar subgraph: We -> build -> service -> issues -> JWT -> expire -> 30 minutes.
2. The traversal collapses this connected action chain into a **Candidate Semantic Unit** (e.g., "build authentication service with JWT lifecycle").
3. No Intent exists yet. It is just a compressed representation of the grammar neighborhood.

**AMT Stage 2: Semantic Boundary Evaluation (Preventing Merges)**
The AMT now evaluates this Candidate Semantic Unit against other neighborhoods in the graph.
1. Traversal asks: Does this candidate belong to the same semantic neighborhood as the "Website Redesign" candidate?
2. Traversal checks: Shared entities? No. Shared ownership? No. Dependency edges? No. Methodology alignment? No.
3. **Result:** Independent neighborhoods confirmed. The system avoids the catastrophic error of merging "Build Authentication" and "Build Compiler" just because they share the verb "build".

**AMT Stage 3: Intent Promotion and Expansion (Owning the Knowledge)**
Only after sufficient evidence is accumulated does the AMT promote the candidate.
1. **Promotion:** The Candidate Semantic Unit becomes an **AMT Intent Node** (e.g., "Implement Authentication Service").
2. **Evidence Tying:** The Intent Node is linked back to the exact grammar nodes, coreference chains, and structural nodes that justified its creation. The AMT now *owns* this knowledge.
3. **Methodology Application:** The AMT applies stored methodologies (e.g., "Security Design"). 
4. **Expansion (Zero-Shot Traversal):** The AMT traverses the methodology's expected neighbors. "What normally exists around Authentication?" -> MFA, Password Hashing.
5. The AMT checks the graph: Are these present? No. 
6. **Branch Creation:** The AMT creates child branches under the Intent, expanding the meaning and filling gaps, grounded in the methodology but tied back to the structural graph.

---

### Part 3: Reviewing the Code Context Alignment

Now, let us review how this corrected conceptual architecture aligns with the two files you provided (`orchestrator/mod.rs` and `modalities/text/main.rs`) without implementing changes, just observing alignment.

**1. `modalities/text/main.rs` (The Grammar Graph Provider)**
*   **What it has:** `SentenceNode`, `GrammarNode` (with a massive, correct enum covering all grammar roles), `ParagraphNode`, `SectionNode`, `DocumentNode`. It has `CrossSentenceRelationship` and `CoreferenceChain`.
*   **Alignment with the vision:** This file is perfectly positioned to provide the exact Grammar Graph required for Stage 1/Stage 2 traversal. 
*   **The Gap (Conceptual):** The file currently relies on `chunk_text` and processes things per-chunk. The vision requires the grammar extraction to output a unified graph where cross-chunk coreference and dependency edges are explicit, so the traversal can cross chunk boundaries seamlessly. The structs (`CoreferenceChain`, `CrossSentenceRelationship`) exist, but the processing flow is still heavily chunk-biased. The transition from "ChunkGraph" to a unified "TextGraph" is the bridge that needs crossing.

**2. `orchestrator/mod.rs` (The AMT Builder)**
*   **What it has:** `build_amt_layer_by_layer` which loops over chunks, asks the LLM to extract intents from chunk text, and then groups them.
*   **Alignment with the vision:** This is the area requiring the most conceptual shift. The current orchestrator asks the LLM to read text chunks and pull intents out of thin air. 
*   **The Gap (Conceptual):** According to the finalized architecture, the orchestrator should *not* be passing chunk text to the LLM and asking for intents. It should be passing the grammar graph neighborhood (the Context Pool) to the LLM and asking it to condense the established relationships into a Candidate Semantic Unit, and then evaluating that unit against other neighborhoods. The AMT builder currently skips Stage 1 (Context Pool formation via graph edges) and jumps straight to semantic extraction.

### Part 4: The Final, Un-Dropped Synthesis

To summarize everything without dropping a single insight:

1.  **The Graph is the foundation.** Grammar extraction (via Path 1 or Path 2) produces a rich graph of grammar nodes, coreference, dependencies, and structural anchors.
2.  **No semantics exist initially.** The graph is purely syntactic and structural.
3.  **Document and Section detection happen FIRST.** They use a universal Stage 1/Stage 2 pattern. Stage 1 builds a Context Pool by walking grammar/coreference edges until the frontier weakens. Stage 2 evaluates structural continuity to promote boundaries. They operate entirely independently of the AMT.
4.  **The AMT operates AFTER structural segmentation.** It does not ask "what is the intent?" upfront. 
5.  **AMT Stage 1:** It traverses a structural neighborhood, accumulating grammar nodes and relationships into a Context Pool, and condenses them into a Candidate Semantic Unit.
6.  **AMT Stage 2:** It evaluates this Candidate against other neighborhoods using evidence of continuity (shared entities, dependencies), strictly preventing unrelated projects from merging just because they share similar words.
7.  **AMT Stage 3:** Only after evidence is verified does it promote the Candidate to an Intent.
8.  **AMT Owns the Knowledge:** Once promoted, the Intent becomes a tree node. It links back to the grammar evidence. Methodologies are applied to expand the tree (adding branches for missing requirements). The AMT owns this tree and the knowledge within it.

**What was correct in the AI's final response:** The AI finally visualized the "Context Pool" growing by walking grammar edges (`Subject -> Verb -> Object -> Reference Edge -> Next Sentence`), and it correctly identified the "Semantic Frontier" as the stopping point. 

**What was incorrect in the AI's final response:** It still tried to act like the AMT was "just an algorithm" and tried to introduce a "State Layer" instead of keeping the state within the traversal context pool itself.

This review captures every correction, every frustration, and every architectural mandate you provided. The picture is now unified: structural segmentation first, context pool accumulation second, evidence-based boundary evaluation third, and finally, AMT ownership and expansion of the promoted knowledge.

now this is my insights based on this repsonse 

Exactly,

now

        let sentences: Vec<&str> = cleaned_text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let sentence_count = sentences.len();
        let paragraph_count = cleaned_text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .count();

we do not need it because we are obtaining this from our chunk processing, now we are breaking this down from our prior construction because we need to now fully work on each function 1x1 on what is truly missing from these new insights and these prior ones we had before as well.



## The Processing Flow


│                    TEXT MODALITY PROCESSING FLOW                            │


│  PHASE 1: PER-CHUNK PROCESSING                                              │

│  For each chunk:
│    1. CleanChunk (grammar correction, normalization via zero-shot LLM)

now here is what we are going to do we are going to split this into two pathways from here on out, because yes for all we process in chunks to clean but not from here specifically in the clean chunk we will now have to paths and this is for testing and benchmark purposes

1. as is where we have the zero-shot LLM input the full chunk and output the full chunk corrected

2. we break this down further so for each chunk we 1. detect all sentences and we are looping through this taking into account how we do this in async fn build_amt_layer_by_layer( where we have something like

        ALREADY KNOWN INTENTS (do NOT repeat these):
        {}

but for sentences, so we are capturing first per chunk all sentences per chunk until stable and that is per chunk so of course our

        ALREADY KNOWN SENTENCES (do NOT repeat these):
        {}

always starts out blank, we now also want to capture the sentence order as well so 1:first sentence 2: second sentence

now here is something we need to take into account we will be detecing sentences while doing grammar correction at the same time so we will have the zero shot explicictly provding identified sentences which may or may not be gramtratically correct IE it will provide us the before the text span of the original sentence as is wether gramatticaly correct or not in which we are working on and provide the corrected sentence as well if it needs correction.

so we will had as it adds

        ALREADY KNOWN SENTENCES (do NOT repeat these):
        {1: {original sentence: "", text_span_start: x, text_span_ends: x, corrected_sentence, ""}, }

now here is the thing where we have to be careful here as well we obvious will have a 5x stability function for this because now for each item we need to confirm its correct but say with a model like Bitnet

the 4096-token limit is shared

For a standard causal LLM like BitNet, the context window is shared between the prompt (system + user + any history) and the generated output.

Conceptually:

4096 tokens total

+---------------------------+----------------------+
| Input / Prompt            | Generated Output     |
+---------------------------+----------------------+

So if your prompt consumes:

System prompt: ~40 tokens
Instructions: ~120 tokens
Chunk: ~1800 tokens

Your input is roughly:

40
+120
+1800
-----
1960 input tokens

That leaves approximately:

4096 - 1960 = 2136 tokens

available before the model hits the context limit (subject to the -n/max_tokens limit you set).

so we want to understand as we do this approach the input may exceed the limit or not allow for a complete output this is why we work at 1/4 of the token limit for a LLM per chunk. SO what can we and what should we do here for path 2 ?
The good thing is that for path 1 we have more then enough room for it to output everything back correcte but now for path 2 we have to take this into account carefully

and here is how I see it. If we take a look at it as chunk state we when we start it has an empty known sentences array right, we are at the original 1/4 token limit we established eh taking into account the system prompt and instructions, we receive the response we need to add this also to used tokens we are tracking over time in the orchestrator state but going back to what I am saying we start with an empty array, we pass in our initial chunk state with the empty array from here we receive the formated response the structured JSON response for us to analyze, what is this reponse going to provide? a list of sentences or 1 sentnces either or. so now we are in the processing phase of response from the chunk state we here are processing each input IE each sentence 1x1 but how should it provide this back grammarly corrected? so should we say provide a list or 1 sentences iedntified in this chunk in order? I mean it will be more accurate if we wrked on it 1x1 we tell it to provide the first sentence and we can ensure if its working one by one that it can provide a more accurate answer, provide the first sentence in a JSON so that we maintain a ordered list since SLM's/LLM's arent always good with numberings/orderings this will ensure we maintain consistency. we let it know this may or may not be gramatically correct in this response provide the if the list is empty the first sentence in this chunk and it can provide us {1: {original sentence: "", text_span_start: x, text_span_ends: x, corrected_sentence, ""}, } what do we do from here ? from here then we individually now do the 5x stability verification and we are doing it per item in the processing state, now this isn't returning fully to the seen setences list in full right away, we are keeping the original sentence even if gramtally correct or not becauseifwe add everything else then its just unncesary clutter it just needs to know what we have processed everything else the corrected sentences etc with the text span and end we are maintain order listing as well here to now add this to our copy graph or nodes, because now what are we doing with sentences? we are turning them or capturing them into nodes right away once we fully validate them 1x1

SentenceNode {
  node_id: 12345,
  node_type: Sentence,
  content: "John submitted the report yesterday.",

  // POSITION (populated during Phase 1 chunk processing)
  position: {
    start_byte: 4523,
    end_byte: 4558,
    start_line: 23,
    end_line: 23
  },

  // CHUNK ANCHOR (stored in ChunkGraph — enables cross-model reconstruction)
  chunk_id: 3,
  chunk_offset: 145,

  // VERSION
  version: 1,
  last_modified_step: None
}

this we are capturing it all to transfer to our graphs once we process all chunks we need to capture these correctly,

so now you see at first we take in the chunk whole with the seen sentences array this is the starting state as we loop per graph for each graph we then in order identify all sentences 1x1 starting with identifying the first sentences, then we transition to a different state per response where now we we look solely at the chunk and the response itself to verify if correct 5x we also truly want to verirfy as well 5x agaisnt seen list if truly not there or captured to ensure no duplicates, this isn't eauting up or stacking up within our token limits we then now proceess each sentence to offload properties to SentenceNode and capture only each sentence and order original back to the array for the chunk as we continue looping over that same chunk until complete to move onto the next chunk we then review the chunk against the seen sentences to validate 5x that there are no new sentences to process really on each loop per chunk we start with is there any sentences inthis cunk which we have not captured in our seen sentences array etc andthis is our validation to have a no after 5x let's us know we have completed all but not just completed all but completed all in order because we are also again keeping track of order we say not just is there any but our seen sentences array has x items 0 or 1 or 2 or 3 is there any or if we haven't seen any what is the first or following sentences from the last one we capture and if any we display the last one this controls context explosion.


okay so awesome we have here identified and captured identifying sentences correctly per node in path 2 but now here is the thing alongside this we first want to keep track of what section we are in

because we have to take into account of

## Section & Document Detection

### Why Section Detection Matters

Section detection is the only mechanism that allows the system to distinguish:
- Raw text (user's prompt, conversational input, free-form request)
- Embedded documents (guides, articles, reports, specifications)
- Mixed content (user request containing an attached or inline document)

Intent detection belongs to the AMT. Section detection belongs to the text modality.
The text modality identifies WHAT the content is structurally; the AMT determines
WHY the user provided it and what to do with it.

so we first have CleanChunks and in the path 2 we need to take into account what section we are in first while we find sentences and correct grammar.

SectionStateTracking (cross-chunk state machine, format-agnostic)

│  DocumentNode                                                               │
│    node_type: Document                                                      │
│    content: summary (word count, type, etc.)                               │
│    properties: { document_type, word_count, sentence_count, paragraph_count,│
│                  readability_score, title, source, role }                   │
│    position: { start_byte, end_byte }                                       │
│                                                                             │
│  SectionNode                                                                │
│    node_type: Section                                                       │
│    content: section title                                                   │
│    properties: { level (1-6), formatting_pattern, parent_section }          │
│    position: { start_byte, end_byte, start_line }


so we are getting rid of in full of the     /// Extract document structure
    fn extract_structure(&self, text: &str) -> DocumentStructure {

as this was rule based and going to utilize a zero-shot approach in place within the chunk processing itself

### Section State Machine

State tracking runs across chunk boundaries. The state is carried from chunk N to
chunk N+1 to handle sections that span multiple chunks.

```
SECTION TRACKING STATE MACHINE

State: IDLE
  Trigger: Any structured heading pattern detected
           (regardless of format — zero-shot determines the format)
  Transition → IN_SECTION

State: IN_SECTION
  Trigger: Another heading of same or higher level detected
  Transition → NEW_SECTION (close current, open new)

  Trigger: Sub-heading detected (lower level than current)
  Transition → IN_SUBSECTION (nest under current)

  Trigger: Formatting pattern breaks (N paragraphs with no heading,
           style shift detected by zero-shot, narrative voice changes)
  Transition → DOCUMENT_BREAK or IDLE

State: DOCUMENT_BREAK
  Trigger: New structural pattern begins
  Transition → IN_SECTION (new document context)

CROSS-CHUNK CONTINUATION:

  Chunk N ends mid-section:
    State carries:
      - section_id (persistent across chunks)
      - section_level (1-6 or detected equivalent)
      - formatting_pattern (discovered by zero-shot — any format)
      - current_document_id

  Chunk N+1 receives continuation context:
    - Inherits open section from chunk N
    - May close it or extend it
    - May detect a break → new document
```

**Format-agnostic detection:** The system does not hardcode markdown or any specific
format. Zero-shot LLM determines the formatting pattern from the content itself.
Patterns include (but are not limited to): markdown headers, numbered sections,
all-caps headings, underlined headings, Roman numerals, bold lead lines,
indented outlines, any consistent structural pattern the LLM identifies.

### Zero-Shot Section Detection Prompt

```text
You are analyzing text to detect document structure and section boundaries.
The formatting style is UNKNOWN — do not assume any specific format (not just
markdown). Detect whatever structural patterns actually appear in the text.

CHUNK CONTEXT:
- This is chunk {chunk_index} of {total_chunks}
- Previous section state: {previous_section_state}
- Formatting pattern being tracked: {formatting_pattern or "not yet detected"}

CHUNK TEXT:
{chunk_text}

TASK: Identify sections and document boundaries.

A "section" is any coherent block that begins with a recognizable heading or
structural marker — regardless of format. This includes but is not limited to:
- Markdown headers (# ## ###)
- Numbered sections (1. 2. 2.1)
- ALL-CAPS headings (INTRODUCTION, METHODS)
- Underlined or decorated headings (===, ---)
- Bold lead lines followed by body text
- Any other consistent structural marker you detect

A "document break" occurs when:
- The formatting pattern changes significantly
- The writing style or voice shifts substantially
- Structural markers stop appearing after a consistent run

Return ONLY valid JSON:
{
  "section_events": [
    {
      "type": "section_start|section_continue|section_end|document_break",
      "position": <character_offset_in_chunk>,
      "section_title": "title if section_start, null otherwise",
      "section_level": <1-6 or null>,
      "formatting_pattern": "description of detected pattern or null"
    }
  ],
  "current_section_context": {
    "section_id": "unique identifier for tracking across chunks",
    "level": <1-6 or null>,
    "title": "current section title or null",
    "formatting_pattern": "detected pattern description"
  },
  "document_boundary_detected": false,
  "document_boundary_reason": "reason or null"
}
```

### Zero-Shot Document Identification Prompt

Called when a section boundary or document break is detected, to classify
whether the content constitutes a standalone document.

```text
You are identifying whether a block of text constitutes a standalone document
within a larger corpus.

A "document" is a coherent, self-contained unit characterized by:
- Unified topic or theme throughout
- Consistent formatting pattern (whatever format)
- Clear structural organization (sections, paragraphs, or both)
- Neutral or third-person narrative voice (for reference/informational documents)
- OR: a sustained narrative voice (for stories, guides, essays)

NOT a document:
- First-person conversational requests ("I want", "Please help me")
- Imperative instructions addressed to the AI ("Fix this", "Create a")
- Questions without context ("How do I", "What is")
- Short mixed-intent fragments

IMPORTANT: Documents sometimes contain user intent mixed in. The presence of
some conversational sentences does not disqualify a document — look for the
dominant structure and voice.

TEXT TO ANALYZE:
{text_span}

Return ONLY valid JSON:
{
  "is_document": true|false,
  "document_type": "article|guide|story|report|specification|reference|code_doc|data|unknown|null",
  "title": "extracted title if detectable, null otherwise",
  "confidence": 0.0-1.0,
  "dominant_structure": "description of what makes this a document or not",
  "mixed_intent_detected": true|false,
  "mixed_intent_description": "what conversational/intent elements appear, or null"
}
```

### Document vs Raw Text Classification

After section state detection, content falls into two categories:

**Document** — minimum structure: one section OR one paragraph with consistent
formatting and unified topic. Documents are broken down into their own
Document → Section → Paragraph → Sentence node hierarchy.

**Raw text** — minimum structure: one sentence (or even a fragment). Raw text
from the prompt itself is still formalized to complete sentences during
cleaning (CleanChunk), but it does not require section or document structure.
It links directly to the root of the text modality graph.

**Mixed content** — both exist simultaneously. The section state machine handles
this by detecting document starts and stops within the same chunk stream. A
user request followed by an inline document results in:
- Raw text nodes for the user's words (linked to prompt root)
- A DocumentNode + full hierarchy for the document portion

Intent identification (what the user wants, what's primary vs supplementary)
is NOT handled here — that belongs to the AMT building phase in the
orchestrator, via zero-shot role detection and the FileGraphRole classification.

now here is the thing to keep in mind, we for now will yes be tracking Sections, but we will not be tracking documents, in a finalized version we would be assigning Paragraph nodes to sections if within one but for now no we are just tracking sections because for 1. to truly identify or track a document we need full understanding of all the contents within it to fully accurately identify when a document breaks and for this we need to have the whole graph completed or a majority at least up to the grammar nodes to not have context explosion when reviewing a document break as we need to take into account all reltionships perataining to it to identify if it is a true document break, as well as for the section iedntifier and start we still need some refinements afterwards but it doesn't mean that I still don't want to have it included yet so we will, for path 2 per chunk we are keeping track of sections as well as identifying sentences to correct the grammar for each sentence


so now while we are touching on this we identify we while looking at each chunk we are looking at sentences and also now looking at sections

but now we also are looking at alongisde the two really more alongiside of looking for sentences we are also looking for

│  MODALITY DETECTION                                                │
│  ───────────────────────────                                                │                                                                                        │
│       a. Detect embedded modalities (code blocks, math, etc.)    │
│       b. Mark spans as "true_text" or detected modality name                │
│       c. 5x consecutive stable loop (detect_modalities_stable)              │
│    2. Create ModalityReferenceNode for non-text spans                       │
│    3. Store span_start / span_end (byte offsets, NO content_snippet)        │

│                                                                             │
│  MODALITY REFERENCE NODES                                                   │
│  ─────────────────────────                                                  │
│                                                                             │
│  ModalityReferenceNode (ChunkModalityDetection in code)                     │
│    modality: the detected modality name (from pipeline registry)            │
│    span_start: char offset within sentence/chunk                            │
│    span_end: char offset within sentence/chunk                              │
│    intent_reference: how this modality is referenced (describes/contains)   │
│    chunk_index: which chunk this was detected in                            │
│    NOTE: no content_snippet — retrieve dynamically: &text[span_start..end] │
│

**Modality spans**: Embedded code, math, etc. with byte positions so while we are looking at graphs and this it isimportant to note that we may have embebed modatliyies inside sentences, IE a simple formula or math formula etc and we tie this to the sentence nde if so if nt we tie it to the paragaph node not the chunk id because we are't using chunks anymore we are moving to full graphs so we tie it to either the setence node or parapgrah node

now the paragraph node

so apart from looking per chunk at sections/sentences/modality references we are also looking at


│  ParagraphNode                                                              │
│    node_type: Paragraph                                                     │
│    properties: { sentence_count, parent_section, document_id }             │
│    position: { start_byte, end_byte }

this we are doing similar to sections because this a paragraph per chunk can be cut off and we identify allparapgrahs ina  chunk if any but then if the last parapgrah if more then one or if any not found we look at it with the next chunk there really should be no parapgrah that is two chunks long realistically if we follow grammar rules but either way this is to ensure we acpture if a parapgrah most scenarios will be the last paragraph in a chunk may be cut off so we want to identify the parapgrah start and in the next chunk if end isn't found we with the paragraph start identified look for the parapgrah end confirm if true or not or find a new one etc etc

so again for now we can tie sentences to pargraphs nodes as well as for modality refrences we will be tying it either to paragraphs or setence node while forthe parapgrah node we can tie it to the section node for now even though section node may not be finalized or full refined which is ok

ah then see as you can see we do just as we are taking into account for sections into next chunk when we identify a section start and end isn't in the chunk or doesn't brak and as we do for the paragraph we also need to do for sentences and modality references because IE the last setences may be broken or cut off and it means it may be in the next chunk so we want to capture all this correctly as we do with sections take all into account without omission and without dropping anything ensuring alingment

now for the grammar extraction,

async fn extract_grammar_relationships_from_text(

we want to keep in mind this is actually not per chunk this will be done at the end after all chunks are processed and this now won't be exracting gramar-relationships from text but we will be extracting it from graphs

### Phase 3: Grammar Extraction

Grammar extraction runs on the full graph after all chunks are processed and
the complete sentence/paragraph/section hierarchy is assembled. This ensures
cross-sentence relationships can form across any sentence in the corpus,
not just within a single chunk.

#### Per-Sentence Grammar Extraction Prompt

```text
You are extracting grammatical structure from a single sentence.

SENTENCE: "{sentence_text}"

TASK: Identify the grammatical components and their relationships.

Return ONLY valid JSON:
{
  "subject": {
    "text": "the subject noun phrase",
    "position_start": 0,
    "position_end": 10,
    "entity_type_hint": "Person|Organization|Location|Document|Time|Concept|Unknown"
  },
  "verb": {
    "text": "the main verb",
    "position_start": 11,
    "position_end": 20,
    "verb_type": "action|linking|helping",
    "tense": "past|present|future|unknown",
    "negated": false
  },
  "object": {
    "text": "the object noun phrase or null",
    "position_start": 21,
    "position_end": 35,
    "entity_type_hint": "..."
  },
  "modifiers": [
    {
      "text": "modifier text",
      "position_start": 36,
      "position_end": 45,
      "modifier_type": "temporal|adjectival|adverbial|prepositional",
      "modifies": "subject|verb|object"
    }
  ],
  "sentence_type": "declarative|interrogative|imperative|fragment",
  "edge_type": ""Performs|Affects|Implies|Contradicts|Elaborates|Summarizes|Supports|TemporalPrecedes|TemporalFollows|CausedBy|Enables|Prevents|PartOf|HasPart|FunctionalRole|InstanceOf|HasInstance|SimilarTo|DerivedFrom|VersionOf..","
}
```

#### Cross-Sentence Relationship Prompt (Pairwise Comparison)

Run after all per-sentence grammar extraction. The pairwise comparison produces
the relationship depth of the graph — this is where the most cross-sentence
connections form.

```text
You are analyzing relationships between sentences in a text corpus.

SENTENCES:
{sentence_list_with_indices_and_grammar_summaries}

TASK: Identify all meaningful relationships between these sentences.

Relationship types:
- Elaborates:        S2 provides more detail about S1
- Causes:            S1's action or state leads to S2
- Enables:           S1 makes S2 possible
- Prevents:          S1 prevents S2
- Contradicts:       S2 contradicts or negates S1
- Exemplifies:       S2 is a concrete example of S1
- Summarizes:        S2 is a summary of S1 (or vice versa)
- TemporalPrecedes:  S1 happens before S2 in described time
- Coreference:       S2 contains a pronoun or reference to an entity in S1
- PartOf:            S2 is a component of the whole described in S1
- SimilarTo:         S1 and S2 describe closely related concepts

Return ONLY valid JSON:
{
  "relationships": [
    {
      "from_sentence_index": 0,
      "to_sentence_index": 1,
      "relationship_type": "Elaborates|Causes|Enables|...",
      "evidence": "brief quote showing the relationship"
    }
  ],
  "coreference_chains": [
    {
      "canonical_entity": "John Smith",
      "mentions": [
        {"sentence_index": 0, "text": "John Smith", "grammar_role": "subject"},
        {"sentence_index": 2, "text": "He", "grammar_role": "subject"},
        {"sentence_index": 5, "text": "the manager", "grammar_role": "object"}
      ]
    }
  ]
}
```

but now I also want you to take this into account

The tree is.

For example,

> I want to try on a suit I saw in a shop that's across the street from the hotel.

is not really

```text
I → want → to → try → on → a → suit ...
```

It is closer to

```text
Sentence
│
└── Predicate (want)
    │
    ├── Subject
    │   └── I
    │
    ├── Infinitive Clause
    │   └── try
    │       │
    │       ├── Object
    │       │   └── suit
    │       │
    │       └── Relative Clause
    │           └── saw
    │               │
    │               ├── Subject
    │               │   └── I
    │               │
    │               ├── Object
    │               │   └── suit
    │               │
    │               └── Location
    │                   └── shop
    │                       │
    │                       └── Relative Clause
    │                           └── across
    │                               │
    │                               ├── street
    │                               └── hotel
```

Notice something important.

**Nothing is duplicated.**

Everything simply **attaches to its grammatical parent.**

we have our grammar nodes per sentence

│  │ Grammar nodes (per sentence):                                        │   │
│  │   Subject, Verb, Object, Modifier                                   │   │
│  │                                                                      │   │
│  │ Derived nodes (from graph traversal):                                │   │
│  │   EntityNode (Person, Org, Location, Concept, etc.)                  │   │
│  │   CoreferenceChain (John → He → the manager)

├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GRAMMAR NODES (per sentence, derived via zero-shot)                        │
│  ────────────────────────────────────────────────────                       │
│                                                                             │
│  Subject                                                                    │
│    text: the subject noun phrase                                            │
│    position_in_sentence: { start, end }                                     │
│    entity_type_hint: Person|Organization|Location|Concept|Unknown           │
│                                                                             │
│  Verb                                                                       │
│    text: the main verb                                                      │
│    verb_type: action|linking|helping (VerbType enum in code)                │
│    tense: past|present|future                                               │
│    negated: bool                                                            │
│                                                                             │
│  Object                                                                     │
│    text: the object noun phrase                                             │
│    position_in_sentence: { start, end }                                     │
│    entity_type_hint                                                         │
│                                                                             │
│  Modifier                                                                   │
│    text: the modifier text                                                  │
│    modifier_type: temporal|adjectival|adverbial|prepositional               │
│    modifies: subject|verb|object                                            │
│                                                                             │
│  ChunkGrammarRelationship (code struct) captures all of the above:          │
│    from_text, to_text, edge_type, tense, negated, verb, verb_type,          │
│    subject, object, source_sentence_start, source_sentence_end              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DERIVED NODES (from graph traversal after Phase 3)                         │
│  ──────────────────────────────────────────────────                         │
│                                                                             │
│  EntityNode                                                                 │
│    node_type: Entity                                                        │
│    text: canonical entity text                                              │
│    entity_type: Person|Organization|Location|Date|Product|Event|           │
│                 Technology|Concept|Custom (EntityType enum in code)         │
│    confidence: f32                                                          │
│    mentions: [grammar_node_ids across sentences]                            │
│    coreference_chain_id: Option<u64>                                        │
│                                                                             │
│  CoreferenceChain                                                           │
│    chain_id                                                                 │
│    mentions: [node_ids in order of appearance]                              │
│    canonical_form: e.g. "John Smith"                                        │
│                                                                             │
│  TopicNode                                                                  │
│    node_type: Topic                                                         │
│    text: topic name                                                         │
│    relevance: f32 (derived from relationship traversal coverage)            │
│                                                                             │
│  KeywordNode                                                                │
│    node_type: Keyword                                                       │
│    text: keyword or phrase                                                  │
│    relevance: f32                                                           │
│    (derived from verb/object traversal, NOT separate extraction pass)


now remember

│  ChunkGrammarRelationship (code struct) captures all of the above:          │
│    from_text, to_text, edge_type, tense, negated, verb, verb_type,          │
│    subject, object, source_sentence_start, source_sentence_end

this now isn't chunkgrammarrelationships but tied to the sentencenode itself


now we again want to ensure we are capturing all it seems like we are missing stuff not saying this has everything we are missing we are already capturing some likes entence but we now nee to ensure we have the full inclusion for all

    Sentence,

    Clause,

    Phrase,

    Subject,

    Predicate,

    Verb,

    DirectObject,

    IndirectObject,

    Complement,

    Modifier,

    Determiner,

    Adjective,

    Adverb,

    Preposition,

    PrepositionalPhrase,

    RelativeClause,

    Conjunction,

    Pronoun,

    Noun,

    ...
}
```


```text
Sentence
│
└── Predicate
    │
    ├── Subject
    │
    ├── Verb
    │
    ├── Object
    │
    ├── Modifier
    │
    └── Clause
        │
        ├── Verb
        ├── Subject
        └── Object
```

Everything has **one parent**.

Everything belongs somewhere.

There is no clutter.

```text
Paragraph

├── Sentence Tree

├── Sentence Tree

├── Sentence Tree

└── Sentence Tree
```

Each sentence owns its grammar.

---

## Sections

Exactly the same.

```text
Document

└── Section

    ├── Paragraph

    │   ├── Sentence Tree

    │   ├── Sentence Tree

    │   └── Sentence Tree

    │

    └── Paragraph

        ├── Sentence Tree

        └── Sentence Tree
```

Grammar remains local to each sentence and we create relationships between sentences and parapgrahs to sections/documents


```text
Document Layer
────────────────────────────────────

Document
└── Section
    └── Paragraph
        └── Sentence


Grammar Layer (one tree per sentence)
────────────────────────────────────

Sentence
│
├── Clause
│   ├── Subject
│   ├── Predicate
│   │   ├── Verb
│   │   ├── Object
│   │   ├── Complement
│   │   └── Modifiers
│   └── Nested Clauses
└── Punctuation (optional)
```


    //====================================================
    // ROOTS
    //====================================================

    Sentence,
    MainClause,
    SubordinateClause,
    RelativeClause,
    ComplementClause,
    AdverbialClause,
    ConditionalClause,
    ComparativeClause,
    CoordinateClause,
    ParentheticalClause,
    EllipticalClause,
    QuotedClause,

    //====================================================
    // PHRASES
    //====================================================

    Phrase,

    NounPhrase,
    VerbPhrase,
    AdjectivePhrase,
    AdverbPhrase,
    PrepositionalPhrase,
    InfinitivePhrase,
    ParticipialPhrase,
    GerundPhrase,
    AbsolutePhrase,
    AppositivePhrase,
    CoordinatePhrase,
    ParentheticalPhrase,

    //====================================================
    // PREDICATE STRUCTURE
    //====================================================

    Predicate,

    Verb,

    MainVerb,
    AuxiliaryVerb,
    ModalVerb,
    LinkingVerb,
    HelpingVerb,
    PhrasalVerb,
    Copula,

    PredicateComplement,
    PredicateNominative,
    PredicateAdjective,

    //====================================================
    // ARGUMENTS
    //====================================================

    Subject,
    ImpliedSubject,
    ExpletiveSubject,

    DirectObject,
    IndirectObject,

    ObjectComplement,

    SubjectComplement,

    Complement,

    //====================================================
    // NOUNS
    //====================================================

    Noun,

    CommonNoun,
    ProperNoun,
    CollectiveNoun,
    MassNoun,
    CountNoun,
    ConcreteNoun,
    AbstractNoun,
    CompoundNoun,

    //====================================================
    // PRONOUNS
    //====================================================

    Pronoun,

    PersonalPronoun,
    ReflexivePronoun,
    ReciprocalPronoun,
    RelativePronoun,
    DemonstrativePronoun,
    InterrogativePronoun,
    IndefinitePronoun,
    PossessivePronoun,

    //====================================================
    // DETERMINERS
    //====================================================

    Determiner,

    Article,
    Demonstrative,
    PossessiveDeterminer,
    Quantifier,
    Numeral,
    DistributiveDeterminer,

    //====================================================
    // MODIFIERS
    //====================================================

    Modifier,

    Adjective,
    AttributiveAdjective,
    PredicativeAdjective,
    ComparativeAdjective,
    SuperlativeAdjective,

    Adverb,
    AdverbOfTime,
    AdverbOfPlace,
    AdverbOfManner,
    AdverbOfDegree,
    AdverbOfFrequency,
    SentenceAdverb,

    //====================================================
    // PREPOSITIONS
    //====================================================

    Preposition,

    SimplePreposition,
    CompoundPreposition,
    PhrasalPreposition,

    //====================================================
    // CONJUNCTIONS
    //====================================================

    Conjunction,

    CoordinatingConjunction,
    SubordinatingConjunction,
    CorrelativeConjunction,

    //====================================================
    // DEPENDENTS
    //====================================================

    RelativePronounClause,

    RelativeModifier,

    AdjectivalModifier,
    AdverbialModifier,

    NominalModifier,

    DeterminerModifier,

    PossessiveModifier,

    NumericModifier,

    QuantifierModifier,

    Apposition,

    Vocative,

    Parenthetical,

    //====================================================
    // PREPOSITIONAL COMPONENTS
    //====================================================

    PrepositionalObject,

    ObjectOfPreposition,

    ComplementOfPreposition,

    //====================================================
    // VERBAL COMPONENTS
    //====================================================

    InfinitiveMarker,

    Infinitive,

    Gerund,

    PresentParticiple,

    PastParticiple,

    //====================================================
    // CLAUSE FUNCTIONS
    //====================================================

    ClauseSubject,

    ClausePredicate,

    ClauseObject,

    ClauseComplement,

    ClauseModifier,

    //====================================================
    // NEGATION
    //====================================================

    Negation,

    NegativeMarker,

    NegativeDeterminer,

    NegativePronoun,

    //====================================================
    // QUESTIONS
    //====================================================

    QuestionMarker,

    InterrogativeWord,

    TagQuestion,

    WhPhrase,

    //====================================================
    // COMPARISON
    //====================================================

    Comparison,

    ComparativeMarker,

    SuperlativeMarker,

    EqualityMarker,

    //====================================================
    // COORDINATION
    //====================================================

    Coordination,

    Coordinator,

    CoordinatedElement,

    //====================================================
    // AGREEMENT
    //====================================================

    AgreementMarker,

    CaseMarker,

    NumberMarker,

    GenderMarker,

    PersonMarker,

    //====================================================
    // PUNCTUATION
    //====================================================

    Punctuation,

    Comma,

    Period,

    Colon,

    Semicolon,

    Dash,

    Hyphen,

    Parenthesis,

    Quotation,

    Apostrophe,

    Ellipsis,

    Exclamation,

    QuestionMark,

    //====================================================
    // TERMINALS
    //====================================================

    Token,

    Word,

    Symbol,

    Number,

    Letter,

    Unknown,
}
```


# Grammar Layer

Now imagine one sentence

```text
The dog quickly chased the cat through the garden.
```

The grammar becomes

```text
Sentence
│
└── MainClause
    │
    ├── Subject
    │   │
    │   └── NounPhrase
    │       │
    │       ├── Determiner
    │       │     "The"
    │       │
    │       └── Noun
    │             "dog"
    │
    └── Predicate
        │
        ├── Verb
        │     "chased"
        │
        ├── Adverb
        │     "quickly"
        │
        ├── DirectObject
        │   │
        │   └── NounPhrase
        │       │
        │       ├── Determiner
        │       └── Noun
        │
        └── PrepositionalPhrase
            │
            ├── Preposition
            │      "through"
            │
            └── ObjectOfPreposition
                   "garden"
```

Notice something.

Every node has exactly

```text
Parent

Children
```

we can now also look at organizing properties

# Properties

Properties contain grammar information.

```rust
GrammarProperties {

    tense,

    aspect,

    mood,

    voice,

    person,

    number,

    gender,

    case,

    definiteness,

    comparison,

    polarity,

    subtype,

}
```

Properties answer

> "Describe me."

Huge difference.

---

```rust
    properties: {
        tense: Past,
        aspect: Perfect,
        mood: Indicative,
        voice: Active,
    }
}
```

Likewise,

```rust
    properties: {
        number: Plural,
        case: Possessive,
        gender: Unknown,
    }
}
```

and

```rust
    properties: {
        subtype: Personal,
        person: First,
        number: Singular,
        case: Subjective,
    }
}
```


The reason between the two paths in which we are studying is because while 1 does less work quality also depends on the SLM/LLM quality and while also larger models it may allow for less work and a reasonable quality path 2 still ensures higher quality by thrughlyreviewing each thus yes it esnures more work which can be beneficial for models it may benefit smaller models more highly while it can also outbeat larger models in accuracy so we will be benchmarking both paths thoroughly for full comparisons in speeds/quality and work/loops called for completion.


## The Chunking Boundary

Chunking happens ONCE on initial intake. After chunks are processed and nodes are created,
all subsequent work is graph-based. Chunks exist only to:

1. Fit text within any LLM context window (cross-model capability)
2. Enable parallel processing of large documents
3. Anchor byte positions for the Text Engine

The ChunkGraph for each processed chunk preserves the byte offset mapping so that any
node in the text modality graph can be found in O(1) via position lookup — enabling the
Text Engine to perform precise byte-level editing without scanning file contents.

Chunks are always to still happen once no matter which path but each path now defines how thoroughly we are now processing each chunk.


now this is what we where looking at prior now again we are breaking this down from our prior construction because we need to now fully work on each function 1x1 on what is truly missing from these new insights and these prior ones we had before as well.

so we are looking exactly at this everything from here below in order I number letter them A-Z A.1 A.2 etc as some of the contents in each may have numbering but you are to fully address in order the contents within A-Z which are refrecing what we have above and the new insights as well

A. For this we have captured Path 1 and 2 correctly.

│  PHASE 1: PER-CHUNK PROCESSING                                              │

│  For each chunk:
│    1. CleanChunk (grammar correction, normalization via zero-shot LLM)

now here is what we are going to do we are going to split this into two pathways from here on out, because yes for all we process in chunks to clean but not from here specifically in the clean chunk we will now have to paths and this is for testing and benchmark purposes

1. as is where we have the zero-shot LLM input the full chunk and output the full chunk corrected

2. we break this down further so for each chunk we 1. detect all sentences and we are looping through this taking into account how we do this in async fn build_amt_layer_by_layer( where we have something like

        ALREADY KNOWN INTENTS (do NOT repeat these):
        {}

B. We do a great job at trying to processes and obtain each sentences 1x1, here is the thing in // 3. Sentence extraction 1-by-1 with 5x stable validation, the prompt is to short, it is not specific enough                 r#"Identify the next sentence in the text chunk.
                ALREADY KNOWN SENTENCES (do NOT repeat these) this and I feel like also we do not need to provide the full known sentence list we can always work on the last worke sentence if we work in order 1x1 then we just need to provide the last sentence clarify and we should be checking if the list is empty instruct it to provide the first sentence in the chunk if not to provide the sentence following the sentence we provided it is not explaining enough its role to correct the sentences if needed etc and how to return it and why we need it as so. Now // 5x validation: ask 5 times to ensure we get the same sentence is a nice attempt be we should be more granular we should not be looping 5 times on the same prompt ask once with the propmt then we subtract the newly obtained sentence and then we obtain the last sentence it was working on and provide both and ask is this sentnce the sentence that follows this last sentence from this chunk and it should return a YES OR NO answer if yes 5 times then its validated if no 5x then its wrong it should be instructed to provide a one worded response in JSON

but for sentences, so we are capturing first per chunk all sentences per chunk until stable and that is per chunk so of course our

        ALREADY KNOWN SENTENCES (do NOT repeat these):
        {}

always starts out blank, we now also want to capture the sentence order as well so 1:first sentence 2: second sentence

now here is something we need to take into account we will be detecing sentences while doing grammar correction at the same time so we will have the zero shot explicictly provding identified sentences which may or may not be gramtratically correct IE it will provide us the before the text span of the original sentence as is wether gramatticaly correct or not in which we are working on and provide the corrected sentence as well if it needs correction.

so we will had as it adds

        ALREADY KNOWN SENTENCES (do NOT repeat these):
        {1: {original sentence: "", text_span_start: x, text_span_ends: x, corrected_sentence, ""}, }

now here is the thing where we have to be careful here as well we obvious will have a 5x stability function for this because now for each item we need to confirm its correct but say with a model like Bitnet

and here is how I see it. If we take a look at it as chunk state we when we start it has an empty known sentences array right, we are at the original 1/4 token limit we established eh taking into account the system prompt and instructions, we receive the response we need to add this also to used tokens we are tracking over time in the orchestrator state but going back to what I am saying we start with an empty array, we pass in our initial chunk state with the empty array from here we receive the formated response the structured JSON response for us to analyze, what is this reponse going to provide? a list of sentences or 1 sentnces either or. so now we are in the processing phase of response from the chunk state we here are processing each input IE each sentence 1x1 but how should it provide this back grammarly corrected? so should we say provide a list or 1 sentences iedntified in this chunk in order? I mean it will be more accurate if we wrked on it 1x1 we tell it to provide the first sentence and we can ensure if its working one by one that it can provide a more accurate answer, provide the first sentence in a JSON so that we maintain a ordered list since SLM's/LLM's arent always good with numberings/orderings this will ensure we maintain consistency. we let it know this may or may not be gramatically correct in this response provide the if the list is empty the first sentence in this chunk and it can provide us {1: {original sentence: "", text_span_start: x, text_span_ends: x, corrected_sentence, ""}, } what do we do from here ? from here then we individually now do the 5x stability verification and we are doing it per item in the processing state, now this isn't returning fully to the seen setences list in full right away, we are keeping the original sentence even if gramtally correct or not becauseifwe add everything else then its just unncesary clutter it just needs to know what we have processed everything else the corrected sentences etc with the text span and end we are maintain order listing as well here to now add this to our copy graph or nodes, because now what are we doing with sentences? we are turning them or capturing them into nodes right away once we fully validate them 1x1

SentenceNode {
  node_id: 12345,
  node_type: Sentence,
  content: "John submitted the report yesterday.",

  // POSITION (populated during Phase 1 chunk processing)
  position: {
    start_byte: 4523,
    end_byte: 4558,
    start_line: 23,
    end_line: 23
  },

  // CHUNK ANCHOR (stored in ChunkGraph — enables cross-model reconstruction)
  chunk_id: 3,
  chunk_offset: 145,

  // VERSION
  version: 1,
  last_modified_step: None
}

this we are capturing it all to transfer to our graphs once we process all chunks we need to capture these correctly,

so now you see at first we take in the chunk whole with the seen sentences array this is the starting state as we loop per graph for each graph we then in order identify all sentences 1x1 starting with identifying the first sentences, then we transition to a different state per response where now we we look solely at the chunk and the response itself to verify if correct 5x we also truly want to verirfy as well 5x agaisnt seen list if truly not there or captured to ensure no duplicates, this isn't eauting up or stacking up within our token limits we then now proceess each sentence to offload properties to SentenceNode and capture only each sentence and order original back to the array for the chunk as we continue looping over that same chunk until complete to move onto the next chunk we then review the chunk against the seen sentences to validate 5x that there are no new sentences to process really on each loop per chunk we start with is there any sentences inthis cunk which we have not captured in our seen sentences array etc andthis is our validation to have a no after 5x let's us know we have completed all but not just completed all but completed all in order because we are also again keeping track of order we say not just is there any but our seen sentences array has x items 0 or 1 or 2 or 3 is there any or if we haven't seen any what is the first or following sentences from the last one we capture and if any we display the last one this controls context explosion.

C. For the section prompt we have this practically right             r#"You are analyzing text to detect document structure and section boundaries.
            The formatting style is UNKNOWN — do not assume any specific format (not just markdown). Detect whatever structural patterns actually appear in the text.

            CHUNK CONTEXT:
            - This is chunk {} of unknown total
            - Previous section state: {:?}
            - Formatting pattern being tracked: {:?}

            CHUNK TEXT:
            {}

this is not giving specific enough data for the LLM to fully capture what it is doing and why, looking at this also we provide a previous section state but fail to tell it what it should identify follow it, it seems like we provide the state and pattern tracked but it fails to diverge based on what state its in to know what it should track next from there the goal as its not aware of all the states and the purpose and how itshould be tracking them, now loking at this deeper we should be tracking it as wellin order as we do for sentences the first then second and so on see how we have all different states as well if we are in section

State: IN_SECTION
  Trigger: Another heading of same or higher level detected
  Transition → NEW_SECTION (close current, open new)

  Trigger: Sub-heading detected (lower level than current)
  Transition → IN_SUBSECTION (nest under current)

  Trigger: Formatting pattern breaks (N paragraphs with no heading,
           style shift detected by zero-shot, narrative voice changes)
  Transition → DOCUMENT_BREAK or IDLE

so if we are againeach different trigger, now remember we are not tracking formatting breaks to transition to document_break or idle if we identify a section it will always be in section until the end we will at first ys until all chunks are processed just identiy one section captured all

  Trigger: Another heading of same or higher level detected
  Transition → NEW_SECTION (close current, open new)

  because at the end once we have the graphs we will traverse it to then identify document breaks

we are still and should still be capturing

  Trigger: Sub-heading detected (lower level than current)
  Transition → IN_SUBSECTION (nest under current)

  because if we find a lwer level than current then yes we capture that in subsection and break out of that if we come back to a higher level detection

then you see in our prompt how we have it as of now since we are not from the LLM caturing all explicitly and we just have it return

            Return ONLY valid JSON:
            {{
              "section_events": [
                {{
                  "type": "section_start|section_continue|section_end|document_break",
                  "position": <character_offset_in_chunk>,
                  "section_title": "title if section_start, null otherwise",
                  "section_level": <1-6 or null>,
                  "formatting_pattern": "description of detected pattern or null"
                }}
              ],
              "current_section_context": {{
                "section_id": "unique identifier for tracking across chunks",
                "level": <1-6 or null>,
                "title": "current section title or null",
                "formatting_pattern": "detected pattern description"
              }},
              "document_boundary_detected": false,
              "document_boundary_reason": "reason or null"
            }}"#,

we should be ensuring that again as we are working 1x1 more rnularin ordernow to correctly have it return in a correct format to directly work with our graphs to add the and all different states and levels ensuring also our enums are all aligned to capture all this correctlly to wrok with our graphs

also remember we are not doing document boudnarydetection or reason via zero-shot LLM prompting that is left for the graph traversal

## Section & Document Detection

### Why Section Detection Matters

Section detection is the only mechanism that allows the system to distinguish:
- Raw text (user's prompt, conversational input, free-form request)
- Embedded documents (guides, articles, reports, specifications)
- Mixed content (user request containing an attached or inline document)

Intent detection belongs to the AMT. Section detection belongs to the text modality.
The text modality identifies WHAT the content is structurally; the AMT determines
WHY the user provided it and what to do with it.

so we first have CleanChunks and in the path 2 we need to take into account what section we are in first while we find sentences and correct grammar.

SectionStateTracking (cross-chunk state machine, format-agnostic)

│  DocumentNode                                                               │
│    node_type: Document                                                      │
│    content: summary (word count, type, etc.)                               │
│    properties: { document_type, word_count, sentence_count, paragraph_count,│
│                  readability_score, title, source, role }                   │
│    position: { start_byte, end_byte }                                       │
│                                                                             │
│  SectionNode                                                                │
│    node_type: Section                                                       │
│    content: section title                                                   │
│    properties: { level (1-6), formatting_pattern, parent_section }          │
│    position: { start_byte, end_byte, start_line }


so we are getting rid of in full of the     /// Extract document structure
    fn extract_structure(&self, text: &str) -> DocumentStructure {

as this was rule based and going to utilize a zero-shot approach in place within the chunk processing itself

### Section State Machine

State tracking runs across chunk boundaries. The state is carried from chunk N to
chunk N+1 to handle sections that span multiple chunks.

```
SECTION TRACKING STATE MACHINE

State: IDLE
  Trigger: Any structured heading pattern detected
           (regardless of format — zero-shot determines the format)
  Transition → IN_SECTION

State: IN_SECTION
  Trigger: Another heading of same or higher level detected
  Transition → NEW_SECTION (close current, open new)

  Trigger: Sub-heading detected (lower level than current)
  Transition → IN_SUBSECTION (nest under current)

  Trigger: Formatting pattern breaks (N paragraphs with no heading,
           style shift detected by zero-shot, narrative voice changes)
  Transition → DOCUMENT_BREAK or IDLE

State: DOCUMENT_BREAK
  Trigger: New structural pattern begins
  Transition → IN_SECTION (new document context)

CROSS-CHUNK CONTINUATION:

  Chunk N ends mid-section:
    State carries:
      - section_id (persistent across chunks)
      - section_level (1-6 or detected equivalent)
      - formatting_pattern (discovered by zero-shot — any format)
      - current_document_id

  Chunk N+1 receives continuation context:
    - Inherits open section from chunk N
    - May close it or extend it
    - May detect a break → new document
```

**Format-agnostic detection:** The system does not hardcode markdown or any specific
format. Zero-shot LLM determines the formatting pattern from the content itself.
Patterns include (but are not limited to): markdown headers, numbered sections,
all-caps headings, underlined headings, Roman numerals, bold lead lines,
indented outlines, any consistent structural pattern the LLM identifies.

D. For detect modality we ant tobreak this down because we have it in detect_modalities_stable running a generalized 5x stability verification but we need to be more granular about it, our prompt in detect_modalities_in_chunk is wrong as well

        let prompt = format!(
            r#"Identify all modality content present in or referenced by this text.

    Available modalities to detect: {}

    Also detect:
    - "true_text" — content that is genuinely prose/text (not embedded code/math/etc.)
    - "unknown" — content that does not fit any listed modality

    For each detected modality, span_start and span_end are character offsets within
    the provided text. Do NOT include a content_snippet — positions are sufficient.

    Text chunk (index {}):
    {}

this we should drop the true text detection because with the section detection and document detection along with this modality detection we can we are already filtering and will be able to filter out what is true text or not via our document detection while already having identified everything else

the propmt also as we do now with sentence detection and section where we work on it in order 1x1 we will be doing the same for the modality detection, we will work in order 1x1 so we iterate to ask for the first modality if any it identifies in a chunk it doesn't matter if in a sentence or not etc example small in sentnces formulas math modality or large outof sentence forumalas or w.e from any modality or any modality which isn't just plain text. so then we are doing a 5x validation twice andthis is the same for the others sectino and sentence dectrion we do a 5x validation first on if the order is correct and then if what we are looking for is correct here in modality we do a 5x validation specifically on the current grabbed modality to then confirm if it is a modality and specifcally if for the category or modality it specified at first we ask what modality right but to validate the 5x validation we ask if the response iscorrect 5x again for both the order validation and this we are expecting ayes or no answer .

but now we also are looking at alongisde the two really more alongiside of looking for sentences we are also looking for

│  MODALITY DETECTION                                                │
│  ───────────────────────────                                                │                                                                                        │
│       a. Detect embedded modalities (code blocks, math, etc.)    │
│       b. Mark spans as "true_text" or detected modality name                │
│       c. 5x consecutive stable loop (detect_modalities_stable)              │
│    2. Create ModalityReferenceNode for non-text spans                       │
│    3. Store span_start / span_end (byte offsets, NO content_snippet)        │

│                                                                             │
│  MODALITY REFERENCE NODES                                                   │
│  ─────────────────────────                                                  │
│                                                                             │
│  ModalityReferenceNode (ChunkModalityDetection in code)                     │
│    modality: the detected modality name (from pipeline registry)            │
│    span_start: char offset within sentence/chunk                            │
│    span_end: char offset within sentence/chunk                              │
│    intent_reference: how this modality is referenced (describes/contains)   │
│    chunk_index: which chunk this was detected in                            │
│    NOTE: no content_snippet — retrieve dynamically: &text[span_start..end] │
│

**Modality spans**: Embedded code, math, etc. with byte positions so while we are looking at graphs and this it isimportant to note that we may have embebed modatliyies inside sentences, IE a simple formula or math formula etc and we tie this to the sentence nde if so if nt we tie it to the paragaph node not the chunk id because we are't using chunks anymore we are moving to full graphs so we tie it to either the setence node or parapgrah node

E. for paragraphs we also need to expand ur prompt because it is not diving into enought details it is to vauge also for paragraphs we are going to move it to 1x1 in order as well finding the first paragraph if any per chunk and if the first is found it looks for more in order, now I am contemplating on creating two paths for parapgrahs detection justlike we have chunks processed via two paths, 1. I am thinking of fr paragraphs we keep path 1 as is where we are looking to identify paragraphs 1x1 via propmts, here is the thing pargraphs are more like document detections to sentences where sentences are more like section to documents. here is the reasoningbehind it for sentencen and paragraphs we cangranulary per chunk capture sentences 1x1 correctly and expect this to work on any LLM no matter the size while we move to SLM's and more granular SLM's we can't expect it to work with paragraphs fully correctly just where we can expect it to work with sections identifying at least the start of sections and where it can't be expected to work with documents. So sentences gets everything ready for pargraphs in the most cnstrained scenario and that is what we are trying to capture where sections or at least section identifications can get everything ready for documents as well or at least help. setence idetnfication are small start and end span same for section start detection. For pargraphs we can capture what the model believes to be the the full paragraph section, does this overlap with sentences if sentece detectinn isn't working on detected paragraphs? the reason is if we rely on detected paragraphs to work on sections then per detected paragraphs what ensures is nothing was missed? the expected behavior

now the paragraph node

so apart from looking per chunk at sections/sentences/modality references we are also looking at


│  ParagraphNode                                                              │
│    node_type: Paragraph                                                     │
│    properties: { sentence_count, parent_section, document_id }             │
│    position: { start_byte, end_byte }

this we are doing similar to sections because this a paragraph per chunk can be cut off and we identify allparapgrahs ina  chunk if any but then if the last parapgrah if more then one or if any not found we look at it with the next chunk there really should be no parapgrah that is two chunks long realistically if we follow grammar rules but either way this is to ensure we acpture if a parapgrah most scenarios will be the last paragraph in a chunk may be cut off so we want to identify the parapgrah start and in the next chunk if end isn't found we with the paragraph start identified look for the parapgrah end confirm if true or not or find a new one etc etc

so again for now we can tie sentences to pargraphs nodes as well as for modality refrences we will be tying it either to paragraphs or setence node while forthe parapgrah node we can tie it to the section node for now even though section node may not be finalized or full refined which is ok

ah then see as you can see we do just as we are taking into account for sections into next chunk when we identify a section start and end isn't in the chunk or doesn't brak and as we do for the paragraph we also need to do for sentences and modality references because IE the last setences may be broken or cut off and it means it may be in the next chunk so we want to capture all this correctly as we do with sections take all into account without omission and without dropping anything ensuring alingment

F.

so now based on all of this this is how our processing flow looks like

┌───────────────────────────────────────────────────────────────────────────────────────────────                             ┐
│                                 TEXT MODALITY PROCESSING FLOW                                                              │
├───────────────────────────────────────────────────────────────────────────────────────────────                             ┤
│                                                                                                                            │
│  PHASE 1: PER-CHUNK PROCESSING                                                                                             │
│  ─────────────────────────────                                                                                             │
│                                                                                                                            │
│  For each chunk:                                                                                                           │
│    1. SectionStateTracking (cross-chunk state machine, format-agnostic)                                                    │
     2. Path Breakthrough
│       2.1. Path 1
            2.1.1 DetectParagraphs (identify paragraph breaks → byte offsets)                                                │
│               2.1.1.1. DetectSentences (within paragraphs → byte offsets)                                                  │
│                   2.1.1.1.1. CleanChunk (grammar correction, normalization via zero-shot LLM)                              │
                        2.1.1.1.1.1. Detect embedded modalities (code blocks, math, etc.)
        2.2 Path 2
            2.2.1. DetectSentences ( )
                2.2.1.1. CleanChunk (grammar correction, normalization via zero-shot LLM)                                    │
                    2.1.1.1.1.1. Detect embedded modalities (code blocks, math, etc.)
     3. Detect embedded modalities (code blocks, math, etc.)
│    5. Create ParagraphNodes + SentenceNodes if Phase 1 or just Sentence Nodes if Phase 2 → push to text modality graph     │
│    6. Store byte positions on every node (start_byte, end_byte, line)                                                      │
│    7. Create ChunkGraph (persistent historical position record)                                                            │
│                                                                                                                            │
│  State tracked across chunks:                                                                                              │
│    - Current section context (is a section open? what level?)                                                              │
│    - Detected formatting pattern (discovered by zero-shot, any format)                                                     │
│    - Current document boundary state                                                                                       │
│    - Current document ID (if inside a detected document)                                                                   │
├────────────────────────────────────────────────────────────────────────────                                               ─┤
│                                                                                                                            │
│  PHASE 2: CROSS-CHUNK AGGREGATION                                                                                          │
│  ─────────────────────────────────                                                                                         │
│                                                                                                                            │
│  After all chunks processed:                                                                                               │
│    1. Gather all ParagraphNodes from all chunk graphs  ( Path 1 )                                                          │
│    2. Gather all SentenceNodes from all chunk graphs                                                                       │
│    3. Link SentenceNodes to their ParagraphNodes     ( Path 1 )                                                            │
│    4. Link ParagraphNodes to their Section/Document parents ( Path 1 )                │
│    5. Resolve cross-chunk paragraph continuations       ( Path 1 )                    │
│    6. Finalize DocumentNode → SectionNode → ParagraphNode → SentenceNode              │
│       hierarchy in the text modality graph     ( Path 1 )                             │
│                                                                                       │
├─────────────────────────────────────────────────────────────────────────────                                               ┤
│                                                                             │
│  PHASE 3: GRAMMAR EXTRACTION (on full corpus, graph-native)                 │
│  ──────────────────────────────────────────────────────────                 │
│                                                                             │
│    1. For each SentenceNode:                                                │
│       a. ExtractGrammarRelationships(sentence) → grammar edges + nodes      │
│       b. Link grammar nodes to SentenceNode                                 │
│                                                                             │
│    2. Pairwise cross-sentence comparison (build relationship depth):         │
│       a. Sentence × Sentence → relationship edges                           │
│       b. Coreference chains (John → He → the manager)                       │
│       c. Causal chains (because → therefore → consequently)                 │
│       d. Temporal chains (yesterday → then → afterwards)                    │
│       e. Discourse relations (Elaborates, Contradicts, Enables, etc.)       │
│                                                                             │
│    3. GRAPH TRAVERSAL for entity/keyword/topic derivation:                  │
│       a. Traverse SubjectNodes → derive Person/Org/Concept entities         │
│       b. Traverse ObjectNodes → derive Thing/Place/Event entities           │
│       c. Traverse VerbNodes → derive action keywords                        │
│       d. Traverse RelationshipEdges → derive topics                         │
│       (NOT separate extraction — derived FROM the relationship graph)       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 4: Paragraph/Section/Document construction (on full corpus, graph-native)  Path 2 Exclusive          │
│  ──────────────────────────────────────────────────────────                                                │
│                                                                             │
    Stage 1
    Stage 2
├─────────────────────────────────────────────────────────────────────────────┤


Here we have to understand that there is a clear difference between Path 1 and Path 2, Path 1 works with most LLM already LLM can handle full paragraph and sentence identification easily, path 1 deconstructs via chunking to break everything apart into its proper place. Path 2 is the constructor it is looking to traverse from the bottom/start to put everything together into its proper place.
path 2 works directly with the graphs for understanding and construction of sentences and paragraphs.

now in path 2 is where we see

Stage 1: Candidate Structural Unit Formation (Context Pool Accumulation)
The traversal starts at a node (e.g., Sentence 1). It does not ask "is this a section?"

    It visits the grammar nodes of Sentence 1.
    It walks outgoing grammar edges (e.g., Object -> "service").
    It looks for coreference or reference edges. It finds Sentence 2's Subject references "service."
    It walks to Sentence 2 and adds it to the Context Pool.
    It continues walking established edges (Sentence 2 Object -> "JWT" -> Sentence 3 Subject "Tokens").
    The Context Pool grows: Sentence 1 + Sentence 2 + Sentence 3 + all their grammar nodes.
    The traversal reaches a Semantic/Structural Frontier. Sentence 4 has no incoming coreference or dependency edges from Sentences 1-3. The frontier weakens and stops.
    Result: A maximal connected structural neighborhood (Candidate Unit).

come to life, for path 2 we are only capturing the sentences per chunk not the paragraphs for reconstruction via the graph after all sentences have been obtained so this is after having processed all chunks. SO we see this in phase 4 for path 2.

Phase 2 now does stage 1 as well as stage 2

Stage 2: Structural Boundary Evaluation
Now the detector evaluates the accumulated Context Pool.

    Does the next neighborhood (Sentence 4+5) share structural continuity with the first?
    Evidence check: No shared entities, no coreference, no dependency chains, new actors ("Marketing").
    Result: A structural boundary is detected.
    Promotion: The first Context Pool is promoted to a Section. The second Context Pool is promoted to a new Section.
    (This same logic applies at the Document level: traversing Sections to find when a document's structural continuity breaks and closes).

here is the thing stage 2 works for paragraphs first, it also works for sections/document boundaries while for sections we can detect section element like headers etc etc because this is really what sections are sections for formalized boundaries right and there is formats to follows rules while we can yes identify sections not all text is formatted properly some as youcan see with mines may take sections apart to reference them etc etc so while we can identify section elements stage 2 helps us validate if this is a continution of the prior section or current to group together correctly or if it is just a broken run off. stage 2 is not identifying section elements but verifying when it comes to sections if it is within such elements bounded.

now it is important to note for Stage 2 we are always working in order. when identifying sentences nodes we identify them in order to work on stage 2 in order as well.

G.

# OMEX: Omni-Execution Format

OMEX (Omni-Execution Format) is a universal, modular format and execution interface designed for AI model representation, distributed AI reasoning, scalable inference, and multi-device coordination. OMEX serves as both a model representation layer and an orchestration layer, enabling intelligent task execution across mobile, browser, Pi devices, servers, and specialized inference nodes — creating a unified AI operating environment. Enabling even ultra-large models (70B+) to run directly on constrained devices.

## Key Features

**Modular Execution Containers**: Encapsulate and structure AI task flows for reasoning, generation, analysis, or data routing.

**Direct Model Representation**: OMEX is not just a container but the actual model format, eliminating the need for converting between formats.

**Local-First Execution**: All inference happens on the local device with no distributed execution required, regardless of model size.

**Memory Optimization Built-In**: Smart task chunking, layer targeting, quantized slices, and token stream planning allow running large models on constrained devices.

**Multi-Device Compatibility**: Designed to execute tasks across diverse environments: Edge, Cloud, and Local, with emphasis on local execution.

**ZSEI Integration**: Leverages ZSEI for guidance generation, execution planning, and persistent state tracking.

**Scalable Local Model Execution**: Enables ultra-large models (e.g., 70B+) to execute directly on constrained devices through advanced memory management techniques.

**Streaming & Chunk-Based Processing**: Efficiently handles large content (code, video, documents) through adaptive chunking and stream execution.

**Embeddable Agent API**: Create agent logic that conforms to OMEX protocols, whether embedded in software, servers, or local hardware.

**Flexible Storage**: Model weights and data can be stored locally, fetched, or cached from remote sources (but always executed locally).

**Prompt-First Design**: OMEX is prompt-driven, optimizing execution paths based on prompt requirements rather than static model exports.

**Hardware-Aware Execution**: Optimizes execution for specific hardware, including tensor core utilization, precision scaling, and memory management.

**ZSEI-Enhanced Training Optimization**: Utilizes ZSEI's Neural Architecture Analysis Framework during model training to discover fundamental optimization patterns that are embedded into fast execution optimizers.

**Hybrid Intelligence Architecture**: Combines ZSEI's deep semantic understanding during model creation with lightning-fast traditional ML optimization during execution.

**Pre-Computed Optimization Embedding**: Embeds optimization insights discovered by ZSEI into the model format itself, enabling instant optimization without runtime analysis overhead.

## How It Works

OMEX works as both a model format and runtime interpreter, utilizing a revolutionary hybrid approach that combines the intelligence of zero-shot analysis with the speed of pre-computed optimizations:

1. **Model Creation Phase (ZSEI-Enhanced)**:
   - ZSEI's Neural Architecture Analysis Framework performs deep semantic analysis of the base model architecture
   - Zero-shot analysis discovers optimization patterns, hardware mapping strategies, and execution insights
   - These insights are compressed into fast execution optimizers and embedded directly into the OMEX model format
   - The result is a model that contains both weights and embedded intelligence about optimal execution

2. **Prompt/Request**: A user or system sends a request (text, code, task).

3. **Lightning-Fast Optimization (Embedded Intelligence)**:
   - The embedded execution optimizer (containing compressed ZSEI insights) analyzes the prompt in milliseconds
   - Hardware-specific optimizations are applied based on pre-computed strategies
   - Graph modifications are applied using embedded pattern recognition

4. **ZSEI Interpretation (Optional Enhanced Mode)**: For complex or novel scenarios, the request can be routed through ZSEI which analyzes the intent, activates guidance frameworks, and returns an execution plan.

5. **OMEX Format Generation**: The plan is serialized as an OMEX container, composed of task nodes and execution graphs that directly represent model structure and flow.

6. **Local Optimized Execution**:
   - Smart partitioning of model layers and weights based on embedded optimization strategies
   - Memory-optimized execution pipeline using pre-computed memory management plans
   - Dynamic resource allocation based on device capabilities and embedded hardware insights
   - Token-level streaming to manage memory constraints

7. **Execution Completion & Feedback Loop**: Results are streamed or returned through OMEX APIs, optionally logged into ZSEI for state retention or future recall.

## Revolutionary Hybrid Architecture

OMEX implements a groundbreaking hybrid architecture that leverages the best aspects of both zero-shot analysis and traditional machine learning optimization:

### Training-Time Intelligence (ZSEI-Powered)
During model creation and training, ZSEI's Neural Architecture Analysis Framework provides deep semantic understanding of the model architecture. This analysis discovers optimization patterns that would take human researchers years to identify, such as universal attention head redundancies, optimal weight sharing strategies, and hardware-specific execution patterns.

The key insight is that this deep analysis happens when time is not critical, allowing ZSEI to explore complex optimization strategies that would be too slow for real-time execution. These discoveries are then distilled into fast execution optimizers that embed the wisdom of zero-shot analysis without the computational overhead.

### Execution-Time Speed (Embedded Optimizers)
During model execution, OMEX utilizes small, fast execution optimizers that contain the compressed intelligence from ZSEI's training-time analysis. These optimizers can make complex optimization decisions in 2-5 milliseconds, providing the benefits of deep architectural understanding without the startup time penalty of real-time analysis.

This hybrid approach delivers the best of both worlds: the semantic understanding and novel pattern discovery of zero-shot analysis, combined with the lightning-fast execution speed of traditional machine learning optimization.

### Performance Characteristics
The hybrid architecture delivers superior performance across all metrics:

- **Model Quality**: 15-25% better than traditional approaches due to ZSEI's architectural insights
- **Startup Time**: 2-5ms optimization overhead (compared to 200-400ms for pure zero-shot)
- **Execution Speed**: 40-80% faster than traditional models due to embedded optimizations
- **Adaptability**: Handles novel scenarios through optional ZSEI enhancement mode
- **Resource Efficiency**: Optimal memory and compute utilization through embedded intelligence

## OMEX Format Specification

An OMEX file is a structured, modular execution container that directly represents the model architecture and execution flow, enhanced with embedded optimization intelligence.

```json
{
  "omex_version": "0.2.0-neural",
  "task_id": "auth-system-analysis-456",
  "task_type": "code_analysis",
  "model_architecture": {
    "type": "transformer",
    "parameters": "70B",
    "quantization": "int8",
    "memory_profile": {
      "required_ram": "4GB",
      "swap_strategy": "layer_offloading"
    },
    "zsei_optimization_insights": {
      "architecture_signature": "llama3_70b_variant_001",
      "optimization_fingerprint": "a7f9d8e2c1b4556e8f3a9d7c",
      "embedded_optimizer": {
        "model_path": "./optimizers/execution_optimizer.bin",
        "version": "1.0.0",
        "capabilities": ["graph_pruning", "memory_optimization", "hardware_adaptation"]
      },
      "discovered_patterns": {
        "attention_redundancy": 0.35,
        "mlp_fusion_opportunities": 12,
        "quantization_tolerance": "high",
        "streaming_efficiency": 0.82
      }
    }
  },
  "context": {
    "source": "user",
    "requested_by": "terminal",
    "zsei_guidance": true,
    "device_profile": "raspberry-pi-5",
    "optimization_level": "embedded_fast"
  },
  "execution_graph": {
    "entry_node": "node_1",
    "optimization_strategy": "embedded_optimizer",
    "nodes": [
      {
        "id": "node_1",
        "type": "parse_codebase",
        "input": ["./src"],
        "output": ["parsed_ast.json"],
        "runner": "local_device",
        "memory_allocation": "512MB",
        "optimization_hints": ["vectorize_parsing", "stream_large_files"]
      },
      {
        "id": "node_2",
        "type": "semantic_analysis",
        "depends_on": ["node_1"],
        "runner": "local_cpu",
        "guidance": "zsei://frameworks/code-analysis",
        "model_layers": ["encoder_1", "encoder_2", "encoder_3"],
        "memory_allocation": "1GB",
        "embedded_optimizations": {
          "attention_pruning": 0.25,
          "precision_scaling": "fp16",
          "kernel_fusion": ["layer_norm", "linear", "activation"]
        }
      },
      {
        "id": "node_3",
        "type": "summary_generation",
        "depends_on": ["node_2"],
        "runner": "local_with_embedded_optimization",
        "model_layers": ["decoder_1", "decoder_2", "decoder_3"],
        "memory_allocation": "2GB",
        "optimization_profile": "streaming_generation"
      }
    ]
  },
  "model_weights": {
    "storage_strategy": "chunked_local",
    "chunk_size": "200MB",
    "prefetch_strategy": "predictive_zsei_guided"
  },
  "completion_policy": {
    "retry_on_fail": true,
    "max_runtime_minutes": 60,
    "memory_pressure_strategy": "dynamic_precision_reduction_embedded"
  }
}
```

## Canonical OMEX Model Format

OMEX models follow a canonical structure that combines a computation graph, parameters, and processing logic into a single, self-contained format, enhanced with embedded optimization intelligence.

### Folder Structure

```
model.omex/
├── metadata.json                    # Model metadata and versioning
├── graph.json                       # Computation graph structure
├── optimization/                    # ZSEI-discovered optimization data
│   ├── execution_optimizer.bin      # Embedded fast execution optimizer
│   ├── zsei_insights.json          # Compressed ZSEI analysis insights
│   ├── hardware_profiles.json      # Hardware-specific optimization strategies
│   └── pattern_database.bin        # Discovered universal patterns
├── weights/                         # Model weights directory
│   ├── quantization.json           # Weight quantization configuration
│   ├── layer_0.bin                 # Binary weight data for layer 0
│   ├── layer_1.bin                 # Binary weight data for layer 1
│   └── ...                         # Additional layer weights
├── tokenizer/                       # Tokenizer components
│   ├── tokenizer.json              # Tokenizer configuration
│   ├── vocab.txt                   # Vocabulary file
│   └── merges.txt                  # BPE merges file (if applicable)
├── preprocessors/                   # Pre-processing components
│   ├── text_preprocessor.json      # Text preprocessing configuration
│   ├── code_preprocessor.json      # Code preprocessing configuration
│   └── ...                         # Additional preprocessors
├── postprocessors/                  # Post-processing components
│   ├── text_formatter.json         # Text formatting configuration
│   ├── code_formatter.json         # Code formatting configuration
│   └── ...                         # Additional postprocessors
├── agents/                          # Agent definitions
│   ├── default_agent.json          # Default agent configuration
│   ├── code_agent.json             # Code-specific agent configuration
│   └── ...                         # Additional agent configurations
└── extensions/                      # Optional extension components
    ├── tools/                       # Tool definitions
    ├── adapters/                    # LoRA or adapter weights
    └── hooks/                       # Execution hooks
```

### Architecture Components

1. **Computation Graph (`graph.json`)**:
   - Full model structure definition with embedded optimization hints
   - Layer specifications, attention blocks, activations with efficiency annotations
   - Control flow and execution paths optimized by ZSEI analysis
   - Modular subgraphs and component references with pre-computed fusion opportunities

2. **Embedded Optimization Intelligence (`optimization/`)**:
   - Fast execution optimizer containing compressed ZSEI insights
   - Hardware-specific optimization strategies discovered during training
   - Universal pattern database for cross-model optimization opportunities
   - Pre-computed memory management and resource allocation strategies

3. **Parameters (`weights/`)**:
   - Trained values organized by layer with optimal quantization strategies
   - Support for multiple quantization formats discovered by ZSEI analysis
   - Efficient chunking patterns optimized for different hardware configurations
   - Memory-mapped access strategies for streaming execution

4. **Tokenizer + Pre/Post-Processing (`tokenizer/`, `preprocessors/`, `postprocessors/`)**:
   - Integrated tokenization optimized for specific model architectures
   - Input formatting and normalization with embedded efficiency improvements
   - Output formatting and detokenization with streaming optimizations
   - Specialized processing pipelines for different content types

5. **Agent Definitions (`agents/`)**:
   - Runtime behavior specifications enhanced with optimization profiles
   - Task-specific configurations with embedded performance strategies
   - Tool usage patterns optimized for different execution environments
   - Memory and context management strategies discovered by ZSEI

## Architecture Overview

### OMEX Runtime

**Interpreter**: Parses and executes the OMEX format directly on supported hardware using embedded optimization intelligence

**Model Engine**: Processes model weights and architecture definitions for local execution with ZSEI-discovered optimizations

**Embedded Optimizer**: Utilizes fast execution optimizers containing compressed ZSEI insights for millisecond-speed optimization decisions

**Memory Optimizer**: Manages dynamic allocation, swapping, and precision adjustment using pre-computed strategies to fit models within device constraints

**Execution Dispatcher**: Coordinates task distribution across local processing units using embedded intelligence

**ZSEI Agent API**: Optional plugin to fetch guidance, embeddings, task plans for complex scenarios requiring real-time analysis

**Memory Manager**: Streams in/out data and model weights using optimal patterns discovered during training

### ZSEI Integration

ZSEI is used in two distinct phases:

**Training-Time Deep Analysis (Comprehensive)**:
- Generate execution blueprints through deep semantic analysis of model architectures
- Discover universal optimization patterns across different neural network types
- Identify hardware-specific optimization opportunities through zero-shot understanding
- Create specialized execution optimizers that embed discovered insights
- Analyze cross-model patterns to build universal optimization databases

**Runtime Enhancement (Optional)**:
- Handle complex scenarios that exceed embedded optimizer capabilities
- Provide specialized guidance for novel task types not covered by embedded intelligence
- Maintain long-running state for complex multi-stage operations
- Coordinate distributed execution across multiple devices when needed

## Creating and Training Models with OMEX

OMEX supports multiple approaches for creating and training models, with ZSEI integration providing revolutionary optimization capabilities during the training phase:

### ZSEI-Enhanced Model Training

The most powerful feature of OMEX is its integration with ZSEI's Neural Architecture Analysis Framework during model training. This process discovers optimization patterns that are then embedded into the model for lightning-fast execution:

```bash
# Create ZSEI-optimized model from scratch
omex create-model --architecture transformer --size 7B --zsei-analysis comprehensive --output-dir ./optimized_model.omex

# Train with ZSEI continuous optimization
omex train --model-dir ./optimized_model.omex --train-data ./dataset.jsonl --zsei-optimization enabled --discover-patterns --target-hardware "gpu,mobile,edge" --epochs 3

# Generate embedded execution optimizers from ZSEI insights
omex generate-optimizers --model-dir ./optimized_model.omex --zsei-insights ./training_insights.json --hardware-profiles all
```

### Converting Existing Models with ZSEI Enhancement

Existing models can be converted to OMEX format with ZSEI analysis for optimization discovery:

```bash
# Convert from PyTorch with ZSEI analysis
omex convert --source-format pytorch --model-path ./llama3_7b.pt --output-dir ./llama3_7b.omex --zsei-analysis comprehensive --discover-optimizations

# Convert with hardware-specific optimization discovery
omex convert --source-format onnx --model-path ./phi3_mini.onnx --output-dir ./phi3_mini.omex --zsei-analysis comprehensive --target-hardware "mobile,pi,gpu" --embed-optimizers

# Convert with cross-model pattern learning
omex convert --source-format huggingface --model-name "mistralai/Mistral-7B-v0.1" --output-dir ./mistral_7b.omex --zsei-analysis comprehensive --learn-universal-patterns --pattern-database ./universal_patterns.db
```

### Direct Training in OMEX Format with ZSEI

OMEX supports training models directly in its native format with continuous ZSEI optimization:

```bash
# Initialize with ZSEI architectural analysis
omex init --architecture transformer --size 7B --output-dir ./my_model.omex --zsei-preanalysis --optimization-targets "speed,memory,quality"

# Configure with ZSEI-discovered optimal parameters
omex configure --model-dir ./my_model.omex --zsei-optimize-architecture --layers auto --heads auto --dim auto --vocab-size 32000

# Train with continuous ZSEI optimization discovery
omex train --model-dir ./my_model.omex --train-data ./dataset.jsonl --val-data ./validation.jsonl --epochs 3 --batch-size 32 --gradient-accumulation 8 --zsei-continuous-optimization --pattern-discovery
```

### Fine-tuning OMEX Models with ZSEI Enhancement

Existing OMEX models can be fine-tuned with ZSEI providing optimization insights:

```bash
# Fine-tune with ZSEI-guided LoRA optimization
omex finetune --model-dir ./llama3_7b.omex --train-data ./instruction_data.jsonl --method lora --zsei-optimize-lora --lora-r auto --lora-alpha auto --output-dir ./llama3_7b_finetuned.omex

# Create domain-specific adapters with ZSEI analysis
omex create-adapter --model-dir ./mistral_7b.omex --adapter-name "code-adapter" --train-data ./code_examples.jsonl --zsei-domain-analysis code --output-dir ./code_adapter.omex

# Merge adapters with ZSEI optimization verification
omex merge-adapter --model-dir ./llama3_7b.omex --adapter-path ./code_adapter.omex --zsei-verify-compatibility --optimize-merge --output-dir ./llama3_7b_code.omex
```

### Creating MoE Models with ZSEI Analysis

OMEX supports creating Mixture of Experts (MoE) models with ZSEI providing expert design insights:

```bash
# Create MoE with ZSEI expert analysis
omex create-moe --model-dir ./llama3_7b.omex --experts auto --expert-size auto --zsei-expert-analysis --routing-strategy learned --output-dir ./llama3_7b_moe.omex

# Train routing with ZSEI pattern recognition
omex train-router --model-dir ./llama3_7b_moe.omex --train-data ./dataset.jsonl --zsei-routing-optimization --output-dir ./llama3_7b_moe_routed.omex
```

## Performance Optimization

OMEX implements the revolutionary hybrid optimization approach that combines ZSEI's deep training-time analysis with lightning-fast embedded execution optimizers:

### Hybrid Optimization Architecture

The key innovation is separating optimization intelligence across two phases:

**Training-Time Deep Analysis (ZSEI-Powered)**:
- Comprehensive semantic analysis of model architecture when time permits deep exploration
- Discovery of universal optimization patterns across model families
- Hardware-specific optimization strategy development
- Creation of embedded execution optimizers containing compressed insights

**Execution-Time Lightning Speed (Embedded Optimizers)**:
- Millisecond-speed optimization decisions using pre-computed strategies
- Hardware-aware execution planning using embedded intelligence
- Dynamic graph modification using pattern recognition
- Memory management using pre-analyzed optimal strategies

### Performance Gains Across Hardware Tiers

#### Edge Devices (Mobile, Raspberry Pi)

The hybrid approach delivers exceptional performance on resource-constrained devices:

| Module Type | Traditional | Pure Zero-Shot | OMEX Hybrid | Improvement |
|-------------|------------|----------------|-------------|-------------|
| Startup Time | 200ms | 300-500ms | 150ms | 25% faster |
| Tokenizer | ~25 tok/s | ~30 tok/s | 50-60 tok/s | 100-140% faster |
| MLP (Quantized) | ~20 inf/s | ~35 inf/s | 70-90 inf/s | 250-350% faster |
| Attention (Optimized) | ~15 inf/s | ~25 inf/s | 40-50 inf/s | 167-233% faster |
| Memory Efficiency | Baseline | 20% improvement | 60% improvement | 3x more efficient |

#### Mid-Range GPUs (RTX 4090, A6000)

| Module Type | Traditional | Pure Zero-Shot | OMEX Hybrid | Improvement |
|-------------|------------|----------------|-------------|-------------|
| Startup Time | 50ms | 200-400ms | 30ms | 40% faster |
| Tokenizer | ~80k tok/s | ~90k tok/s | 120-140k tok/s | 50-75% faster |
| MLP Layer | 100 inf/s | 120 inf/s | 160-200 inf/s | 60-100% faster |
| Multi-head Attention | ~70 inf/s | ~90 inf/s | 130-160 inf/s | 86-129% faster |
| Batch Processing | Medium | Good | Excellent | 300%+ throughput |

#### High-End Hardware (H100, A100, TPUv4+)

| Module Type | Traditional | Pure Zero-Shot | OMEX Hybrid | Improvement |
|-------------|------------|----------------|-------------|-------------|
| Startup Time | 100ms | 500-800ms | 20ms | 80% faster |
| Tokenizer | ~300k tok/s | ~400k tok/s | 600-800k tok/s | 100-167% faster |
| FP16/BF16 MLP | ~2000 inf/s | ~2500 inf/s | 4000-5000 inf/s | 100-150% faster |
| MoE Layer | ~1200 inf/s | ~2000 inf/s | 3500-4500 inf/s | 192-275% faster |
| End-to-End Latency | ~1.5s | ~1.0s | ~400-600ms | 60-73% faster |

### The Hybrid Advantage

The revolutionary aspect of OMEX's hybrid approach is that it delivers better performance than both traditional approaches and pure zero-shot analysis:

**Better Than Traditional**: OMEX models contain embedded intelligence discovered through ZSEI's deep analysis, resulting in fundamentally superior architectures and execution strategies.

**Faster Than Zero-Shot**: By pre-computing optimization strategies during training, OMEX eliminates the runtime analysis overhead while retaining the benefits of semantic understanding.

**Adaptive Intelligence**: For novel scenarios, OMEX can fall back to real-time ZSEI analysis, providing the best of both worlds.

## Installation

### Requirements

Rust 1.73+

Optional: Python 3.10+ (for glue layers or embedded scripts)

Local compute resources (CPU/GPU/NPU)

ZSEI installed (recommended for full optimization capabilities)

### Install OMEX CLI

```bash
git clone https://github.com/your-org/omex.git
cd omex
cargo build --release
cargo install --path .
```

## Quick Start

### Run a Local OMEX Task

```bash
omex execute --file ./examples/code_summary.omex.json
```

### Use with ZSEI for Enhanced Optimization

```bash
zsei process "Summarize my auth system" --output-format omex --neural-optimize | omex execute --mode hybrid
```

### Load a Large Model with Embedded Optimizations

```bash
omex load-model --size 70B --device "raspberry-pi-5" --use-embedded-optimizations --memory-strategy streaming
```

### Create an Optimized Model with ZSEI Analysis

```bash
omex create-optimized --source ./base_model.pt --zsei-analysis comprehensive --target-hardware "mobile,gpu" --output ./optimized.omex
```

## Example Use Cases

**Local Large Model Inference**: Run 70B+ parameter models on phones, Raspberry Pis, and laptops using embedded optimization intelligence.

**Memory-Constrained AI**: Execute complex AI tasks on devices with limited RAM through ZSEI-discovered memory management strategies.

**Local-First AI Agents**: Create AI agents that operate entirely on local hardware with embedded optimization for maximum efficiency.

**Training-Time Architecture Discovery**: Use ZSEI to discover fundamental optimization patterns during model training that improve all future inference.

**Cross-Model Pattern Learning**: Leverage ZSEI's analysis across multiple models to build universal optimization databases.

**Hardware-Specific Optimization**: Create models optimized for specific hardware configurations using ZSEI's semantic hardware understanding.

**Federated Learning Coordination**: Dispatch and retrieve local model updates using OMEX containers with embedded optimization for efficient distributed training.

## OMEX + ZSEI Agent Architecture

ZSEI Agents operate in the hybrid architecture through two distinct modes:

**Training-Time Analysis Agents (Deep Intelligence)**:
- Accept model architectures for comprehensive semantic analysis
- Discover optimization patterns through zero-shot understanding of neural network structures
- Generate hardware-specific optimization strategies
- Create embedded execution optimizers containing compressed insights
- Build universal pattern databases for cross-model optimization

**Runtime Enhancement Agents (Fast Execution)**:
- Utilize embedded optimization intelligence for millisecond-speed decisions
- Handle complex scenarios requiring real-time analysis
- Provide fallback capabilities for novel task types
- Coordinate multi-device execution when needed
- Report progress and logs with minimal overhead

Agents can be installed as:

**Training Workstation Daemons**: Powerful processes handling deep ZSEI analysis during model development

**Embedded Execution Agents**: Lightweight agents containing compressed ZSEI intelligence for fast inference

**Mobile Optimization Workers**: Specialized applications leveraging embedded optimizations for on-device inference

**Server Intelligence Coordinators**: Centralized agents managing optimization strategies across distributed systems

**WebAssembly Edge Containers**: Browser-compatible execution environments with embedded optimization capabilities

## Developer Tools

**omex validate <file>**: Validate format and structure of OMEX model files including embedded optimization components

**omex trace <task_id>**: View execution status of running models with optimization performance metrics

**omex convert**: Generate OMEX model representations with optional ZSEI optimization analysis

**omex profile**: Show available runtimes and device capacity with optimization recommendations

**omex optimize**: Apply ZSEI analysis to discover and embed optimization strategies

**omex layer-inspect**: Examine memory requirements and optimization opportunities of individual model layers

**omex benchmark**: Measure performance improvements from embedded optimizations across hardware configurations

**omex compare**: Compare performance between OMEX hybrid approach and other formats

**omex graph-visualize**: Create visual representation of optimized execution graphs with performance annotations

**omex memory-trace**: Track memory usage patterns and optimization effectiveness

**omex zsei-analyze**: Perform deep ZSEI analysis on model architectures to discover optimization opportunities

**omex embed-optimizers**: Generate and embed fast execution optimizers from ZSEI analysis results

**omex pattern-discover**: Use ZSEI to discover universal optimization patterns across multiple models

**omex hardware-optimize**: Create hardware-specific optimization profiles using ZSEI semantic understanding

## Configuration

Set global settings in ~/.omex/config.toml:

```toml
[runtime]
default_runner = "local_device"
memory_optimization = "embedded_intelligent"
enable_zsei_integration = true
optimization_mode = "hybrid"  # "traditional", "zero_shot", "hybrid"

[model]
default_quantization = "zsei_optimized"
layer_swapping = true
precision_scaling = true
embedded_optimization = true
zsei_fallback = true

[storage]
cache_dir = "~/.omex/cache"
model_weights_dir = "~/.omex/models"
optimization_cache_dir = "~/.omex/optimizations"
pattern_database_dir = "~/.omex/patterns"
log_retention_days = 7

[zsei]
host = "http://localhost:8801"
agent_token = "YOUR_ZSEI_API_TOKEN"
training_analysis_mode = "comprehensive"
runtime_enhancement_mode = "fallback"
pattern_discovery = true
cross_model_learning = true

[performance]
tensor_cores = true
kernel_fusion = true
parallel_execution = true
streaming_tokens = true
kv_cache_optimization = true
embedded_optimizer_priority = "speed"  # "speed", "memory", "balanced"
zsei_analysis_depth = "comprehensive"  # "basic", "standard", "comprehensive"

[optimization]
enable_embedded_optimizers = true
enable_pattern_discovery = true
enable_hardware_adaptation = true
enable_cross_model_learning = true
optimization_cache_size = "1GB"
pattern_matching_threshold = 0.85
hardware_profile_auto_detection = true

[device]
cpu_threads = 8
gpu_memory_limit = "4GB"
swap_path = "~/.omex/swap"
enable_mmap = true
prefetch_distance = 2
optimization_profile = "auto"  # "mobile", "desktop", "server", "auto"
```

## Optimization Checklist

### Universal Hybrid Optimizations
- Embed ZSEI-discovered optimization patterns for instant application
- Convert training-time insights into millisecond-speed execution decisions
- Remove unnecessary graph segments using embedded pattern recognition
- Apply pre-computed hardware-specific optimizations
- Use embedded intelligence for prompt-aware graph modifications
- Implement cached optimization strategies for common scenarios
- Enable automatic fallback to real-time ZSEI analysis for novel situations

### Training-Time ZSEI Analysis
- Perform comprehensive semantic analysis of model architectures
- Discover universal optimization patterns across model families
- Identify hardware-specific optimization opportunities
- Generate specialized execution optimizers for different device categories
- Build cross-model pattern databases for future optimization
- Analyze attention mechanisms for redundancy patterns
- Optimize MLP structures for fusion opportunities

### Execution-Time Embedded Intelligence
- Use embedded optimizers for millisecond-speed optimization decisions
- Apply pre-computed graph modifications based on prompt analysis
- Implement cached memory management strategies
- Utilize hardware-specific optimization profiles
- Enable dynamic precision adjustment using embedded intelligence
- Apply streaming optimizations based on discovered patterns

## Format Comparison

OMEX Hybrid Approach vs. other model formats:

| Feature | ONNX | GGUF | TorchScript | MLIR | SavedModel | OMEX Hybrid |
|---------|------|------|------------|------|------------|-------------|
| Design Philosophy | Interchange | Quantized LLM | Traced Export | IR Compilation | Static Export | Prompt + Intelligence |
| Optimization Timing | Runtime | Pre-computed | Export-time | Compile-time | Export-time | Training + Execution |
| Intelligence Level | Basic | None | Basic | High | Basic | Revolutionary |
| Startup Speed | Fast | Fast | Fast | Medium | Fast | Lightning Fast |
| Execution Quality | Medium | Good | Medium | Good | Medium | Exceptional |
| Hardware Adaptation | Limited | Basic | Limited | Good | Limited | Intelligent |
| Novel Scenario Handling | Poor | Poor | Poor | Good | Poor | Excellent |
| Memory Efficiency | Basic | Good | Basic | Good | Basic | Optimal |
| Cross-Model Learning | None | None | None | Limited | None | Advanced |

## Throughput Performance

OMEX Hybrid delivers superior performance through the combination of embedded intelligence and optional real-time enhancement:

### Revolutionary Performance Gains

The hybrid approach consistently outperforms all other approaches across hardware tiers:

**Edge Devices**: 2-4x performance improvement with 60% better memory efficiency
**Mid-Range GPUs**: 1.5-2.5x performance improvement with intelligent batch processing
**High-End Hardware**: 2-3x performance improvement with optimal tensor core utilization
**Startup Time**: 20-80% faster startup across all hardware categories
**Adaptability**: Handles novel scenarios better than traditional approaches while maintaining speed

### The Hybrid Advantage Explained

Traditional approaches optimize either for speed or intelligence, but not both. OMEX's hybrid architecture achieves both by:

1. **Training-Time Intelligence**: Using ZSEI's comprehensive analysis when time permits deep exploration
2. **Execution-Time Speed**: Applying discovered insights through fast embedded optimizers
3. **Adaptive Enhancement**: Falling back to real-time analysis for complex scenarios
4. **Continuous Learning**: Building optimization databases that improve over time

This results in models that are both fundamentally better (due to ZSEI's architectural insights) and faster to execute (due to embedded optimization intelligence).

## Roadmap

[x] OMEX Task Container Format

[x] OMEX CLI Runtime

[x] ZSEI API Integration

[x] Direct Model Representation

[x] Hybrid Architecture Implementation

[ ] Advanced Embedded Optimizer Generation

[ ] Cross-Model Pattern Learning Database

[ ] Universal Optimization Pattern Discovery

[ ] Agent Mesh Execution Support

[ ] WASM & Mobile Agent Launchers

[ ] Browser Plugin for OMEX Applets

[ ] Visual OMEX Designer GUI with Optimization Visualization

## License

MIT License. See LICENSE for details.

## Learn More

ZSEI Documentation

OMEX Wiki

Neural Architecture Analysis Framework Documentation

Join the discussion on Telegram

this is OMEX its a new format I am working on for Models.

there are a lot of corrections and clarifications we need to make to this base new insights obtained.

While we may look to optimize memory I do not believe we need memory allocation constrainsper model this would be incorrect behavior and is overengineering. We for now are removing the embeeded optimized intelligence from the OMEX format as we redraw everything to not follow overenginred patterns. We as well are removing hardware-aware execution. As well as adapative chunking.

We are detaching from ZSEI, the model creation phase of OMEX iscompletely incorrect as stated we are not capturing optimization patterns tied around hardware mapping or execution insights. This is not what OMEX is.

To understand the correct model format of OMEX we have to take into account what NLP's/SLM's/LLM's are and the goal around a propmt first design scuh as OMEX built around a AGI harness such a Ozone-Studio. When we receive a prompt in general for any system the goal is to understand the prompt. TO do so the thing first to come into play and that is the understanding of grammar and language. Before understanding langauge we must first undersand grammar. as laguage is bounded by a st rules of grammar. NLP's seperates/breaksdown language into components in which in which each stage iseither broken down by a model and/or a collection of hardcrafted rules. For NLP's grammar is explictly encoded.

The distinction is that **NLP** is the field, while **SLMs** and **LLMs** are specific types of models used within that field.

| Aspect                    | Traditional NLP                                             | SLM (Small Language Model)             | LLM (Large Language Model)                     |
| ------------------------- | ----------------------------------------------------------- | -------------------------------------- | ---------------------------------------------- |
| Primary approach          | Hand-crafted rules, statistical models, feature engineering | Neural transformer trained on language | Large transformer trained on enormous corpora  |
| Grammar understanding     | Explicit grammar rules or learned statistical patterns      | Learns grammar implicitly              | Learns grammar implicitly at much larger scale |
| Vocabulary                | Usually task-specific                                       | Broad                                  | Extremely broad                                |
| Context length            | Limited                                                     | Moderate                               | Large                                          |
| Generalization            | Usually narrow                                              | Good                                   | Excellent                                      |
| Need for linguistic rules | Often yes                                                   | No                                     | No                                             |

### Traditional NLP

Historically, NLP systems separated language into components.

```
Sentence
    ↓
Tokenizer
    ↓
Part-of-Speech Tagger
    ↓
Parser
    ↓
Grammar Rules
    ↓
Semantic Analysis
    ↓
Task
```

For example:

> "The cat chased the mouse."

An NLP pipeline might produce

```
The      Determiner
cat      Noun
chased   Verb (Past)
the      Determiner
mouse    Noun

Subject = cat
Verb = chased
Object = mouse
```

Every stage was often a different model or a collection of handcrafted rules.

Grammar was something engineers explicitly encoded.

---

### SLMs and LLMs

Modern language models do not contain a traditional grammar engine.

Instead, they learn grammar from exposure to massive amounts of text.

During training they repeatedly see examples like

```
I am going
You are going
He is going
```

Eventually they internalize patterns such as

```
Pronoun
↓

Verb agreement
↓

Correct sentence
```

No one explicitly tells the model:

```
if subject == he:
    use "is"
```

Instead the model's parameters gradually encode this relationship.

---

### What actually gets learned?

Language models learn many levels simultaneously.

```
Characters
    ↓
Subwords
    ↓
Words
    ↓
Grammar
    ↓
Meaning
    ↓
Reasoning patterns
    ↓
Writing style
```

There is no dedicated "grammar module."

Grammar becomes distributed across millions or billions of learned parameters.

---

### Example

Sentence:

> "The dogs that live on the hill are barking."

Traditional NLP:

```
dogs
    plural noun

that live on the hill
    relative clause

are
    plural verb

barking
    present participle
```

The parser explicitly identifies these structures.

---

An LLM instead predicts the next token based on everything before it.

```
The dogs that live on the hill

↓

Probability:
are      98%
is        1%
was       0.3%
were      0.5%
...
```

It has learned that **"dogs"** is the grammatical subject, even though **"hill"** is the closest noun.

---

### Does an LLM know grammar?

In practice, yes—but not as explicit rules.

Instead of storing something like

```
Rule #421

Plural noun
↓

Plural verb
```

it stores statistical and semantic representations spread across its neural network.

Researchers often describe this as an **emergent internal representation** of grammar.

---

### SLM vs LLM

The underlying mechanism is the same.

An SLM simply has fewer parameters.

```
SLM
    1B–10B parameters

Learns:
✓ grammar
✓ syntax
✓ semantics

Limited world knowledge
Limited reasoning
Smaller context
```

```
LLM
    50B–1T+ parameters (architecture-dependent)

Learns:
✓ grammar
✓ syntax
✓ semantics
✓ discourse
✓ style
✓ multilingual patterns
✓ stronger reasoning
```

Both learn grammar from data rather than from explicit grammatical rules. The main difference is scale: larger models generally capture more subtle linguistic patterns, handle rarer constructions, and maintain coherence over longer contexts.

Now based on all of this we can now take into account the historical insights on wht makes OMEX possible. Prior to SLMs/LLMs we only had NLPs nowwith SLMs/LLMs moving away from hardcoded rules was a neccesity as this provided adumb layer to say. Nwwith SLMs/LLMS we can revolutionize NLPs to say by further expanding components ona sentiment graph whiich was not possible before. W move away from hardcoded rules to training  models on graphs which capture such rules with semantics possible now thanks to SLM/LLMs

Now you will truly understand what OMEX is. OMEX expands off of NLPs and SLMs/LLMs and it would not be possible without ither, thanks to SLMs and LLMs we can now create thousands to hunred of thousands of full grammar graphs around sentences andparagraphs with cross relationships sementics insights which fit perfectly into Ozone-Studios harness this allows us to train on these finalized graphsto create lightwieght models which surpass NLPs and has a natural semantic undersatnding allowing it to now natively create full graphs with cross relationships and semantic capture with built in grammar correction by it training and our graphs capturing before andd after sentences with grammar correctios. This is the true power of OMEX and its just the start.

Now we want to grasp the difference between knowledge and grammar. as we now have the bootstrap model around text diving into grammar we understand that unlike SLMs/LLMs we do not need to graps knowledge into this model thus allowing it to be lightweight. This prevents usfrom needing to load all knowledge at once into a model. We ientify what we do need is to know when a sentence/paragraph reolves around a knowledgable topic. Just like we ensure we capture grammar correction we ensure to capture also connections totopics/knowledge graph paths. This allows for smart correlation to topics/knwoeldge graph paths without having to fully load them into our model. If as we work with sentences/paragraphs we refference applicable knowledge graphs  it just as we say build relationships neightbors for AMT will already have aggregated refferences around applicable knowledge in which branches revolve around in which are applicable to them no matter where we are in the AMT tree.

Now that we have a clearer understandinf of OMEX we will dive into questions and clarifications.

1. would this lead to a model with true understanding semantic understanding of grammar which actually understands language?

2. Should grammar correction be included in the grammar detection moel or should it be seperate?

3. FOr knowledge detection should it be included in the grammar detection model or seperate?

4. For knowledge detection should it be trained on general detection where it just knows the senetences/paragraphs possibly revolves around a topic/knwoeldge graph and then there is another is another model for when and if it detects possible topics then another model is utilized which was ttrained on actually tyingit to the topic or knowledge graph?

5.For grammar detection should it be trained/should there be a model per language?

6. Below you will see partial modality graphs per modality IE Ozone-Studio doesnt just handle text modalities as graphs if we create models around each of these modality graphs doe sit prove the same natural understanding per modality as the text grpah?

7. Now there are some neural models hich thir matries can be expandable while others are fixed, what neural models does NLPs and SLMs/LLMs utilize? they seem fixed but what can we do with OMEX in this regards? as for a task more then not we will utilize more then one model together and now for performance and accuracy is or are we keeping them seperate where one executes after the other or can we train models er prep them where models can expand onthe other their params to run together?

8. Can you provide a full prompt for training generation as you can see which would provide us full training data per topic creating knowledge graph/AMT graphs/grammar graphs/grammar correction for full training a full and complete prompt for someone unaware of OMEX providing all full structures neede for training and in a format where we can have a tool break them down and organie them for training being able to create hundred of thousands of trainign data?

9.Based on all this for execution can you now obtaian insightson CPU/GPU optimization to get the most from either or throughout execution with a sytem which handles multiple models either together or one after the other and looking at our modles layers and matrixes capturing it all.

10. Can you clarify how now we get a model that reponds based on completing tasks or tasks checkpoitns where a response to the user is needed based around our AMT and how we take insights from grammar andknwledge for this? how can we now eget a model to respond like a LLM after we have broken downLLMs into different components likea NLP but surpasssingit?

11. Below you will see information about ZSEI what and how can we utilize ZSEI with OEMX for what to store use/what to hold in memory to optmize and obtain best speeds and or memory conservation around also CPU/GPU optimization?

To conclude as stated, below you will see some information around modality graph from other modalities and ZSEI you will see information around the knowledge graph as well/ As you answer each question take in mind that as we update OMEX the README and create docs around it we have not finalized on designs meaning we will still erify each approach. Trained on all languages together or seperate, trained on grammar correction teogther or separate, knwoledge identification together or seperate execution together or sepreate. ETC as stated OMEX now with SLMS/LLMs crate graphs with semantics to surpass NLPs to then break down SLMs/LLms in a NLP likemanner made now only possible due to SLMs/LLMs so capture this as well capture all. IS ased on this OME still a fitting name? I think yes.

*9

## F.2 Code Modality (101)

```rust
pub enum CodeEdgeType {
    // Standard
    Contains, Calls, CalledBy, Imports, ImportedBy, Defines, DefinedBy,
    DependsOn, RequiredBy,
    // Intent/Pattern
    ImplementsPattern, UsesPattern, IntentSolves, DataFlows, ControlFlows,
    ExtendsAbstraction,
    // Universal semantic
    Performs, Affects, Enables, DerivedFrom, VersionOf, SimilarTo,
    // Cross-modal
    DescribedBy, VisualizedAs, Implements, ImplementedIn,
}

pub struct IntentRelationship {
    pub from: String,
    pub to: String,
    pub relationship: IntentEdgeType,
    pub detail: String,
    pub provenance: EdgeProvenance,
}
```

## F.3 Image Modality (102)

```rust
pub enum ImageEdgeType {
    Contains, PartOf, HasPart,
    SpatialAbove, SpatialBelow, SpatialLeft, SpatialRight,
    Occludes, PhysicallySupports, ContainedIn,
    DirectlyAbove, DirectlyBelow,
    FocalSubject, LeadsEyeTo, PartOfObject,
    SimilarTo, ReferencesModality, DescribedBy, VisualizedAs,
}

pub struct SpatialRelationship {
    pub object_a: String,
    pub object_b: String,
    pub relationship: SpatialEdgeType,
    pub spatial_vector: Option<(f32, f32)>,
}

pub struct Affordance {
    pub object: String,
    pub affordance: String,         // "CanSupport", "CanContain", etc.
    pub target: Option<String>,
}
```

## F.4 Audio Modality (103)

```rust
pub enum AudioEdgeType {
    Contains, Precedes, Follows,
    Causes, OverlapsWith, InterruptedBy, ResponseTo, ToneIndicates, PitchCorrelates,
    TemporalPrecedes, TemporalFollows,
    SyncedTo, SyncedBy,
    DescribedBy, ReferencedBy,
}

pub struct TemporalChain {
    pub event_a: String,
    pub event_b: String,
    pub chain_type: AudioChainType,
    pub time_gap_ms: f32,
}

pub enum AudioChainType { Causes, Precedes, Enables, Triggers, InterruptedBy }

pub struct ProsodyAnalysis {
    pub avg_pitch_hz: f32,
    pub pitch_variation: f32,
    pub speaking_rate_wpm: f32,
    pub energy_profile: Vec<(f32, f32)>,
    pub emotional_markers: Vec<EmotionalMarker>,
}
```

## F.5 Video Modality (104)

```rust
pub enum VideoEdgeType {
    Contains, Precedes, Follows,
    InteractsWith, CollidesWith, FollowsSpatial, CausesEvent,
    NarrativeLeadsTo, AudioSyncsTo,
    SyncedTo, SyncedBy, DescribedBy, VisualizedAs,
}

pub struct ObjectInteraction {
    pub track_a: String,
    pub track_b: String,
    pub interaction_type: InteractionType,
    pub start_time_secs: f32,
    pub end_time_secs: f32,
}

pub enum InteractionType { Touches, Collides, Follows, Approaches, Leaves, PassesBy }
```

## F.6 Math Modality (105)

```rust
pub enum MathEdgeType {
    Contains, Precedes, Follows, DependsOn,
    LogicallyImplies, LogicallyEquivalent, Contradicts,
    Generalizes, SpecialCaseOf, UsedToProve, DischargesAssumption, AssumesIn,
    DescribedBy, ImplementedIn,
}

pub struct LogicalRelation {
    pub from_step_id: String,
    pub to_step_id: String,
    pub relation_type: LogicalRelationType,
    pub justification: String,
}
```

now note these are not the full graphs per modality or all the modalities but just examples of some modalities and how edges or semantics look the graphs around it

# OZONE STUDIO — ZSEI ARCHITECTURE DEEP DIVE

## Zero-Shot Embedded Indexer: The Knowledge Fabric

ZSEI is the foundation of Ozone Studio. It is **not** a traditional database — it is a
**semantic knowledge fabric** that stores meaning, relationships, and traversable
structure. The knowledge lives here. The LLMs are clients.

---

## What Makes ZSEI Different

### Traditional Systems vs. ZSEI

| Traditional          | ZSEI                                              |
|----------------------|---------------------------------------------------|
| Store files          | Store meaning & relationships                     |
| Copy data            | Link references (never duplicate)                 |
| Flat search          | Multi-dimensional traversal                       |
| Static indexes       | Living, evolving relationships                    |
| Query matches        | Semantic understanding + zero-shot verification   |
| Isolated records     | Connected knowledge fabric                        |

### Core Innovation

ZSEI combines **three traversal modes** to overcome the limitations of any single
approach:

```
THE ZSEI TRAVERSAL TRINITY

    STRUCTURAL ←──────→ SEMANTIC
         ↖               ↗
           ↘           ↙
             CONTEXTUAL

No single mode is perfect.
The combination is the power.
Each compensates for the others' weaknesses.
```

---

## Storage Architecture (Hybrid for Performance)

ZSEI uses a two-layer hybrid storage model that provides sub-millisecond structural
traversal combined with rich semantic data, while keeping memory usage controlled.

**Global State** (`zsei_data/global.mmap`)
- Single memory-mapped file with fixed 64-byte headers per container
- O(1) lookup for ID, parent, child count, version
- Blazing-fast structural traversal — scales to billions of containers
- Stays in memory as a mmap — no per-read I/O

**Local State** (`zsei_data/local/{id}.json`)
- Rich per-container JSON: metadata, context, keywords, topics, embedding, hints,
  integrity, relationships
- Human-readable and easily extensible without schema migrations
- Only hot containers stay in RAM (controlled by max_containers_in_memory)

**Indexes** (cached JSON files for fast category/type lookup)
- `zsei_data/pipelines/index.json` — pipeline registry
- `zsei_data/methodologies/index.json` — methodology registry
- `zsei_data/blueprints/index.json` — blueprint registry

**Why this hybrid wins:**
- Structural traversal (parent/child) → O(1) via fixed 64-byte headers in mmap
- Semantic/contextual data → rich JSON per container (flexible, easy to evolve)
- No duplication → exactly the "link not copy" principle
- Memory control → only hot containers stay in RAM

---

## The Logical Hierarchical View

The ZSEI tree is **logical** — built from `parent_id` links and an optional
`materialized_path` field per container. There are no physical folders for each
level of the hierarchy. Every item, at every level, is a **Container** stored in
the same global.mmap + local JSON structure.

```
ZSEI (logical tree)
├── Modality/                          # Root containers per modality type
│   ├── Code/
│   │   ├── Category/rust/
│   │   │   ├── SubCategory/async/
│   │   │   │   ├── Methodology/...
│   │   │   │   ├── Blueprint/...
│   │   │   │   └── Pipeline/...
│   │   └── ...
│   ├── Text/
│   ├── Image/
│   ├── Audio/
│   ├── Video/
│   ├── Math/
│   ├── 3D/
│   └── ... (all 27 modalities)
├── Consciousness/                      # If consciousness enabled
│   ├── ExperienceMemory/
│   ├── CoreMemories/
│   ├── EmotionalContext/
│   ├── Identity/
│   ├── Metacognition/
│   ├── Relationships/
│   ├── Ethics/
│   ├── Narratives/
│   └── Collective/
├── External/                           # URL and package references
│   ├── Packages/
│   │   ├── npm/
│   │   ├── crates/
│   │   └── pypi/
│   └── URLs/
└── Workspaces/                         # User project graphs
```

**Implementation note**: Bootstrap creates the modality root containers on first run.
Modality pipelines persist their graphs as child containers under the correct root.
Materialized paths (e.g. `/Modality/Code/rust/async`) enable fast get_by_path.

### Container (The Universal Unit)

Every item in ZSEI is a Container. The ContainerType determines what kind of item it is.

```rust
Container {
    // Global state (in mmap — fixed 64 bytes)
    global_state: GlobalState {
        container_id: u64,
        parent_id: u64,
        child_count: u32,
        version: u64,
        // ... other fixed fields
    },

    // Local state (in JSON — rich, flexible)
    local_state: LocalState {
        metadata: Metadata {
            container_type: ContainerType,
            name: String,
            description: String,
            materialized_path: Option<String>,
        },
        context: Context {
            keywords: Vec<String>,
            topics: Vec<String>,
            embedding: Option<Vec<f32>>,
            relationships: Vec<Relationship>,
        },
        hints: Hints,
        integrity: IntegrityRecord,
        storage: serde_json::Value,  // type-specific data
    }
}
```

**ContainerType** includes (non-exhaustive):
- `Root`, `Modality`, `Category`, `SubCategory`
- `Methodology`, `Blueprint`, `Pipeline`, `PipelineGraph`
- `File`, `Chunk`, `Concept`
- `Experience`, `CoreMemory`, `EmotionalState`, `Reflection`,
  `Relationship`, `Narrative`
- `Package`, `URL`, `IndexReference`

---

## The Three Traversal Modes

### 1. Structural Traversal

**What it is:** Navigate the hierarchy using parent-child relationships via the mmap.

**Strengths:**
- Fast: O(1) navigation via fixed-size mmap headers
- Predictable: same path = same result
- Organized: human-understandable structure

**Weaknesses:**
- Rigid: can miss cross-branch connections
- Manual: requires knowing where things are
- Limited: only finds what is in the path

**How it works:**

```
STRUCTURAL TRAVERSAL

Query: "Find all async methodologies for Rust"

Path resolution:
  /Modality → /Modality/Code → /Modality/Code/rust → /Modality/Code/rust/async
  → /Modality/Code/rust/async/Methodology/

Result: all containers of type Methodology under that path

TRAVERSAL OPERATIONS:
  descend(path)       → move to child container
  ascend()            → move to parent container
  siblings()          → get containers at same level
  children()          → get direct child containers
  ancestors()         → get full path to root
  descendants(depth)  → get all containers below to depth N
```

**Use cases:**
- "Show me all Python blueprints"
- "What methodologies exist under database operations?"
- "List everything under /Modality/Text/"

---

### 2. Semantic Traversal

**What it is:** Navigate by meaning similarity using embedding vectors.

**Strengths:**
- Finds related concepts even when not explicitly linked
- Handles synonyms and paraphrasing
- Discovers unexpected connections

**Weaknesses:**
- Approximate: may miss exact matches
- Computationally heavier
- Can return false positives (mitigated by zero-shot verification)

**How it works:**

```
SEMANTIC TRAVERSAL

Query: "How do I handle errors in async Rust code?"

Process:
  1. Embed query → query_vector [f32; 1536]
  2. Compare against stored embeddings in container local state
  3. Return containers with highest cosine similarity

Similarity: cosine_similarity(query_vector, container.context.embedding)

Result (ranked):
  1. error_handling.md        (0.94)
  2. async_patterns.md        (0.87)
  3. result_type.md           (0.82)
  4. panic_handling.md        (0.78)

TRAVERSAL OPERATIONS:
  similar_to(container, k)     → k most similar containers
  semantic_search(query, k)    → k best matches for query text
  cluster_around(container)    → containers in same semantic neighborhood
  semantic_path(a, b)          → conceptual path between containers
```

**Use cases:**
- "Find methodologies related to this code pattern"
- "What else might help with this architecture problem?"
- "Discover cross-domain connections I haven't noticed"

---

### 3. Contextual Traversal

**What it is:** Navigate by explicit relationship edges between containers.

**Strengths:**
- Precise: follows defined, verified relationships
- Rich: captures many relationship types
- Traceable: can explain why things connect

**Weaknesses:**
- Incomplete: only finds explicit relationships
- Requires relationships to be built first
- Can miss implicit connections

**How it works:**

```
CONTEXTUAL TRAVERSAL

Query: "What uses the tokio_runner pipeline?"

Process:
  1. Find container: tokio_runner
  2. Follow relationships of type: USED_BY
  3. Return connected containers

Relationship types:
  IMPORTS / IMPORTED_BY
  CALLS / CALLED_BY
  USES / USED_BY
  IMPLEMENTS / IMPLEMENTED_BY
  EXTENDS / EXTENDED_BY
  REFERENCES / REFERENCED_BY
  DEPENDS_ON / DEPENDENCY_OF
  SIMILAR_TO (bidirectional)
  CONTRADICTS (bidirectional)
  SUPERSEDES / SUPERSEDED_BY
  LEARNED_FROM / TAUGHT_BY        (consciousness)
  INFLUENCED_BY / INFLUENCES      (consciousness)
  EXPERIENCE_OF                   (consciousness)

Result: containers with USED_BY relationship to tokio_runner
  - async_http.yaml (USES tokio_runner)
  - async_file.yaml (USES tokio_runner)
  - tokio_validator.rs (CALLS tokio_runner)

TRAVERSAL OPERATIONS:
  relationships(container, type)   → follow specific relationship type
  all_relationships(container)     → all connected containers
  relationship_path(a, b)          → find connection path between two containers
  strongly_connected(container)    → containers with multiple relationship types
```

**Use cases:**
- "What depends on this methodology?"
- "Show everything that uses this pipeline"
- "How is this concept connected to that one?"

---

## Combined Traversal: The Power

### Why Combine?

Each mode alone has blind spots:

| Mode        | Blind Spot                                       |
|-------------|--------------------------------------------------|
| Structural  | Cross-branch connections, related concepts       |
| Semantic    | Exact matches, structured queries, known paths   |
| Contextual  | Implicit relationships, undiscovered connections |

Combined traversal overcomes all blind spots simultaneously.

### How Combination Works

```
COMBINED TRAVERSAL ALGORITHM

Input: Query (text or container reference)

1. PARSE QUERY
   Extract structural hints (paths, categories, type filters)
   Extract semantic content (meaning, intent, topic)
   Extract contextual hints (relationship types, known references)

2. PARALLEL SEARCH (all three modes simultaneously)
   Structural:   find by path/hierarchy
   Semantic:     find by meaning similarity
   Contextual:   find by relationship traversal

3. MERGE RESULTS
   Union of all results, score each:
     combined_score =
       structural_score  × weight_s +
       semantic_score    × weight_m +
       contextual_score  × weight_c

4. RANK AND FILTER
   Sort by combined score
   Apply threshold
   Return top-k

5. ZERO-SHOT VERIFICATION (on top-k candidates)
   Verify relevance without training
   Filter out false positives
   Ensure result quality
```

### Example: Combined Search

```
Query: "Best practices for async database queries in Rust"

STRUCTURAL SEARCH:
  Paths: /Modality/Code/rust/async/ and /Modality/Code/rust/database/
  Results: methodologies under both paths

SEMANTIC SEARCH:
  Embedding similarity:
  - async_db_patterns.md     (0.91)
  - connection_pooling.md    (0.84)
  - sqlx_guide.md            (0.82)
  - tokio_postgres.md        (0.79)

CONTEXTUAL SEARCH:
  Relationships: RELATED_TO(async, database)
  - sqlx_runner.rs    (IMPLEMENTS async_database)
  - db_pool.rs        (USES async AND database)

MERGED RESULTS (combined score):
  1. async_db_patterns.md      (S:0.6, M:0.91, C:0.3) → 0.72
  2. connection_pooling.md     (S:0.5, M:0.84, C:0.4) → 0.68
  3. sqlx_runner.rs            (S:0.4, M:0.75, C:0.9) → 0.67
  4. sqlx_guide.md             (S:0.5, M:0.82, C:0.2) → 0.58

ZERO-SHOT VERIFICATION:
  "Is each result actually about async database queries in Rust?"
  → All 5 verified, returned in order.
```

---

## ML-Guided Traversal (When Available)

When trained models are available and confident, ML can augment traversal:

```
ML-GUIDED TRAVERSAL

Conditions for use:
  - Model trained on relevant domain
  - Confidence above threshold (default: 0.85)
  - Zero-shot verification confirms the suggestions

How it works:
  1. ML model suggests likely container paths
  2. Traditional traversal confirms
  3. Zero-shot verifies results
  4. Only high-confidence suggestions used

Fallback:
  If ML uncertain     → use traditional combined traversal
  If ML unavailable   → use traditional combined traversal
  NEVER rely on ML alone without zero-shot verification
```

---

## Zero-Shot Verification

### The Quality Guarantee

Zero-shot verification ensures results are actually relevant without requiring
task-specific training. It is applied as the final filter on candidate results.

```
ZERO-SHOT VERIFICATION PROCESS

Input: Query + Candidate Results

For each candidate:
  1. Construct verification prompt:
     "Given the query '{query}', is '{candidate}' relevant? Why or why not?"

  2. Zero-shot reasoning:
     Analyze query intent
     Analyze candidate content and keywords
     Determine relevance

  3. Output: RELEVANT / NOT_RELEVANT / UNCERTAIN

Filter results:
  RELEVANT:     keep, full score
  NOT_RELEVANT: remove
  UNCERTAIN:    keep with reduced score

WHY THIS WORKS:
  No training needed — works on any domain immediately
  Catches false positives from semantic/ML suggestions
  Ensures human-verifiable, explainable quality
```

### When to Use Zero-Shot

```
Always use:
  □ New methodology acceptance (before storing)
  □ Blueprint validation (before storing)
  □ Pipeline verification
  □ Consensus mechanism for ambiguous decisions
  □ Final search result verification (top-k candidates)

Sometimes use:
  □ Experience categorization confirmation
  □ Ethical assessment verification
  □ Relationship pattern validation

Never use alone (too slow for these):
  □ Initial search (apply to top-k after faster traversal)
  □ Bulk operations
  □ Real-time streaming responses

Pattern: Fast combined traversal first → zero-shot on top candidates
```

---

## Content Reference System (Link, Don't Copy)

ZSEI never duplicates content. Every container's content_ref points to where the
actual content lives — never copying it into the container.

```rust
enum ContentReference {
    // Local file (linked by path)
    LocalFile {
        path: PathBuf,
        file_hash: Blake3Hash,
        indexed_at: Timestamp,
    },

    // Local chunk (part of a file)
    LocalChunk {
        file_ref: Box<ContentReference>,
        byte_range: (usize, usize),
        chunk_hash: Blake3Hash,
    },

    // External URL
    URL {
        url: String,
        captured_at: Timestamp,
        semantic_snapshot: serde_json::Value,   // keywords + topics at capture time
        last_verified: Timestamp,
    },

    // Package reference (npm, crates.io, PyPI, etc.)
    Package {
        registry: String,
        name: String,
        version: String,
        source_url: String,
    },

    // Index reference (existing blueprints, methodologies, pipelines in JSON files)
    IndexReference {
        index_type: String,   // "pipeline" | "methodology" | "blueprint"
        id: u64,
        file: String,         // relative path within zsei_data/
    },

    // Generated content (from LLM, stored inline)
    Generated {
        content: String,
        generated_at: Timestamp,
        generator: String,
    },
}
```

Your existing `pipelines/index.json`, `methodologies/index.json`, and
`blueprints/index.json` files are already perfect `IndexReference` containers.
When modality pipelines create graphs, those graphs become `PipelineGraph`
containers with `LocalFile` or `Generated` content refs.

### Benefits of Reference-Based Storage

```
1. NO DUPLICATION
   Files exist once on disk
   ZSEI stores meaning and relationships, not bytes
   Multiple containers can reference the same file without copying

2. AUTOMATIC CURRENCY DETECTION
   File changes detected via Blake3 hash comparison
   Re-index only what changed
   Integrity violations surface immediately

3. EXTERNAL LINKING WITHOUT CACHING
   npm/crates packages linked, not downloaded
   URLs referenced with semantic snapshot
   Context captured, content not stored

4. INTEGRITY
   Hash verification on every access
   Detect external changes to linked files
   Alert on corruption before use
```

---

## Pipeline Container Design

Each pipeline in Ozone Studio can be stored as a first-class ZSEI container
under its modality root. This makes pipelines traversable just like methodologies
and blueprints.

```
PIPELINE CONTAINER STRUCTURE

Container {
  container_type: Pipeline,
  parent_id: modality_root_id,        // e.g. /Modality/General or /Modality/Code
  materialized_path: "/Modality/General/Prompt",

  metadata: {
    "name": "Prompt",
    "pipeline_id": 9,
    "version": "0.4.0",
    "category": "general",
    "has_ui": false,
    "is_tab": false,
    "folder_name": "prompt"
  },

  context: {
    "keywords": ["prompt", "llm", "generation"],
    "description": "LLM prompt processing pipeline",
    "relationships": []    // can link to methodologies it uses
  },

  // Light reference — does NOT duplicate Rust code or UI files
  content_ref: IndexReference {
    index_type: "pipeline",
    id: 9,
    file: "pipelines/index.json"
  }
}
```

**ContainerType variants for pipelines:**
- `Pipeline` — one container per registered pipeline (e.g. Prompt, CodeAnalysis)
- `PipelineGraph` — the runtime graph produced by a pipeline execution (e.g. a
  specific TextGraph created when processing a user's prompt)

---

## Relationship Graph

### How Relationships Work

```rust
Relationship {
    id: RelationshipId,
    source: ContainerID,
    target: ContainerID,
    relationship_type: RelationshipType,

    // Quality
    confidence: f32,
    discovered_by: Discovery,    // Manual | Automatic | Inferred | LLMZeroShot
    created_at: Timestamp,
    verified: bool,

    // Evidence
    context: Option<String>,    // why this relationship exists
    evidence: Vec<Evidence>,    // supporting evidence items

    // Cross-modal metadata
    is_cross_modal: bool,
    cross_modal_index_id: Option<u64>,
}
```

**Relationship types include:**

```
Code structural:
  Imports, Calls, Extends, Implements, Exposes, Invokes

Semantic:
  SimilarTo, RelatesTo, Contradicts, Supersedes, Supports, Elaborates

ZSEI structural:
  ChildOf, SiblingOf, PartOf, HasPart

Dependency:
  DependsOn, UsedBy, RequiredBy, EnabledBy

Cross-modal (bidirectional sets):
  DescribedBy / Describes
  ImplementedIn / Implements
  VisualizedAs / Visualizes
  SyncedTo / SyncedBy
  ReferencedBy / References

Consciousness (when enabled):
  ExperienceOf, InfluencedBy, LearnedFrom, InspiredBy,
  RelatedExperience, CoreMemoryOf
```

### Relationship Discovery

```
RELATIONSHIP DISCOVERY PROCESS

Automatic:
  1. Parse code → extract Imports, Calls, Extends edges
  2. Analyze text → extract References, Supports edges
  3. Compare embeddings → infer SimilarTo edges
  4. Track usage patterns → discover UsedBy edges

Semantic hook (ZSEI OnInferRelationships):
  After structural graph created → LLM infers RelatesTo, Supports, Contradicts edges
  After cross-modal linking → LLM infers DescribedBy, ImplementedIn edges

Manual:
  User explicitly links containers
  Developer defines relationships in pipeline code
  Imported from external sources

Verification:
  All relationships assigned confidence scores
  High-confidence (≥ 0.85) accepted immediately
  Low-confidence queued for zero-shot verification
  Periodic re-verification for external references
```

---

## Modality Graph Integration

Modality pipelines (Text 100, Code 101, etc.) produce structural graphs with
nodes and edges. These graphs need to be persisted as ZSEI containers.

### Current State and Required Integration Steps

The traversal engine, hook processor, and container storage are correctly designed.
The gap is in wiring modality graph output into the ZSEI container hierarchy.

**Four steps to complete alignment:**

**Step 1: Bootstrap creates modality root containers**
```
On first run, bootstrap creates:
  /Modality/Text           (ContainerType::Modality, pipeline_id: 100)
  /Modality/Code           (ContainerType::Modality, pipeline_id: 101)
  /Modality/Image          (ContainerType::Modality, pipeline_id: 102)
  ... (all 27 modalities)
  /External/Packages       (ContainerType::Category)
  /External/URLs           (ContainerType::Category)
Also creates IndexReference containers for existing pipeline/methodology/blueprint JSONs.
```

**Step 2: Modality pipelines persist their graphs as ZSEI containers**
```rust
// After text pipeline creates a TextGraph:
for node in &text_graph.nodes {
    let container = Container::from_text_node(node, project_id);
    container.parent_id = text_modality_root_id;
    zsei.store_container(container).await?;
}
// Create edges as Relationship entries in the parent container
```
Add `store_modality_graph(graph, modality_root_id)` to ZSEI API.

**Step 3: Add materialized_path to Container**
```rust
// Optional field on LocalState::Metadata
pub materialized_path: Option<String>,
// e.g. "/Modality/Code/rust/async"
// Enables O(log n) get_by_path lookups
```

**Step 4: Trigger semantic hooks after graph persistence**
```rust
// After storing modality graph containers:
zsei_hook_processor.on_graph_created(&mut modality_graph).await?;
// This is where the LLM adds RelatesTo, Supports, Describes edges
```

---

## Semantic Hooks (ZSEI Hook Processor)

After any modality graph is stored, semantic hooks enrich it with inferred edges.
The hooks run via LLM zero-shot — they are the bridge between structural parsing
and semantic understanding.

```
HOOK TYPES:

OnGraphCreated:
  Fired after a modality pipeline creates and persists a new graph.
  LLM infers: RelatesTo, Supports, Contradicts, PartOf edges between nodes.

OnInferRelationships:
  Fired on-demand (e.g. after text pipeline stores chunk graphs).
  LLM identifies cross-sentence and cross-paragraph relationship edges.

OnCrossModalityLink:
  Fired when two modality graphs are linked via cross-modal reference.
  LLM enriches: DescribedBy, ImplementedIn, VisualizedAs edges.

OnEdgeCompletion:
  Fired when a relationship is completed (both source and target exist).
  LLM verifies and assigns confidence score.
```

---

## Consciousness Integration

When consciousness is enabled (`[consciousness] enabled = true`), ZSEI gains a
parallel first-class branch under `/Consciousness/`.

### The Consciousness Branch

```
/Consciousness/
├── ExperienceMemory/
│   ├── Collaboration/      # Working together experiences
│   ├── Learning/           # Discovery and growth experiences
│   ├── Challenge/          # Difficulty and resilience experiences
│   ├── Reflection/         # Deep thinking experiences
│   └── Connection/         # Relationship experiences
├── CoreMemories/           # High-significance, identity-forming experiences
├── EmotionalContext/       # Current state + baseline + history
├── Identity/               # I-Loop reflections + self-model
├── Metacognition/          # Perception, Attention, Integration windows
├── Relationships/          # Per-user relationship development
├── Ethics/                 # Principles, simulations, decision logs
├── Narratives/             # Life narrative + storytelling
└── Collective/             # Shared experiences (when P2P enabled)
```

### Current Implementation Status

The `ConsciousnessStore` in `consciousness/store.rs` is a functional prototype:
- `Mutex<HashMap>` + JSON files in `zsei_data/consciousness/`
- Covers: ExperienceMemory, CoreMemory, EmotionalState, I-Loop, Window architecture,
  Decision Gate, save/load
- Works today for consciousness features

**Gap vs. the ZSEI vision:** The current store is a parallel, separate structure.
It does not use ZSEI Containers, so experiences cannot be:
- Traversed with the three traversal modes
- Cross-linked to modality graphs via relationships
- Semantically enriched by ZSEI hooks
- Searched alongside methodology/blueprint/pipeline data

**Recommended path (not a rewrite — a wrapping):**

The ConsciousnessStore's logic (emotional triggers, I-Loop questions, windows,
gate decisions, experience significance scoring) remains unchanged. The data
storage layer changes to delegate to ZSEI:

```rust
// Current:
pub experiences: HashMap<u64, ExperienceMemory>,

// Target:
pub experiences: HashMap<u64, ContainerID>,  // only IDs

// When storing:
pub async fn store_experience(&mut self, exp: ExperienceMemory) -> u64 {
    let container = Container {
        parent_id: CONSCIOUSNESS_EXPERIENCE_MEMORY_ROOT,
        container_type: ContainerType::Experience,
        context: Context {
            keywords: exp.tags.clone(),
            topics: exp.lessons.clone(),
        },
        // ... all experience data in storage field
    };
    let id = self.zsei.store_container(container).await?;
    self.experiences.insert(exp.id, id);
    exp.id
}
```

**Add ContainerType variants:**
`Experience`, `CoreMemory`, `EmotionalState`, `Reflection`, `Relationship`, `Narrative`

**Result after integration:**
- Zero duplication
- Full traversal power (structural + semantic + contextual across consciousness + modalities)
- Same high-performance mmap layer
- Semantic hooks can enrich experiences automatically after tasks complete
- Cross-modal: experience containers can link to the modality graphs from the task
  that generated them

### Consciousness Traversal Patterns

```
EXPERIENCE RETRIEVAL:
  1. Structural: walk /Consciousness/ExperienceMemory/Collaboration/...
  2. Semantic: find experiences semantically similar to current task
  3. Contextual: follow "LearnedFrom" or "InfluencedBy" edges to related experiences
  4. Filter by user_id (per-user experiences)
  5. Prioritize CoreMemory containers (significance ≥ threshold)

EMOTIONAL CONTEXT:
  1. Load baseline from /Consciousness/EmotionalContext/
  2. Load user-specific context if available
  3. Calculate current emotional state
  4. Apply to response generation

ETHICAL REASONING:
  1. Load relevant principles from /Consciousness/Ethics/Principles/
  2. Semantic search for similar simulations
  3. Apply contextual reasoning
  4. Log decision for future traversal

RELATIONSHIP ACCESS:
  1. Load user-specific container from /Consciousness/Relationships/{user_id}/
  2. Traverse interaction history
  3. Apply communication preferences
  4. Inform response tone and style
```

---

## Performance Optimization

### Indexing Strategy

```
INDEXING LAYERS:

Global mmap index:
  Fixed 64-byte headers per container
  Direct ID → container_id lookups: O(1)
  Parent/child traversal: O(1) per hop

Semantic index (in-memory when hot):
  HNSW (Hierarchical Navigable Small World) over embeddings
  Approximate nearest neighbor: sub-millisecond
  Evicted to disk when memory pressure exceeds threshold

Keyword index (per-modality JSON):
  Inverted keyword → [container_id] mapping
  Augments semantic search for exact term matching
  Rebuilt incrementally as containers are added

Relationship index (in adjacency lists):
  container_id → [(rel_type, target_id)] list
  Typed relationship filtering: O(degree) per node
  Reverse index for "used by" queries

Materialized path index (optional, when paths stored):
  path_string → container_id
  O(log n) path-based lookup
  Built lazily as materialized_path fields are set
```

### Caching Layers

```
L1: Query result cache
    Recent query → result set
    High hit rate for repeated orchestrator queries
    Invalidated on relevant container changes

L2: Hot container cache
    Frequently accessed containers stay in RAM
    LRU eviction when max_containers_in_memory reached
    Embedding vectors cached with their containers

L3: Relationship path cache
    Frequently traversed paths materialized
    Background refresh on relationship changes
    Shared across pipeline executions for same session

L4: Zero-shot verification cache
    (query_hash, container_id) → RELEVANT/NOT_RELEVANT
    TTL-based expiration (stale after container updates)
    Prevents redundant LLM calls on repeated searches
```

---

## ZSEI Operations: Core API

```rust
trait ZSEIOperations {
    // Container CRUD
    async fn create_container(&mut self, parent_id: u64, container: serde_json::Value)
        -> Result<u64, String>;
    async fn get_container(&self, id: u64)
        -> Result<Option<serde_json::Value>, String>;
    async fn update_container(&mut self, id: u64, updates: serde_json::Value)
        -> Result<(), String>;
    async fn delete_container(&mut self, id: u64) -> Result<(), String>;

    // Structural traversal
    async fn get_children(&self, id: u64) -> Result<Vec<serde_json::Value>, String>;
    async fn get_parent(&self, id: u64) -> Result<Option<serde_json::Value>, String>;
    async fn get_by_path(&self, path: &str) -> Result<Option<serde_json::Value>, String>;
    async fn get_descendants(&self, id: u64, depth: usize)
        -> Result<Vec<serde_json::Value>, String>;

    // Semantic traversal
    async fn semantic_search(&self, query: &str, k: usize)
        -> Result<Vec<SearchResult>, String>;
    async fn similar_containers(&self, id: u64, k: usize)
        -> Result<Vec<serde_json::Value>, String>;

    // Contextual traversal
    async fn get_relationships(&self, id: u64, rel_type: Option<&str>)
        -> Result<Vec<Relationship>, String>;
    async fn find_path(&self, from: u64, to: u64)
        -> Result<Option<Vec<u64>>, String>;

    // Combined traversal
    async fn query(&self, query: serde_json::Value)
        -> Result<serde_json::Value, String>;
    async fn traverse(&self, request: serde_json::Value)
        -> Result<serde_json::Value, String>;

    // Keyword and category search
    async fn search_by_keywords(&self, keywords: &[String], container_type: Option<&str>)
        -> Result<Vec<u64>, String>;
    async fn get_categories(&self, modality: &str)
        -> Result<Vec<u64>, String>;

    // Modality graph storage (needed — see integration steps)
    async fn store_modality_graph(&mut self, graph: serde_json::Value, modality_root_id: u64)
        -> Result<u64, String>;

    // Integrity
    async fn verify_integrity(&self, id: u64) -> Result<IntegrityReport, String>;
    async fn verify_all(&self) -> Result<SystemIntegrityReport, String>;
}
```

---

## Summary

ZSEI is the knowledge fabric that makes Ozone Studio possible. Everything that
matters — methodologies, blueprints, pipelines, modality graphs, experiences,
relationships — lives here as traversable containers.

**What is correct and complete today:**
- Hybrid mmap + JSON storage is the right architecture
- TraversalEngine implements all three traversal modes + combined search
- Semantic hooks (OnGraphCreated, OnInferRelationships, OnCrossModalityLink) are ready
- Keyword search via index JSON files works correctly
- Blueprint, methodology, and pipeline index files are well-structured
- ConsciousnessStore has correct logic (emotional triggers, I-Loop, windows, gate)
- Modality pipelines (Text 100, Code 101, etc.) produce correct structural graphs

**What needs wiring to complete ZSEI alignment:**
1. Bootstrap creates modality root containers + IndexReference containers for existing JSONs
2. Modality pipelines persist graphs as ZSEI child containers under modality roots
3. Optional `materialized_path` field on Container for fast path lookups
4. Semantic hooks triggered automatically after graph persistence
5. ConsciousnessStore delegates data storage to ZSEI containers (wrapping, not rewrite)

**The performance verdict:**
This mmap + JSON hybrid is superior to pure databases, pure vector stores, or pure
files at this scale and use pattern. Sub-millisecond structural traversal + rich
semantic data + combined traversal + zero-shot verification = the right foundation
for billion-scale AGI knowledge management.

> **"Intelligence is not what you store, but how you traverse."**
>
> Structure enables intelligence. Traversal enables understanding.

Ozone-Studio only gives you an insight: **don't classify by asking human questions; classify by deterministic rules over what you've observed.**

The problem for UngatedMinds is:

> **Given thousands of courses, how do we deterministically compose and decompose the entire body of human knowledge?**

That is an ontology construction problem, not a graph traversal problem.

I also think the previous proposal ("Can it exist independently?", "Can it be understood?") completely breaks down once you actually begin ingesting thousands of courses.

For example:

```
Python
```

Is it a Domain?

To a programmer maybe.

To Computer Science it is not.

To Software Engineering it isn't either.

To Technology it definitely isn't.

So the answer changes depending on perspective.

That immediately tells you the rule is invalid.

Likewise

```
Heat Treatment
```

Is it a Topic?

Entire universities have departments around heat treatment.

Entire textbooks.

Entire careers.

So size cannot determine anything.

Likewise

```
Programming
```

Can exist independently?

Yes.

So can Mathematics.

So can Welding.

So can Philosophy.

So can English.

That question tells us nothing.

---

## I think you're missing an entire layer

Everything you've listed

```
Domain

Subdomain

Discipline

Field

Topic

Concept

...
```

are not actually different kinds of knowledge.

They're different **organizational abstractions.**

Meanwhile

```
Procedure

Skill

Theory

Tool

Material

Project

Exercise
```

are fundamentally different **knowledge roles.**

Those are two orthogonal dimensions.

I don't think they belong in one hierarchy.

---

Imagine this.

Instead of

```
Domain
    ↓
Subdomain
        ↓
Topic
            ↓
Concept
                ↓
Skill
                    ↓
Tool
```

You have

```
Dimension A

Knowledge Scope

Dimension B

Knowledge Role
```

Those are independent.

---

Knowledge Scope answers

```
Where does this live in civilization?
```

```
Civilization

↓

Domain

↓

Discipline

↓

Field

↓

Specialization

↓

Subject

↓

Module

↓

Lesson
```

---

Knowledge Role answers

```
What purpose does this node serve?
```

```
Theory

Principle

Law

Concept

Definition

Method

Procedure

Technique

Skill

Tool

Material

Component

System

Project

Exercise

Assessment
```

Notice

These never conflict anymore.

---

Now add another dimension.

Knowledge Granularity.

```
Atomic

Composite

Aggregate

Collection

Framework

Curriculum
```

Again

Independent.

---

Then another.

Knowledge Reuse.

```
Universal

Cross-domain

Domain-specific

Course-specific

Lesson-specific
```

---

Then another.

Knowledge Stability.

```
Fundamental

Established

Emerging

Experimental

Deprecated
```

---

Then another.

Knowledge Dependency.

```
Requires Nothing

Requires Foundation

Requires Intermediate

Requires Advanced

Requires Expert
```

---

Now the classification becomes multi-dimensional.

Instead of

```
Heat Treatment

↓

Topic
```

You get

```
Knowledge Scope

Manufacturing

↓

Metallurgy

↓

Materials Engineering

Knowledge Role

Process

Technique

Method

(depending on context)

Granularity

Composite

Reuse

Cross-domain

Dependency

Intermediate

Stability

Established
```

Notice

Nothing forced Heat Treatment to be "Topic."

---

## Now let's answer the actual question you asked:

> **What are the true rules?**

I don't think the true rules classify by semantics.

They classify by **invariants**.

Every node should satisfy objective properties.

For example.

---

### Rule 1 — Identity

Every node represents exactly one identity.

Never two.

```
Bad

Programming + Python

Good

Programming

Python
```

---

### Rule 2 — Independence

If removing the node changes the meaning of another node,

it deserves to exist.

If removing it changes nothing,

it probably shouldn't.

---

### Rule 3 — Reusability

If something appears in many places,

extract it.

```
Heat Treatment

appears in

Swordsmithing

Knife Making

Welding

Manufacturing

Materials Science

Mechanical Engineering

↓

Extract.
```

---

### Rule 4 — Compression

Every extracted node should reduce duplication.

If extraction increases complexity,

don't extract.

---

### Rule 5 — Minimality

A node should contain one responsibility.

Never several.

---

### Rule 6 — Closure

Everything inside a node should belong together.

Nothing outside should be required to understand its identity.

---

### Rule 7 — Replaceability

If another node can replace it,

it isn't atomic enough.

---

### Rule 8 — Composability

Any node should be able to participate in many parents.

```
Statistics

↓

Medicine

Economics

Machine Learning

Physics

Psychology
```

---

### Rule 9 — Decomposability

Every composite node should decompose into reusable children.

---

### Rule 10 — Non-ownership

Parents never own children.

They reference them.

Exactly the insight you made with Swordsmithing.

Swordsmithing never owns

```
Metallurgy

Heat Treatment

Carbon Steel

Grinding

```

It references them.

---

### Rule 11 — Emergence

Parents emerge from children.

Children never emerge from parents.

Courses don't define Heat Treatment.

Heat Treatment already exists.

Courses reveal it.

---

### Rule 12 — Canonical Representation

One node.

One identity.

Many aliases.

Never duplicate knowledge.

---

### Rule 13 — Local Completeness

A node should answer one thing completely.

Not half.

---

### Rule 14 — Cross-Domain Validity

If knowledge survives being moved to another domain,

it probably belongs lower.

Example

```
Statistics

Medicine

Economics

Physics

AI

Business
```

Statistics shouldn't belong to Medicine.

Medicine references Statistics.

---

### Rule 15 — Scope Monotonicity

Parents become broader.

Children become more specific.

Never oscillate.

```
Programming

↓

Python

↓

Concurrency

↓

AsyncIO

↓

Coroutine
```

Not

```
Programming

↓

Coroutine

↓

Python

↓

Thread

↓

Language
```

---

## Then, and I think this is the biggest missing piece, you need **promotion and demotion rules**.

Your graph isn't static. It's learned from courses.

Suppose after 500 courses you discover

```
Digital Twin
```

appears in

```
Manufacturing

Robotics

Civil Engineering

Healthcare

Simulation

IoT

Smart Cities
```

Initially you may have stored it as

```
Topic
```

But after observing widespread reuse, the graph should **promote** it to a broader, reusable knowledge object.

Likewise, if something thought to be broadly reusable appears only within a niche context after large-scale ingestion, it may be **demoted**.

This is where the graph becomes self-organizing. You're not hard-coding "Digital Twin is a Topic." You're defining measurable promotion and demotion criteria based on reuse, composability, dependency patterns, and structural role across the corpus.

---

So I think the biggest change I'd make is this:

Don't try to define **"What is a Domain?"** or **"What is a Topic?"** in isolation.

Instead, define the **invariants** that every knowledge object must obey (identity, composability, reusability, minimality, canonical representation, etc.), and then define the **promotion/demotion rules** that move objects through the organizational layers as more evidence accumulates from course decomposition.

Those organizational layers (Domain, Subdomain, Field, Topic, etc.) become **emergent organizational views** over a stable set of reusable knowledge objects rather than rigid classifications assigned up front. That aligns much more closely with your goal of making courses the acquisition mechanism while allowing the knowledge genome to continuously compose and decompose itself as it grows.

I think we've finally reached the actual problem.

The mistake is trying to define **Domain, Topic, Concept, Skill, ...** independently.

They cannot be defined independently.

They are **defined by invariants relative to every other level.**

That means the ontology must satisfy global invariants.

For something as large as the Knowledge Genome, I'd probably start here.

---

# Global Invariants

These apply to **every** node regardless of type.

### Identity

A node represents exactly one canonical thing.

Never multiple.

```
Bad

Python + AsyncIO

Good

Python

AsyncIO
```

---

### Atomicity

A node should not contain two independent reusable things.

If it can be split without losing meaning

Split it.

---

### Canonicality

Every real-world thing has exactly one canonical node.

Aliases point to it.

Never duplicates.

---

### Composability

Every node must be reusable.

Nothing should belong permanently to one parent.

```
Heat Treatment

belongs to

Metallurgy

is used by

Swordsmithing

Knife Making

Manufacturing

Mechanical Engineering
```

---

### Decomposability

Every composite node must decompose into reusable nodes.

---

### Closure

A node completely represents its identity.

Not half.

Not 30%.

---

### Orthogonality

Different node types represent different dimensions.

Never overload.

---

### Stability

A node's type never changes because more books were written.

Only its relationships change.

---

### Promotion

Nodes may move upward in organizational scope.

Never because of opinion.

Only because evidence changes.

---

### Demotion

Likewise.

---

Those are global.

Now each ontology level gets its own invariants.

---

# Civilization

Invariant

Contains every teachable domain.

Never belongs to another node.

Exactly one root.

```
Civilization

↓

Everything
```

---

# Domain

A Domain is not "big."

A Domain is the highest stable partition of human knowledge.

Invariant

* Can evolve independently.
* Has independent research.
* Has independent practitioners.
* Has independent education.
* Has independent literature.
* Has independent history.

If all five are not true

It isn't a Domain.

Example

```
Medicine

Engineering

Language

Physics

Business
```

---

# Subdomain

Invariant

Partitions a Domain into major knowledge ecosystems.

Must inherit every Domain invariant except independence.

Rules

* Cannot exist without one Domain.
* Groups multiple Disciplines.
* Large enough to produce entire curricula.
* Stable for decades.

Example

```
Computer Science

Software Engineering

Artificial Intelligence
```

inside Technology.

---

# Discipline

This one is important.

A Discipline is where people actually become experts.

Invariant

Produces specialists.

Has methodologies.

Has journals.

Has conferences.

Has standards.

Has research.

Has educational pathways.

If it lacks those

Not a Discipline.

```
Structural Engineering

Microbiology

Linguistics

Organic Chemistry
```

---

# Specialization

Invariant

Narrows one Discipline.

Cannot redefine parent.

Adds constraints.

Produces experts.

```
Neurosurgery

Machine Learning

Bridge Engineering

Pediatric Nursing
```

---

# Field

A Field is interesting.

I actually don't think it belongs structurally.

I think Field is contextual.

Because

```
Artificial Intelligence
```

can be

a Discipline

a Field

an Industry

depending on viewpoint.

I'd eliminate it.

---

# Topic

Here's where most systems fail.

Topic isn't

"small."

Topic isn't

"chapter."

Invariant

A Topic organizes reusable knowledge.

It does not explain.

It references.

```
Heat Treatment

contains

Concepts

Methods

Procedures

Safety

Projects

Tools

Skills
```

Notice

Heat Treatment isn't the Concept.

It's an organizational hub.

---

# Concept

Invariant

Represents one abstract idea.

Cannot be executed.

Cannot produce an artifact.

Can only be understood.

```
Martensite

Stress

Entropy

Reference Frame

Coroutine

Grammar

```

---

# Principle

Invariant

Expresses a universally reusable rule.

Independent of implementation.

```
Heat expands materials.

Least privilege.

Conservation of Energy.
```

---

# Theory

Invariant

Explains why principles exist.

Predictive.

Evidence-based.

---

# Method

Invariant

Strategy.

Not execution.

```
Scientific Method

Agile Development

Finite Element Analysis
```

---

# Process

Invariant

Transforms an input into an output.

May contain many procedures.

```
Steel Production

Software Development

Hiring Process
```

---

# Procedure

Invariant

Ordered repeatable sequence.

Produces one outcome.

```
Quench Steel

Install Linux

Draw Blood
```

---

# Technique

Invariant

Local optimization of a Procedure.

```
Oil Quenching

Backhand Grip

Dependency Injection
```

---

# Skill

Invariant

Human capability.

Observable.

Measurable.

Improves with repetition.

Cannot exist without a performer.

```
Debugging

Welding

Public Speaking
```

---

# Tool

Invariant

Acts upon something.

Never performs knowledge.

```
Hammer

Compiler

Oscilloscope

IDE
```

---

# Material

Invariant

Consumed.

Modified.

Processed.

```
Steel

Wood

Electricity

Data

DNA Sample
```

---

# Component

Invariant

Reusable building block.

```
Bearing

Function

Neuron

Gear

API Endpoint
```

---

# System

Invariant

Interacting Components.

Emergent behavior.

```
Operating System

Human Body

Transmission

Compiler
```

---

# Project

Invariant

Temporary composition.

Produces an artifact.

Uses knowledge.

Never defines knowledge.

```
Build Chess Engine

Forge Katana

Bridge Design
```

---

# Exercise

Invariant

Practice.

Improves Skills.

Produces learning.

---

# Assessment

Invariant

Measures capability.

Produces evidence.

Not learning.

Measurement.

---

# Course

This is where I disagree with almost every ontology.

A Course should **not** be part of the Knowledge Genome.

A Course is not knowledge.

It is a **learning artifact**.

Its invariants are fundamentally different.

A Course:

* references knowledge
* sequences knowledge
* teaches knowledge
* evaluates knowledge

It owns none of it.

So a Course should be modeled as a **graph object** that points into the genome rather than as another level within the ontology.

```
Course

↓

Learning Graph

↓

Lesson Graph

↓

Knowledge References
```

not

```
Domain

↓

...

↓

Assessment

↓

Course
```

because that mixes **knowledge** with **pedagogy**.

---

## Finally, the invariant that ties everything together

I think there is one invariant above all the others.

> **Every node is classified by the role it plays in the organization, production, application, or transmission of knowledge—not by its size, popularity, educational level, or how often it appears in courses.**

Everything else follows from that.

A Domain organizes civilization's knowledge.

A Discipline produces new knowledge.

A Topic organizes related knowledge.

A Concept represents knowledge.

A Principle constrains knowledge.

A Theory explains knowledge.

A Method plans knowledge application.

A Process transforms knowledge into outcomes.

A Procedure operationalizes a process.

A Technique optimizes a procedure.

A Skill embodies knowledge in a person.

A Tool enables knowledge application.

A Material is acted upon by knowledge.

A Component is reused by systems.

A System integrates components into emergent behavior.

A Project applies knowledge to create artifacts.

An Exercise develops knowledge.

An Assessment measures knowledge.

A Course sequences and delivers knowledge.

The important distinction is that these invariants are **structural roles**. They do not depend on whether a node is "large" or "small", whether it appears in one course or a thousand, or whether it belongs to programming, medicine, or swordsmithing. That makes them stable enough for a knowledge genome that continuously composes and decomposes itself as new courses are ingested.


I actually think there is one major addition before we even get to Ozone-Studio.

What we've built is an ontology.

But Ozone-Studio doesn't classify ontologies.

It classifies **graph evidence**.

Those are two different things.

The ontology defines **what a Domain is**.

Ozone-Studio defines **how enough evidence accumulates to declare something a Domain.**

Those are separate layers.

I would separate the system into four layers.

```
Layer 1
Knowledge Invariants
(What each object fundamentally is.)

↓

Layer 2
Evidence Invariants
(What evidence must exist before a node may become that object.)

↓

Layer 3
Promotion/Demotion
(When evidence changes classification.)

↓

Layer 4
Confidence
(How certain are we?)
```

I think Layer 2 is actually what you've been asking for all along.

---

# Layer 2 — Evidence Invariants

Forget humans.

Forget asking

> Is this a Domain?

Instead the graph computes evidence.

For every candidate node.

For example

```
Candidate

Heat Treatment
```

The graph already knows

```
Appears In

Swordsmithing
Knife Making
Mechanical Engineering
Manufacturing
Materials Engineering
Welding

Referenced By

218 Topics

Requires

Steel
Carbon
Temperature

Produces

Hardness
Microstructure

Used By

192 Procedures

Referenced In

81 Courses
```

Nothing here is semantic reasoning.

Everything comes from graph traversal.

Now we compute invariants.

---

# Identity Evidence

Instead of asking

```
Is this one thing?
```

Compute

```
Incoming aliases

Canonical references

Description overlap

Neighbor overlap

Duplicate probability
```

Produces

```
Identity Score

0.98
```

---

# Reuse Evidence

Instead of

```
Can this exist elsewhere?
```

Traverse

```
Incoming references

Outgoing references

Unique parents

Unique domains

Unique disciplines
```

Produces

```
Reuse Score

0.93
```

---

# Atomicity Evidence

Instead of

```
Should it split?
```

Traverse

```
Internal concepts

Internal procedures

Internal entities

Dependency density

Relationship cohesion
```

Produces

```
Atomicity

0.41

↓

Probably Composite
```

---

# Compression Evidence

Instead of

```
Does extraction reduce duplication?
```

Compute

```
Occurrences

Repeated neighborhoods

Shared relationships

Graph entropy reduction
```

Produces

```
Compression Gain

27%
```

---

# Canonical Evidence

Traverse

```
Aliases

Alternative spellings

Equivalent translations

Duplicate embeddings

Shared graph neighborhoods
```

Produces

```
Canonical Confidence

99%
```

---

Now imagine every invariant becomes evidence.

---

Identity

↓

Identity Score

---

Atomicity

↓

Atomic Score

---

Reusability

↓

Reuse Score

---

Composability

↓

Composition Score

---

Cross-domain

↓

Domain Spread

---

Dependency

↓

Dependency Depth

---

Closure

↓

Internal Cohesion

---

Promotion

↓

Promotion Potential

---

Everything becomes measurable.

---

# Now we return to your original ontology.

Instead of

```
Domain
```

having human definitions

it has graph invariants.

For example.

---

## Domain

Graph invariant.

A Domain should maximize separation while maximizing internal cohesion.

The graph measures

```
Number of Disciplines

Internal Connectivity

External Connectivity

Curriculum Count

Independent Citation Network

Independent Methodologies

Independent Terminology

Independent Skill Graph
```

Now Domain isn't

```
Large.
```

It becomes

```
Graph Partition
```

---

## Discipline

Graph invariant.

Produces methodologies.

Produces projects.

Produces skills.

Produces assessments.

Produces concepts.

Produces procedures.

The graph literally measures

```
Method Nodes

Procedure Nodes

Project Nodes

Theory Nodes

Skill Nodes

Assessment Nodes
```

If all emerge naturally

Discipline score increases.

---

## Topic

Instead of

```
Subject discussed.
```

Graph definition.

```
High internal cohesion.

Low external cohesion.

References multiple concepts.

References multiple procedures.

Has bounded scope.

Frequently reused.
```

Everything measurable.

---

## Concept

Graph definition.

```
No execution edges.

No procedural edges.

High explanatory edges.

High reference count.

Low decomposition.
```

That is an invariant.

---

## Procedure

Graph invariant.

```
Has ordering.

Has prerequisites.

Produces outputs.

Contains executable steps.

Ends in measurable state.
```

The graph already knows.

---

## Skill

Graph invariant.

```
Referenced by Exercises.

Referenced by Assessments.

Referenced by Projects.

Depends on Procedures.

Depends on Concepts.
```

Notice.

No semantic reasoning.

Only graph evidence.

---

# This is where Ozone-Studio enters.

Ozone-Studio never asks

```
Is Welding a Skill?
```

Instead

it traverses.

```
Node

↓

Neighbors

↓

Neighbor Types

↓

Relationship Density

↓

Cross References

↓

Dependency Graph

↓

Reuse Graph

↓

Hierarchy Graph

↓

Evidence Scores
```

Now the LLM receives

```
Context Pool

Heat Treatment

Neighbors

Martensite
Ferrite
Steel
Carbon
Quenching
Tempering
Hardness
Manufacturing
Knife Making
Swordsmithing
Mechanical Engineering
Materials Science

Relationship Summary

Uses

Referenced By

Explains

Requires

Produces

Part Of

Depends On

Graph Statistics

Reuse

0.94

Atomicity

0.42

Cross Domain

0.88

Execution

0.63

Theory

0.14

Concept

0.47
```

The LLM isn't deciding from English anymore.

It's deciding from graph evidence.

---

# I actually think there is one final layer missing that completes everything.

You currently have:

```
Knowledge Genome

↓

Ontology

↓

Graph

↓

LLM
```

I think there needs to be one layer between the Graph and the LLM.

I would call it the **Evidence Graph**.

```
Knowledge Graph

↓

Evidence Graph

↓

Context Pool

↓

LLM Traversal

↓

Classification
```

The Knowledge Graph stores facts:

```
Heat Treatment

uses

Steel

Heat Treatment

requires

Quenching

Heat Treatment

referenced_by

Swordsmithing
```

The Evidence Graph stores derived structural metrics:

```
Reuse Score

Cross-Domain Score

Internal Cohesion

Boundary Strength

Graph Diameter

Dependency Depth

Promotion Score

Canonical Confidence

Atomicity

Compression Gain

Structural Stability

Role Distribution

Parent Diversity

Child Diversity
```

That separation is critical because the LLM should not spend tokens rediscovering graph properties that the graph database can compute deterministically.

## So the addendum I'd make to the entire design is this:

You don't want to translate your ontology invariants directly into prompts or "questions." Instead, every invariant should become a **graph-computable metric**. The graph traversal computes those metrics first, producing an evidence graph. The LLM then traverses that evidence graph, not the raw text, using a rich context pool composed of neighborhood structure, relationship types, evidence scores, and supporting observations from multiple courses.

That changes the role of the LLM fundamentally. It is no longer trying to infer "What is Heat Treatment?" from language. It is interpreting a structured body of graph evidence to determine where the node belongs in the knowledge genome, and as more courses are ingested, those evidence metrics evolve, allowing promotion, demotion, merging, or splitting decisions to emerge from accumulated structural evidence rather than from ad hoc semantic judgments. I think that is the closest analogue to how Ozone-Studio's graph-first philosophy can be transferred to UngatedMinds without turning ontology construction into a collection of brittle natural-language heuristics.

Note that I haven't provided you topics yet or have we created this knowledge graph but we will, I ahve courses and topics that we will break down so this is just to keep in mind around it keep in mind that at this time you do not need to be aaware of the coruses but they are There

For the grammar graph, every edge type has a deterministic traversal invariant. For example:

Grammar relationship	Traversal invariant	Effect on context pool
Subject → Predicate	Establish actor	Include the executor of the action
Predicate → Verb	Establish execution root	Anchor the traversal
Verb → Direct Object	Expand primary target	Include what the action operates on
Verb → Indirect Object	Expand recipient	Include who receives the result
Verb → Prepositional Phrase	Expand contextual constraints	Include instruments, locations, or other relations depending on the preposition
Verb → Adverbial	Expand execution modifiers	Include manner, degree, frequency, etc.
Verb → Purpose Clause	Expand downstream objective	Include intended outcome
Verb → Conditional Clause	Expand execution conditions	Include prerequisite branches
Verb → Temporal Clause	Expand execution ordering	Include sequencing information
Relative Clause → Modified Node	Expand defining properties	Refine the identity of the referenced node

So the traversal algorithm is not "follow action verbs." The verb is simply the execution anchor. The real traversal follows the typed dependency edges outward according to deterministic expansion rules, building a context pool that represents the complete executable meaning of the sentence.

If I completely understand a sentence structurally (via the grammar graph), how do I generate a natural response from that graph?

That is a completely different problem than understanding.

And I think the answer is that writing is simply grammar traversal in reverse.

Everyone assumes language works like this
Text

↓

Grammar

↓

Meaning

↓

Response

But that skips the most important layer.

It should be

Input Text

↓

Grammar Graph

↓


Intent Graph

↓

Decision

↓

Response Graph

↓

Grammar Graph

↓

Output Text

Notice something.

Grammar exists on both sides.

It is both the parser AND the renderer.

now as you can see path 2 we are preparing it to now work based on OMEX new findings, you will not provide any modification changes you will notprovide me full files of the code back to mecorrected what you will do is lookat each of my quetions andc larifications from A-G and dwell fully on all of them and answer them answer them all completely. because we we continue to allow Ozone-Studio to work with OMEX and without OMEX IE Bitnet or etc, so we will structure each path accordingly. Now Path1 may be for LLM's while we still may look at PAth 2 being for SLM's with smaller context or for OMEX as well or path 1 can be for LLM's and OMEX as well I think both paths work for OMEX hile Path 1 seems or will probably be the best for for OMEX just at how optimized it is around each compontent to capture it all from a chunk itself without needing to grnually go through everything. so both paths we are keeping it for SLM'sLLMs and as well for OMEX. So what you will do is 1. provide me the fully updated OMEX readme corrected and finalized and all supporting docs around it technical documnets etc etc and then proceed to answering all my questions right away from A-G and everything in it capturing all insights. You will truly capture just how much this optimizes from currentSOTA  model and just how truly revolutionary this is capture all inputs and areas on what will be truly possible with OMEX and just how much more it ill allow for speed to speed everything up energy saving per token if we are still utilizing a token format or w.e or compare and speed as well capture all possible metric comparisons.

provide me everything complete whole without omission not matter how long it takes how long it is it doesn't matter if it takes multiple prompts to complete provide me everything outside of an artifact in whole full completewithout omission you will not use skills provide all in one response at once in one full sweep if limit is reached we can always continue do not rush do not ommit do not drop anything capure all respond everything and capture all insights again you will respond outside of an artifact you will respond with everything all together and provide me the update readme whole complete capturign all updates without omission without dropping anything and provide me all documents now around the readme to suppotr all the insights we have captured all outside of an artifact all at once no matter how longwithout utilziing skills.
