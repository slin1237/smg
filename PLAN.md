# Batch API Support for gRPC Router — Implementation Plan

## Architecture Overview

SMG (Shepherd Model Gateway) is a production-grade inference gateway with:
- **gRPC Router** (`model_gateway/src/routers/grpc/`) — handles chat completions, generate, embeddings, classify, messages via request pipelines with retry logic
- **HTTP Router** (`model_gateway/src/routers/http/`) — policy-based routing with PD support
- **OpenAI Router** (`model_gateway/src/routers/openai/`) — OpenAI-compatible REST endpoints
- **Workflow Engine** (`crates/workflow/`) — DAG-based execution engine with retry, state persistence, events
- **Job Queue** (`model_gateway/src/core/job_queue.rs`) — async control plane job processing with semaphore concurrency
- **Token Bucket** (`model_gateway/src/core/token_bucket.rs`) — smooth rate limiting with burst capacity
- **Worker Registry** — manages backend workers (SGLang, vLLM, TensorRT-LLM) with load monitoring

## Design Goals

1. **OpenAI Batch API compatibility** — match the OpenAI Batch API semantics (create, retrieve, list, cancel batches)
2. **gRPC-native** — expose batch operations as gRPC RPCs, not just REST wrappers
3. **Non-disruptive scheduling** — batch requests must not starve real-time inference
4. **Leverage existing workflow engine** — use the DAG workflow engine for batch lifecycle management
5. **Pluggable storage** — batch input/output files and state need persistent storage

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   gRPC Service Layer                 │
│  CreateBatch | GetBatch | ListBatches | CancelBatch  │
│  UploadBatchFile | DownloadBatchFile                 │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│              Batch Manager (new)                     │
│  - Validates input files                             │
│  - Creates workflow instances                        │
│  - Tracks batch state (validating→in_progress→done)  │
│  - Manages file storage                              │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│          Workflow Engine (existing crate)             │
│  DAG: Validate → Schedule → Process → Finalize       │
│  - Retry policies, timeouts, event bus               │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│          Batch Scheduler (new)                       │
│  - Separate token bucket for batch requests          │
│  - Priority: real-time > batch                       │
│  - Backpressure-aware: monitors worker load          │
│  - Configurable concurrency limit for batch          │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│        Existing Request Pipeline                     │
│  Preparation → WorkerSelection → ClientAcquisition   │
│  → RequestBuilding → Dispatch → ResponseProcessing   │
└─────────────────────────────────────────────────────┘
```

## Detailed Implementation Plan

### Phase 1: Protocol & Storage Layer

#### 1.1 Define Batch Proto Messages

**File:** `crates/grpc_client/proto/batch.proto` (new)

```protobuf
service BatchService {
  rpc CreateBatch(CreateBatchRequest) returns (BatchObject);
  rpc GetBatch(GetBatchRequest) returns (BatchObject);
  rpc ListBatches(ListBatchesRequest) returns (ListBatchesResponse);
  rpc CancelBatch(CancelBatchRequest) returns (BatchObject);
  rpc UploadBatchFile(stream UploadBatchFileRequest) returns (FileObject);
  rpc GetBatchFileContent(GetBatchFileContentRequest) returns (stream FileChunk);
}

message CreateBatchRequest {
  string input_file_id = 1;
  string endpoint = 2;           // "/v1/chat/completions", "/v1/embeddings", etc.
  string completion_window = 3;  // "24h"
  map<string, string> metadata = 4;
}

message BatchObject {
  string id = 1;
  string object = 2;             // "batch"
  string endpoint = 3;
  BatchErrors errors = 4;
  string input_file_id = 5;
  string completion_window = 6;
  string status = 7;             // validating|failed|in_progress|finalizing|completed|expired|cancelling|cancelled
  string output_file_id = 8;
  string error_file_id = 9;
  int64 created_at = 10;
  int64 in_progress_at = 11;
  int64 expires_at = 12;
  int64 completed_at = 13;
  int64 failed_at = 14;
  int64 expired_at = 15;
  int64 cancelling_at = 16;
  int64 cancelled_at = 17;
  int64 finalizing_at = 18;
  RequestCounts request_counts = 19;
  map<string, string> metadata = 20;
}

message RequestCounts {
  int32 total = 1;
  int32 completed = 2;
  int32 failed = 3;
}

message BatchErrors {
  string object = 1;             // "list"
  repeated BatchError data = 2;
}

message BatchError {
  string code = 1;
  string message = 2;
  string param = 3;
  int32 line = 4;
}
```

#### 1.2 Define Batch Protocol Types

**File:** `crates/protocols/src/batch.rs` (new)

Define Rust types mirroring OpenAI's Batch API:
- `BatchRequest` — individual request line in JSONL (custom_id, method, url, body)
- `BatchResponse` — individual response line (id, custom_id, response, error)
- `BatchObject` — batch metadata with status tracking
- `BatchStatus` enum — validating, failed, in_progress, finalizing, completed, expired, cancelling, cancelled
- `FileObject` — uploaded file metadata

#### 1.3 Batch File Storage

**File:** `model_gateway/src/core/batch_store.rs` (new)

```rust
#[async_trait]
pub trait BatchStore: Send + Sync {
    // File operations
    async fn store_file(&self, file_id: &str, content: Vec<u8>, purpose: &str) -> Result<FileObject>;
    async fn get_file(&self, file_id: &str) -> Result<Vec<u8>>;
    async fn delete_file(&self, file_id: &str) -> Result<()>;

    // Batch state operations
    async fn save_batch(&self, batch: &BatchObject) -> Result<()>;
    async fn get_batch(&self, batch_id: &str) -> Result<BatchObject>;
    async fn list_batches(&self, limit: usize, after: Option<&str>) -> Result<Vec<BatchObject>>;
    async fn update_batch_status(&self, batch_id: &str, status: BatchStatus) -> Result<()>;
    async fn update_request_counts(&self, batch_id: &str, completed: i32, failed: i32) -> Result<()>;

    // Output operations
    async fn append_output(&self, batch_id: &str, response: &BatchResponseLine) -> Result<()>;
    async fn append_error(&self, batch_id: &str, error: &BatchResponseLine) -> Result<()>;
    async fn finalize_output(&self, batch_id: &str) -> Result<String>; // returns output_file_id
    async fn finalize_errors(&self, batch_id: &str) -> Result<String>; // returns error_file_id
}
```

**Initial implementation:** `InMemoryBatchStore` using `DashMap`. Future: pluggable backends (filesystem, S3, database).

### Phase 2: Batch Workflow Definition

#### 2.1 Batch Workflow Data

**File:** `model_gateway/src/batch/workflow.rs` (new)

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct BatchWorkflowData {
    pub batch_id: String,
    pub input_file_id: String,
    pub endpoint: String,
    pub completion_window: Duration,
    pub requests: Vec<BatchRequestLine>,       // populated after validation
    pub completed_count: AtomicU32,
    pub failed_count: AtomicU32,
    pub expires_at: DateTime<Utc>,
}
```

#### 2.2 Workflow Steps (DAG)

Define a 4-step workflow using the existing workflow engine:

```
Step 1: ValidateInput
  - Parse JSONL input file
  - Validate each request (schema, endpoint match, model consistency)
  - Populate requests in workflow context
  - On failure: mark batch as "failed"

Step 2: ProcessRequests (depends_on: ValidateInput)
  - Fan out individual requests through the BatchScheduler
  - Each request goes through existing RequestPipeline
  - Results accumulated in output/error buffers
  - Supports cancellation mid-processing

Step 3: FinalizeOutput (depends_on: ProcessRequests)
  - Aggregate all results into output JSONL file
  - Aggregate errors into error JSONL file
  - Store files via BatchStore
  - Update batch object with file IDs

Step 4: Cleanup (depends_on: FinalizeOutput, on_failure: ContinueNextStep)
  - Set batch status to "completed" or "expired"
  - Emit completion metrics
  - Schedule file cleanup after 30 days
```

### Phase 3: Batch Scheduler

#### 3.1 Scheduler Design

**File:** `model_gateway/src/core/batch_scheduler.rs` (new)

The batch scheduler ensures batch requests don't impact real-time inference:

```rust
pub struct BatchScheduler {
    /// Separate token bucket for batch — lower rate than real-time
    batch_token_bucket: TokenBucket,

    /// Maximum concurrent batch requests in-flight
    max_concurrent_batch: Arc<Semaphore>,

    /// Worker load monitor — pause batch when workers are loaded
    load_monitor: Arc<LoadMonitor>,

    /// Load threshold: pause batch processing when worker load exceeds this
    load_threshold: f64,

    /// Batch request queue (bounded)
    request_queue: mpsc::Sender<BatchRequestItem>,

    /// Shutdown signal
    shutdown: CancellationToken,
}
```

**Scheduling Strategy:**

1. **Separate token bucket** — batch requests get their own rate limit, completely isolated from real-time traffic. Configurable via `BatchSchedulerConfig`.

2. **Load-aware throttling** — before dispatching each batch request:
   - Query `LoadMonitor` for current worker utilization
   - If load > threshold (e.g., 80%), pause batch processing
   - Resume when load drops below threshold
   - Check on a configurable interval (e.g., every 5 seconds)

3. **Concurrency cap** — `max_concurrent_batch` semaphore limits in-flight batch requests (default: 10-20% of total worker capacity).

4. **Priority queue** — batch requests enqueued at lower priority than real-time. The existing `ConcurrencyLimiter` middleware handles real-time; batch goes through its own path.

5. **Backpressure** — if batch queue is full, new batch creation returns appropriate error.

#### 3.2 Configuration

```rust
pub struct BatchSchedulerConfig {
    /// Tokens per second for batch processing (default: 10.0)
    pub batch_rate_limit: f64,

    /// Burst capacity for batch token bucket (default: 20)
    pub batch_burst_capacity: f64,

    /// Max concurrent batch requests (default: 10)
    pub max_concurrent_batch: usize,

    /// Worker load threshold to pause batch (default: 0.8)
    pub load_pause_threshold: f64,

    /// Worker load threshold to resume batch (default: 0.6)
    pub load_resume_threshold: f64,

    /// Load check interval (default: 5s)
    pub load_check_interval: Duration,

    /// Batch expiration window (default: 24h)
    pub default_completion_window: Duration,

    /// Max requests per batch (default: 50000)
    pub max_requests_per_batch: usize,

    /// Max input file size (default: 200MB)
    pub max_input_file_size: usize,
}
```

### Phase 4: Batch Manager & gRPC Service

#### 4.1 Batch Manager

**File:** `model_gateway/src/batch/manager.rs` (new)

Central coordinator that ties together storage, workflow, and scheduler:

```rust
pub struct BatchManager {
    store: Arc<dyn BatchStore>,
    workflow_engine: Arc<WorkflowEngine<BatchWorkflowData>>,
    scheduler: Arc<BatchScheduler>,
    batch_workflow_def: Arc<WorkflowDefinition<BatchWorkflowData>>,
}

impl BatchManager {
    pub async fn create_batch(&self, req: CreateBatchRequest) -> Result<BatchObject>;
    pub async fn get_batch(&self, batch_id: &str) -> Result<BatchObject>;
    pub async fn list_batches(&self, limit: usize, after: Option<&str>) -> Result<Vec<BatchObject>>;
    pub async fn cancel_batch(&self, batch_id: &str) -> Result<BatchObject>;
    pub async fn upload_file(&self, content: Vec<u8>, purpose: &str) -> Result<FileObject>;
    pub async fn get_file_content(&self, file_id: &str) -> Result<Vec<u8>>;
}
```

#### 4.2 gRPC Service Implementation

**File:** `model_gateway/src/routers/grpc/batch_service.rs` (new)

Implement the `BatchService` gRPC trait, delegating to `BatchManager`.

#### 4.3 Integration with GrpcRouter

**File:** `model_gateway/src/routers/grpc/router.rs` (modify)

- Add `BatchManager` to `GrpcRouter`
- Register `BatchService` with the tonic server
- Wire batch scheduler to use the existing request pipelines

### Phase 5: Request Processing Integration

#### 5.1 Batch Request Executor

**File:** `model_gateway/src/batch/executor.rs` (new)

The `ProcessRequests` workflow step executor:

```rust
pub struct BatchRequestExecutor {
    scheduler: Arc<BatchScheduler>,
    pipeline: RequestPipeline,
    store: Arc<dyn BatchStore>,
}

impl StepExecutor<BatchWorkflowData> for BatchRequestExecutor {
    async fn execute(&self, context: &mut WorkflowContext<BatchWorkflowData>) -> WorkflowResult<StepResult> {
        let requests = &context.data.requests;
        let batch_id = &context.data.batch_id;

        // Process requests through scheduler (respects rate limits + load)
        let mut futures = FuturesUnordered::new();

        for request in requests {
            // Acquire batch scheduler permit (blocks if overloaded)
            let permit = self.scheduler.acquire().await?;

            // Check cancellation
            if context.is_cancelled() {
                return Ok(StepResult::Failure);
            }

            // Dispatch through existing pipeline
            let fut = self.process_single_request(batch_id, request, permit);
            futures.push(fut);
        }

        // Collect results, update counts
        while let Some(result) = futures.next().await {
            match result {
                Ok(response) => {
                    self.store.append_output(batch_id, &response).await?;
                    context.data.completed_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(error) => {
                    self.store.append_error(batch_id, &error).await?;
                    context.data.failed_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            // Periodically update batch request_counts in store
        }

        Ok(StepResult::Success)
    }
}
```

#### 5.2 Request Routing

Each batch request line specifies an endpoint (e.g., `/v1/chat/completions`). The executor routes to the appropriate existing pipeline:
- `/v1/chat/completions` → `route_chat` pipeline
- `/v1/embeddings` → `route_embeddings` pipeline
- `/v1/completions` → `route_generate` pipeline

This reuses all existing logic: worker selection, load balancing, retry, response processing.

### Phase 6: Observability & Metrics

#### 6.1 Metrics

Add batch-specific Prometheus metrics:
- `smg_batch_total` — counter of batches created (by status)
- `smg_batch_requests_total` — counter of individual batch requests (by status)
- `smg_batch_duration_seconds` — histogram of batch completion time
- `smg_batch_active` — gauge of currently active batches
- `smg_batch_scheduler_queue_depth` — gauge of pending batch requests
- `smg_batch_scheduler_load_paused` — gauge (0/1) of scheduler pause state

#### 6.2 Workflow Events

Subscribe to workflow events via the existing `EventBus` for batch-specific logging and metrics.

### Phase 7: Expiration & Cleanup

- Background task monitors batch expiration (24h window)
- Expired batches: cancel remaining requests, finalize partial results
- Output files auto-deleted after 30 days (configurable)
- Use existing workflow cleanup mechanism for completed batch workflow states

## File Summary

### New Files
| File | Purpose |
|------|---------|
| `crates/grpc_client/proto/batch.proto` | Batch gRPC service definition |
| `crates/protocols/src/batch.rs` | Batch protocol types |
| `model_gateway/src/core/batch_store.rs` | Batch file & state storage |
| `model_gateway/src/core/batch_scheduler.rs` | Load-aware batch scheduler |
| `model_gateway/src/batch/mod.rs` | Batch module root |
| `model_gateway/src/batch/manager.rs` | Batch lifecycle manager |
| `model_gateway/src/batch/workflow.rs` | Batch workflow definition |
| `model_gateway/src/batch/executor.rs` | Batch request processing steps |
| `model_gateway/src/routers/grpc/batch_service.rs` | gRPC BatchService impl |

### Modified Files
| File | Change |
|------|--------|
| `model_gateway/src/routers/grpc/router.rs` | Add BatchManager, register BatchService |
| `model_gateway/src/config/mod.rs` | Add BatchSchedulerConfig |
| `model_gateway/src/core/mod.rs` | Export batch_store, batch_scheduler |
| `model_gateway/src/observability/metrics.rs` | Add batch metrics |
| `crates/protocols/src/lib.rs` | Export batch module |
| `model_gateway/src/main.rs` | Initialize batch subsystem |

## Key Design Decisions & Rationale

### Why use the existing Workflow Engine?
- Already provides DAG execution, retry, state persistence, events, cancellation
- Batch lifecycle maps perfectly to a 4-step workflow (validate → process → finalize → cleanup)
- No need to build custom state machine from scratch
- Gets retry with backoff for free
- Gets event-based observability for free

### Why a separate token bucket for batch?
- OpenAI's Batch API explicitly states "separate pool of significantly higher rate limits"
- Prevents batch from consuming real-time rate limit tokens
- Load-aware throttling adds a second layer of protection
- Simple to configure independently

### Why gRPC-native (not REST wrapper)?
- SMG's gRPC router already handles the core inference RPCs
- gRPC streaming enables efficient large file upload/download
- Consistent with existing architecture
- Can easily add HTTP/REST facade later via gRPC-gateway or manual mapping

### Why not extend the existing Job Queue?
- Job Queue is designed for control plane operations (add/remove workers)
- Batch processing is data plane with very different characteristics
- Batch needs load-aware scheduling, Job Queue uses simple semaphore
- Batch needs file I/O and result accumulation, Job Queue is fire-and-forget
- Keeping them separate avoids coupling and allows independent scaling
