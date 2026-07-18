# RFC 0001: Cross-Engine RL Rollout Control Plane

- **Status**: Draft
- **Created**: 2026-07-18
- **Scope**: model_gateway, crates/grpc_client, crates/protocols, crates/workflow, crates/data_connector

## Summary

Make SMG the neutral, engine-agnostic control plane for reinforcement-learning
rollout generation across mixed vLLM / SGLang / TensorRT-LLM / TokenSpeed fleets.

Concretely, SMG gains:

1. **Fleet control verbs** — gateway-native, fanned-out, correctly sequenced
   `abort / pause / continue / sleep / wake / flush / update-weights`
   orchestration over heterogeneous engines, exposed as admin APIs and driven
   by the existing DAG workflow engine.
2. **A weight-version registry** — first-class tracking of which policy
   version each worker (and each response) was served from, with
   version-consistent routing, drain-and-flip weight updates, and staleness
   metadata that trainers can use for importance-sampling correction.
3. **Rollout data-plane guarantees** — token-in/token-out fidelity, sampled
   logprobs, partial-output-on-abort semantics, and MoE expert-routing
   passthrough, verified per engine.
4. **Framework adapters** — drop-in integration with slime, verl, TRL, SkyRL,
   and AReaL through their existing extension seams, plus reward-model worker
   roles.

## Motivation

### The category exists; the product does not

SGLang documents its Model Gateway as "the recommended control plane for
large-scale RL rollouts," and it is used in production for GLM training via
slime. But a source-level audit (July 2026) shows the orchestration story is
thinner than the positioning:

- The gateway has **no fan-out** for `update_weights_*`, `pause_generation`,
  `continue_generation`, `release/resume_memory_occupation`, or
  `abort_request`. The issue asking for gateway-level abort
  ([sglang#6531](https://github.com/sgl-project/sglang/issues/6531)) has been
  open since May 2025.
- slime implements drain client-side: it lists workers through the router,
  then **bypasses the router** to call `/abort_request {"abort_all": true}`
  on each engine in a 3-second polling loop — with a vendored "double abort"
  patch because a single abort races in-flight requests.
- Naive load balancing of control verbs split-brains fleets: a
  `pause_generation` that landed on one worker and `continue_generation` on
  another left 1/8 of a fleet paused
  ([sglang#21235](https://github.com/sgl-project/sglang/issues/21235)).
- `weight_version` is a passive label. Nothing prevents mixed-version rollouts
  mid-update, and nothing routes or quarantines by version.
- Everything is coupled to SGLang server APIs. vLLM worker support was
  explicitly deflected ([sglang#11703](https://github.com/sgl-project/sglang/issues/11703)).

Meanwhile the demand side is explicit:

- verl's community has open RFCs for exactly this: a **Trajectory Gateway**
  ([verl#5790](https://github.com/volcengine/verl/issues/5790)) and
  RemoteAgentLoop ([verl#5737](https://github.com/volcengine/verl/issues/5737));
  verl's in-house `GlobalRequestLoadBalancer` is a single Ray actor doing
  least-in-flight + sticky LRU — no cache awareness, no PD, no fault tolerance.
- vLLM shipped **native RL APIs** in 2026 (`/pause`, `/resume`, the
  `WeightTransferEngine` endpoint family) and framed them as "a standard
  interface for RL frameworks" — engine-side primitives waiting for a control
  plane.
- SGLang's own 2026 roadmaps commit to "Gateway as the DP scheduler for
  rollout" and a "shared rollout interface" with slime/verl/AReaL — the race
  is on, and the incumbent is single-engine.
- HuggingFace's survey of 16 RL libraries concludes **no common API standard
  exists**; every framework re-implements weight sync, interrupts, and
  staleness bookkeeping against raw engine APIs.

SMG is uniquely positioned: it is already engine-agnostic (HTTP + gRPC clients
for vLLM, SGLang, TRT-LLM, TokenSpeed), already has a DAG workflow engine, a
worker registry with drain semantics, fleet fan-out helpers, cache-aware
routing, and — critically — token-in/token-out with logprobs and
`weight_version` already plumbed end-to-end through `/generate`.

### Why a gateway at all (the "train/serve WYSIWYG" argument)

Routing rollouts and production traffic through the same gateway eliminates
train/serve scoring discrepancies (same tokenization, parsers, sampling
defaults). Beyond that, rollout traffic *needs* the things SMG already does
well — cache-aware routing (multi-turn agentic rollouts re-send growing
prefixes), PD disaggregation, circuit breakers, load skew management for
long-tail sequence lengths — plus the RL-specific control plane this RFC adds.

## Goals

- G1. Gateway-native fleet control verbs with correct sequencing, over HTTP
  and gRPC engines, engine dialect differences abstracted away.
- G2. Weight-update orchestration as first-class workflows: colocated
  (sleep/wake) and disaggregated (pause/transfer/resume) choreographies,
  including drain-and-flip (rolling, blue/green) update strategies.
- G3. Weight-version registry: per-worker version state, per-response version
  stamping (all engines, not just SGLang), version-consistent routing modes,
  quarantine of stale workers.
- G4. Rollout data-plane guarantees, contract-tested per engine: `input_ids`
  in; `output_ids`, sampled logprobs, `finish_reason=abort` partials,
  `weight_version`, and (where supported) routed experts out.
- G5. Framework adoption seams: slime preset-router drop-in; sgl-router-
  compatible worker CRUD; Python shims for verl `LLMServerClient`, TRL
  `base_url`, SkyRL `remote_inference_engine_urls`, AReaL `InferenceEngine`.
- G6. Reward-model worker role: policy vs reward pools behind one gateway,
  routed and health-checked independently.

## Non-goals

- Moving weight tensors through the gateway. Weights travel over NCCL /
  CUDA-IPC / RDMA between trainer and engines (or via checkpoint-engine). SMG
  sequences and triggers; it never carries tensor payloads.
- Implementing a trainer, reward function, or RL algorithm.
- Replacing checkpoint-engine / Mooncake / NIXL — SMG interoperates with them.
- Trajectory storage and token-faithful trajectory reconstruction from agent
  harness traffic (companion feature; Phase 4 here only reserves the schema
  seam in `data_connector`).

## Background: engine primitives (verified July 2026)

The verbs SMG must orchestrate, per engine. This table is the basis for the
adapter trait in the detailed design.

| Verb | SGLang | vLLM | TRT-LLM |
|---|---|---|---|
| Sleep | `POST /release_memory_occupation {tags:[weights\|kv_cache]}` (needs `--enable-memory-saver`) | `POST /sleep?level=1\|2` (dev mode + `--enable-sleep-mode`) | limited; recent single-rank sleep, treat as coarse |
| Wake (staged) | `POST /resume_memory_occupation {tags}` | `POST /wake_up` with `tags=["weights"]` then `["kv_cache"]` | — |
| Update weights (disk) | `POST /update_weights_from_disk {model_path, weight_version, flush_cache}` | none native (worker-extension only) | refit via orchestrator |
| Update weights (NCCL) | `POST /init_weights_update_group` → `POST /update_weights_from_distributed {names,dtypes,shapes,group_name,weight_version}` → `POST /destroy_weights_update_group` | native: `POST /init_weight_transfer_engine` → `/start_weight_update` → `/update_weights` (chunked) → `/finish_weight_update`; legacy: `collective_rpc("init_weight_update_group"/"update_weight")` | CUDA-IPC via NeMo-RL-style path |
| Update weights (IPC/tensor) | `POST /update_weights_from_tensor {serialized_named_tensors, weight_version}` | `collective_rpc("update_weights_from_ipc_handles")`; native `ipc` backend | `update_weights_from_ipc_handles` |
| Pause / continue | `POST /pause_generation {mode: abort\|retract\|in_place}` / `POST /continue_generation` | `POST /pause?mode=abort\|wait\|keep&clear_cache=` / `POST /resume` | drain via orchestrator |
| Abort requests | `POST /abort_request {rid \| abort_all}` — running requests return partial tokens with `finish_reason=abort` | no public per-request endpoint; disconnect aborts; `/pause?mode=abort` for fleet | engine `Abort` RPC |
| Flush KV | `POST /flush_cache` (auto after weight updates by default) | `POST /reset_prefix_cache` (dev mode; NOT automatic after legacy weight updates) | — |
| Version stamp in responses | yes: `meta_info.weight_version` (+ `GET /get_weight_version`) | **no** — gateway must supply | no — gateway must supply |
| Token-in/token-out + logprobs | `/generate` `input_ids`, `return_logprob`, `output_token_logprobs` | `prompt_token_ids` / `return_token_ids`, `logprobs`, `prompt_logprobs` | via gRPC Generate |
| Routed experts (MoE R3) | fork-only today (`return_routed_experts`) | `TokenOutput.routed_experts` in verl path | — |

Key sequencing constraints learned from engine docs and postmortems:

- **vLLM legacy weight path does not invalidate prefix cache.** After
  `collective_rpc` weight updates the gateway must call
  `/reset_prefix_cache`, or new weights serve stale-weight KV. The native
  path (`/finish_weight_update`, `/pause?clear_cache=true`) handles this.
- **Staged wake avoids OOM**: wake weights → transfer → free buffers → wake
  KV. Both engines support tags for exactly this; the gateway must not wake
  both at once on memory-tight colocated setups.
- **Control verbs must never be load-balanced.** They are fleet-scoped or
  worker-scoped, never request-routed (the sglang#21235 split-brain).
- **Partial-rollout resume is client re-issue.** No engine has server-side
  "continue rid". Aborted requests return accumulated tokens; resumption is a
  new `/generate` with `input_ids = prompt + partial`, which radix/prefix
  cache turns into a cheap re-prefill. The gateway's job: return partials
  faithfully, and route the resume where the prefix lives (cache-aware
  routing already does this).

## Background: what frameworks need (verified July 2026)

| Framework | Data plane | Weight sync | Interrupt model | Version/staleness | SMG seam |
|---|---|---|---|---|---|
| **slime** | router `/generate` `{input_ids, return_logprob}`; sticky `X-SMG-Routing-Key` | direct-to-server SGLang verbs, router bypassed | fan-out abort + poll `/v1/loads` until idle | `weight_version` param; off-policy masked or TIS | preset `--sglang-router-ip/port` → SMG drop-in; `/workers` CRUD compat |
| **verl** | `LLMServerClient.generate(prompt_ids) → TokenOutput{token_ids, log_probs, routed_experts, stop_reason}` | `CheckpointEngineManager` (naive/nccl/nixl/delta_sharded) | `pause_generation(mode=abort)` → partials with `stop_reason=aborted` | `set_global_steps` stamped per response; `min/max_global_steps` per trajectory | custom `client_cls` for `LLMServerManager.get_client()` |
| **TRL** | `POST /generate/ {prompts: [[int]]} → {completion_ids, logprobs}` | `/init_communicator/` + `/update_named_param/` + NCCL bcast (client = last rank) | n/a (sync) | none | implement TRL's 9 endpoints; SMG as `vllm_server_base_url` |
| **SkyRL** | data plane behind one `external_proxy_url` (OpenAI chat/completions + Tinker-style `/inference/v1/generate`, `X-Session-ID` affinity) | control plane fanned to `external_server_urls`: `/init_weight_transfer_engine`, `/update_weights`, `/get_world_size` | `/pause?mode=abort\|keep\|wait` + `/resume`; abort returns partials, client stitches (`AccumulatedResponse`) | client-side `weight_version` counter; `max_staleness_steps` | SMG serves both proxy and control URLs; cleanest external seam of all frameworks |
| **AReaL** | `ModelRequest{input_ids} → ModelResponse{output_tokens, output_logprobs, output_versions}`; rid-sticky routing for KV reuse | SGLang verbs with `abort_all_requests: true` (vLLM via patched `/areal_*` endpoints) | pause aborts in-flight; client loop re-issues `input_ids + partial` to the same server | **per-token** `output_versions` (version appended per resume segment); `max_head_offpolicyness` admission formula | implement `RemoteInfBackendProtocol` (~10 pure request/response-mapping functions) against SMG endpoints |
| **OpenRLHF / NeMo-RL** | Ray-actor tensor contracts, no HTTP | NCCL / CUDA-IPC / ZMQ in-process | sleep/wake around steps | trajectory age drop | out of scope v1 (no HTTP seam); revisit via shims |

Common denominator (the "minimal adoptable surface"): tokenized generate with
logprobs and abort-partials; SGLang-dialect weight verbs plus vLLM-native
verbs; sgl-router-compatible worker CRUD; optional/disableable gateway health
checking (slime does its own); reward-model pools as ordinary multi-model
routing.

## Design overview

```
                    ┌────────────────────────────────────────────────┐
 Trainer ──────────▶│  RL Control API   /v1/rl/*  (admin-auth)      │
 (verl/slime/…)     │   • fleets, versions, update jobs, drain      │
                    ├────────────────────────────────────────────────┤
                    │  Fleet Weight-Update Workflows (wfaas DAG)     │
                    │   quiesce → transfer-window → flush → flip     │
                    ├────────────────────────────────────────────────┤
                    │  EngineControl adapter (per ConnectionMode)    │
                    │   SGLang HTTP/gRPC │ vLLM native │ TRT coarse  │
                    ├────────────────────────────────────────────────┤
 Rollout ──────────▶│  Data plane (existing routers/policies)       │
 requests           │   + version stamping + rollout QoS class      │
                    └────────────────────────────────────────────────┘
                         │            │             │
                      policy pool   policy pool   reward pool
                      (vLLM)        (SGLang)      (any)
```

Five new pieces, all riding existing subsystems:

1. **`EngineControl` trait** — normalizes the verb table above per engine.
2. **RL fleet model** — worker groups with roles (`policy` / `reward` /
   `draft`) and a **weight-version registry**.
3. **Fleet weight-update workflows** — DAG orchestrations for colocated and
   disaggregated update choreographies, with drain strategies.
4. **RL control API** — `/v1/rl/*` admin endpoints exposing 1–3.
5. **Data-plane extensions** — universal version stamping, rollout QoS class,
   MoE expert passthrough, contract tests.

## Detailed design

### 1. `EngineControl` trait and adapters

New trait alongside the existing `Worker` admin ops (which already dual-
dispatch HTTP/gRPC for `flush_cache`, `start_profile`, `stop_profile`):

```rust
#[async_trait]
pub trait EngineControl: Send + Sync {
    /// What this engine supports; probed at registration, cached in labels.
    fn capabilities(&self) -> RlCapabilities;

    async fn sleep(&self, tags: MemoryTags) -> Result<()>;          // weights | kv | all
    async fn wake(&self, tags: MemoryTags) -> Result<()>;
    async fn pause(&self, mode: PauseMode) -> Result<PauseReport>;  // Abort | Drain | Freeze
    async fn resume(&self) -> Result<()>;
    async fn abort_all(&self) -> Result<AbortReport>;               // returns num aborted
    async fn flush_kv(&self) -> Result<()>;
    async fn wait_idle(&self, timeout: Duration) -> Result<()>;     // poll loads until 0 in-flight

    /// Weight update entry points. Tensors never transit SMG.
    async fn begin_weight_update(&self, spec: &WeightUpdateSpec) -> Result<()>;
    async fn finish_weight_update(&self, version: &str) -> Result<()>;
    async fn init_transfer_group(&self, rendezvous: &NcclRendezvous) -> Result<()>;
    async fn destroy_transfer_group(&self, group: &str) -> Result<()>;

    async fn weight_version(&self) -> Result<Option<String>>;
}
```

`PauseMode` maps to engine dialects: `Abort` → SGLang `pause_generation
{mode: abort}` / vLLM `/pause?mode=abort`; `Drain` → `retract` / `wait`;
`Freeze` → `in_place` / `keep`. Adapters:

- **SGLang adapter**: native HTTP endpoints (or gRPC where the servicer
  exposes them). `finish_weight_update` is a no-op (SGLang auto-flushes
  unless told otherwise); SMG still verifies via `GET /get_weight_version`.
- **vLLM adapter**: native RL APIs when detected
  (`/pause`, `/resume`, weight-transfer endpoints); falls back to dev-mode
  endpoints (`/sleep`, `/wake_up`, `/reset_prefix_cache`, `/collective_rpc`)
  with an explicit capability flag so operators know which path is active.
  `finish_weight_update` on the legacy path **always** calls
  `/reset_prefix_cache` (the known stale-KV footgun).
- **TRT-LLM adapter**: coarse-grained — `pause(Drain)` via gateway-side
  draining (stop routing + `wait_idle`), weight updates delegated to the
  external orchestrator, `capabilities()` reports what's absent so workflows
  degrade gracefully.
- **TokenSpeed adapter**: co-designed; the clean-slate opportunity to expose
  the full verb set over gRPC from day one (see §6, proto changes).

Control verbs are **worker-addressed, never load-balanced**. Fleet scope is
achieved by explicit fan-out (`admin_fan_out`, already in
`worker/manager.rs`), with per-worker success/failure reporting.

### 2. RL fleet model and weight-version registry

**Worker roles.** `WorkerSpec` gains `rl_role: Option<RlRole>`
(`policy | reward | draft`), defaulting to `policy` for fleets. Reward pools
are ordinary multi-model routing (register RM workers under their own
`model_id`), but the role lets operators scope control verbs ("pause policy
pool only") and lets health/CB policy differ per role.

**Version registry.** New `WeightVersionRegistry` (in-memory, mesh-synced via
the existing CRDT KV like worker state):

```text
fleet "policy-qwen3" :
  target_version: "step-4200"
  workers:
    w1 → { version: "step-4200", state: Serving }
    w2 → { version: "step-4100", state: Updating }
  history: [ {version, started_at, completed_at, strategy, outcome} ]
```

- Per-worker version lives in `labels["weight_version"]` (already read by
  `dispatch_metadata.rs`), but transitions go through the registry so they're
  atomic with routing changes — the registry updates the label via the
  existing `register_or_replace` (which preserves runtime state), not via
  user-facing PATCH.
- **Version-aware routing modes** (per model, config or per-request header
  `X-SMG-Weight-Version`):
  - `any` (default; today's behavior),
  - `latest-only` — only workers at `target_version` are routable; mid-update
    others are effectively quarantined,
  - `pinned:<v>` — for evaluation replays,
  - `max-staleness:<k>` — AReaL-style admission: route to workers within k
    versions of target; otherwise queue or 503 with `Retry-After`.
- **Universal response stamping**: SMG stamps `weight_version` (and
  `policy_version_at_dispatch`, in case the engine flips mid-request) into
  every response's meta — `meta_info.weight_version` on `/generate`,
  `system_fingerprint` + `metadata` on OpenAI-compat — for **all** engines.
  vLLM/TRT-LLM responses carry no version natively; the gateway is the only
  component that can supply it, which is a headline capability. If the engine
  reports its own version (SGLang), SMG cross-checks and flags divergence.

### 3. Fleet weight-update workflows

New workflow family in `WorkflowEngines` (+ `Job::FleetWeightUpdate`),
following the existing `StepExecutor` pattern. Two choreographies, three
drain strategies.

**Choreography A — colocated (trainer shares GPUs with rollout):**

```text
1. QuiescePool        pause(Abort|Drain per config) on all workers; wait_idle
2. ReleaseMemory      sleep(all) — engines release weights + KV
   … trainer runs its step(s); SMG waits on /v1/rl/updates/{id}/proceed
     or a configured webhook …
3. WakeWeights        wake(weights)
4. TransferWindow     begin_weight_update per worker (disk/tensor/NCCL init);
                      wait for trainer's completion signal
5. FinishUpdate       finish_weight_update (engine flush semantics + verify)
6. WakeKv             wake(kv)
7. FlipVersion        registry: worker → target_version; back to Serving
8. ResumePool         resume; version-aware routing re-admits
```

**Choreography B — disaggregated / async (dedicated rollout fleet):**

Strategies (per update job, `strategy` field):

- `all-at-once` — Choreography A steps 1,4,5,7,8 on the whole fleet. Lowest
  wall-clock, full rollout gap. What slime hand-rolls today.
- `rolling(batch=N)` — take N workers at a time: mark `Draining` (existing
  status: excluded from selection, in-flight completes) → `wait_idle` →
  update → flip → re-admit → next batch. Zero rollout downtime; produces
  bounded version skew, which is why routing mode `max-staleness` exists.
- `blue-green(fraction)` — pre-flip a fraction to the new version while the
  rest serve; new rollouts route `latest-only`; old cohort drains then
  updates. The "drain-and-flip" from the brainstorm: fresh rollouts start on
  new weights immediately, long-tail requests finish on old weights, and
  every response is stamped so the trainer knows which is which.
- Partial-rollout interplay: with `abort_partials: true` the quiesce step
  returns partials to clients (`finish_reason=abort`); clients resubmit and
  cache-aware routing lands them on updated workers. SMG does not buffer
  rollout state server-side in v1 (see Open Questions).

**Failure handling:**

- Any step failure → worker goes `Failed` + version-quarantined (never
  routable at `latest-only`), workflow continues with the rest, terminal
  report lists per-worker outcomes. This mirrors and fixes the slime
  pain points (no more fleet-wide stuck-paused states: pause/continue are
  issued and *verified* per worker, with retries and a reconciliation sweep).
- `KvCacheCleared` events (already consumed by `KvEventMonitor`) are used as
  independent confirmation that a worker's cache actually flushed during an
  update; the kv_index for that worker is reset at the same point, keeping
  cache-aware routing truthful across the flip.
- Elastic join: a worker registering mid-training gets `Pending` +
  registry-version `unknown` → workflow step can trigger
  `update_weights_from_disk` (or checkpoint-engine P2P) to bring it to
  `target_version` before it becomes `Ready`. This is the checkpoint-engine
  interop point: SMG triggers `ParameterServer.update(ranks=[...])`-style
  joins; it never moves tensors.

### 4. RL control API (`/v1/rl/*`, control-plane auth)

```
POST   /v1/rl/fleets                          define fleet {model, role, workers|selector}
GET    /v1/rl/fleets/{fleet}                  state incl. per-worker versions
POST   /v1/rl/fleets/{fleet}/pause            {mode}          fan-out + verify
POST   /v1/rl/fleets/{fleet}/resume
POST   /v1/rl/fleets/{fleet}/abort            {scope: all}    returns per-worker counts
POST   /v1/rl/fleets/{fleet}/sleep            {tags}
POST   /v1/rl/fleets/{fleet}/wake             {tags}
POST   /v1/rl/fleets/{fleet}/flush_kv
POST   /v1/rl/fleets/{fleet}/updates          start update job:
        { target_version, method: disk|tensor|distributed|external,
          strategy: all-at-once|rolling|blue-green, params: {...},
          abort_partials: bool }
GET    /v1/rl/updates/{job}                   step-level progress (workflow events)
POST   /v1/rl/updates/{job}/proceed           trainer-side barrier release
DELETE /v1/rl/updates/{job}                   cancel → reconcile to consistent state
GET    /v1/rl/versions/{model}                version registry view
```

Notes:

- Update jobs ride the `JobQueue` → workflow engine path; `GET` progress is
  served from workflow `EventBus` subscriptions (step-level granularity
  exists today).
- For `method: distributed`, the job body carries the NCCL rendezvous
  (`master_address`, `master_port`, `world_size`, `group_name`, per-worker
  `rank_offset` assignments) — SMG computes and distributes rank offsets
  across the fleet, which every framework currently hand-computes.
- For `method: external` (checkpoint-engine or trainer-driven transfers), SMG
  only does quiesce/flip/verify around a trainer-signaled window — the
  minimal-trust mode frameworks can adopt first.
- Per-request abort (`POST /v1/rl/requests/{rid}/abort`) is included for
  engines that support it (SGLang `rid`; vLLM via tracked-connection drop
  using the existing `AbortOnDropStream`/inflight tracker) — closing the gap
  verl asks for in [verl#6866](https://github.com/volcengine/verl/issues/6866).

### 5. Data-plane extensions

- **Rollout QoS**: map RL traffic to the existing priority scheduler —
  a `rollout` class below `default` (extend the `Class` enum) so co-tenant
  serving+rollout fleets (ROSE-style) keep SLO traffic first; rollout tenants
  clamped via existing `TenantPolicy`.
- **Contract tests per engine** (e2e_test): `input_ids` round-trip fidelity;
  logprob presence/alignment under temperature; abort returns partials with
  correct `finish_reason` and logprobs for generated tokens; version stamp
  present; resume-after-abort lands as cache hit (assert `cached_tokens`).
  These become the advertised "rollout-grade" guarantees, and the retry storm
  tolerance test (slime: 60 retries × 1s, timeout=None) belongs here too.
- **MoE expert-routing passthrough** (`return_routed_experts`): plumb through
  `/generate` and gRPC for engines that support it (today: SGLang fork /
  vLLM verl path). Low cost, and mainline sgl-router doesn't carry it —
  slime users currently install a forked router wheel for R3.
- **Sticky routing for multi-turn rollouts**: `X-SMG-Routing-Key` already
  exists (consistent-hashing policy). Document it as the rollout session key;
  cache-aware policy remains the default recommendation.

### 6. Protocol / crate changes

- `crates/protocols`: `WorkerSpec { rl_role }`, `WeightUpdateSpec`,
  `NcclRendezvous`, `RlCapabilities`, update-job types; extend
  `GenerateRequest` with `return_routed_experts`.
- `crates/grpc_client` protos: add `Pause`, `Resume`, `ReleaseMemory`,
  `ResumeMemory`, `UpdateWeights{Disk,Tensor,Distributed}`,
  `InitWeightTransferGroup`, `GetWeightVersion` RPCs to `common.proto` /
  per-engine protos, implemented first in the SGLang servicer and TokenSpeed
  (co-design), HTTP fallback elsewhere. `GenerateComplete.weight_version`
  already exists.
- `model_gateway`: `EngineControl` impls on `BasicWorker` (dual dispatch like
  `flush_cache`); `WeightVersionRegistry` (+ mesh adapter); workflow steps +
  `create_fleet_weight_update_workflow`; `/v1/rl/*` routes; `Class::Rollout`;
  version-aware filtering in `get_healthy_worker_indices` (one added
  predicate, policy-agnostic).
- `crates/data_connector` (Phase 4 seam only): reserve
  `extra_columns`/hooks-based trajectory columns (`weight_version`, reward,
  token counts) on stored responses.

### 7. Framework adapters (bindings/python + clients/)

Thin shims, in adoption-priority order:

1. **slime** — zero-code path: SMG already accepts sgl-router worker CRUD and
   `/generate`; validate with `--sglang-router-ip/port` pointed at SMG, then
   contribute an optional "gateway-managed update" mode to slime that replaces
   its bypass-and-poll drain with one `/v1/rl/fleets/{f}/updates` call.
2. **verl** — publish `smg.verl.SmgServerClient` implementing
   `LLMServerClient.generate() -> TokenOutput` (maps `stop_reason`,
   `log_probs`, `routed_experts`, stamps `global_steps` from SMG's version
   registry) for `LLMServerManager.get_client(client_cls=...)`.
3. **TRL** — `smg trl-serve` compatibility surface: the 9 endpoints
   (`/generate/`, `/init_communicator/`, `/update_named_param/`, …) fronting
   an SMG fleet, so `vllm_server_base_url` just points at SMG.
4. **SkyRL** — point `external_proxy_url` at SMG's data plane and
   `external_server_urls` at SMG (which fans control verbs out itself,
   collapsing SkyRL's per-server loop to one target). Note: the older
   `remote_inference_engine_urls` key is removed in current SkyRL main.
5. **AReaL** — contribute an `SmgBackend` implementing
   `RemoteInfBackendProtocol` (~10 pure functions mapping
   generate/pause/resume/weight-update requests to SMG endpoints). AReaL's
   per-token `output_versions` is satisfied by stamping version at dispatch
   and on every resume segment; rid-sticky routing maps to
   `X-SMG-Routing-Key`.
6. **NeMo-RL / OpenRLHF** — explicitly out of scope for v1: their contracts
   are Ray-actor + torch-tensor native (ZMQ/CUDA-IPC refit into engine
   internals), with HTTP only as an environment-facing add-on.

## Rollout plan (phases)

- **Phase 0 — data-plane hardening (small)**: contract tests per engine;
  abort-partials fidelity; routed-experts passthrough; retry-storm tolerance;
  document `X-SMG-Routing-Key` for rollouts. Ships alone as "SMG is
  rollout-grade today."
- **Phase 1 — control verbs (medium)**: `EngineControl` trait + SGLang/vLLM
  adapters; fleet fan-out endpoints (pause/resume/abort/sleep/wake/flush) with
  per-worker verification. Immediately fixes slime's bypass loop and
  sglang#6531/#21235-class problems — and works for vLLM fleets, which no
  router offers.
- **Phase 2 — version registry + update workflows (large)**: registry,
  stamping, version-aware routing, update jobs with all three strategies,
  checkpoint-engine external mode, elastic-join catch-up.
- **Phase 3 — framework shims (medium)**: slime validation + upstream PR,
  verl client, TRL surface, SkyRL/AReaL impls; example configs under
  `examples/rl/`.
- **Phase 4 — trajectory capture (separate RFC)**: rollout trajectory
  storage on the data_connector seam; token-faithful capture from black-box
  agent harnesses (Polar-style); reward joins.

## Competitive positioning

| Capability | SGLang gateway | vLLM router / llm-d / Dynamo | verl in-house | **SMG (this RFC)** |
|---|---|---|---|---|
| Cross-engine (vLLM+SGLang+TRT+TokenSpeed) | no | no | n/a | **yes** |
| Gateway fan-out control verbs | no (flush only) | no | no | **yes, verified per worker** |
| Weight-update orchestration | no (client-side) | no | framework-internal | **yes, 3 strategies** |
| Version registry + version-aware routing | no (passive label) | no | `global_steps` stamping only | **yes** |
| Universal per-response version stamping | SGLang-only | no | n/a | **yes, all engines** |
| Abort with partials at gateway | no (open issue) | no | replica-wide only | **yes + per-request where supported** |
| Reward pools behind same gateway | implicit (multi-model) | no | separate sglang-router | **first-class role** |
| Serve+rollout QoS co-tenancy | no | no | no | **priority classes** |

## Risks and open questions

1. **vLLM dev-mode dependency.** Sleep/`collective_rpc`/`reset_prefix_cache`
   sit behind `VLLM_SERVER_DEV_MODE`; the native RL APIs reduce but don't
   eliminate this. Mitigation: capability probing + documented deployment
   flags; track vLLM RFC #48312 correctness fixes and pin known-good
   versions in docs.
2. **Engine API churn.** These surfaces are young (vLLM native APIs 2026-05;
   SGLang gateway rename churn; TRT-LLM sleep support new). The
   `EngineControl` trait + capability flags exist precisely to absorb this;
   contract tests catch drift.
3. **Trainer-side rendezvous trust.** For `method: distributed`, SMG
   distributes NCCL rendezvous info but cannot verify the transfer happened
   correctly; we rely on engine-reported versions (`get_weight_version`) and
   the trainer's `proceed` barrier. Optional `weights_checker`-style probes
   (hash a sentinel tensor via a designated endpoint) are a Phase 2 stretch.
4. **Server-side partial-rollout brokerage** (gateway buffers aborted
   trajectories and re-issues them, rather than clients): deferred. It
   duplicates framework buffers (slime's data buffer, verl's fully-async
   client) and raises state-ownership questions; v1 keeps SMG stateless on
   the rollout path. Revisit after Phase 3 feedback.
5. **Mixed-version PD disaggregation.** During rolling updates a prefill
   worker and decode worker may briefly disagree on version; KV transferred
   across versions is invalid. Rule: update jobs treat a PD group as one
   atomic unit (drain/flip together). Needs an e2e test.
6. **Class enum extension** (`rollout` priority class) touches the packed
   per-class inflight accounting; small but load-bearing change.

## References

Engine primitives: vLLM sleep-mode docs & RFC #15254, native RL APIs blog
(2026-05-28) & weight-transfer docs & async-RL docs, PR #22587
(`return_token_ids`); SGLang native API docs ("SGLang for RL"), PRs
#6855/#6184 (abort), slime blog (2025-07-09); MoonshotAI checkpoint-engine;
NeMo-RL generation design docs; AReaL paper (arXiv:2505.24298).

Frameworks: verl agent-loop docs & `llm_server.py` & RFCs #5790/#5737/#6866;
slime `sglang_rollout.py` / `server_control.py` / external-rollout-engines
doc; TRL `vllm_serve.py` & vLLM-integration docs; SkyRL `InferenceEngine`
interface docs; HF "async RL training landscape" survey.

Competitive: SGLang Model Gateway docs & source (server.rs route table),
issues #6531, #21235, #11703, #13098 (roadmap), #12780 (Q1 2026), #22949
(Q2 2026); slime issues #1391, #1792.
