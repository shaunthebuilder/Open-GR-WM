# Open GR --WM
## Product Requirements Document (PRD) v2.1

**Document status:** Active, implementation-aligned  
**Audience:** Product, Design, Operations, Engineering, and non-technical stakeholders  
**Last updated:** 2026-02-10  
**Runtime scope:** Local-only (Mac-first), no cloud inference dependency

**Primary implementation references:**
- `/Users/shantanurastogi/Documents/New project/app.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/chunking.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/extract_text.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/extract_vision.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/canonicalize.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/indexing.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/retrieval.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/evals.py`
- `/Users/shantanurastogi/Documents/New project/graphrag_v2/storage.py`

---

## 1. How To Read This PRD
This document is intentionally written in two layers:

1. Plain-language layer (non-technical friendly): Sections `2` to `7`, `11`, and `15`.
2. Technical validation layer: Sections `8` to `14`.

If you only want to understand what the product does and why it matters, read Sections `2`, `3`, `5`, `6`, and `7`.

---

## 2. Product Context And Positioning
### 2.1 Plain-language summary
Open GR --WM is a local app that turns complex documents and web pages into a reusable "knowledge map" (graph), then lets people chat with that map using evidence-based answers.

In practical terms, users can:
1. Upload PDF files or add web URLs.
2. Build and save a graph workspace on disk.
3. Re-open that graph later without rebuilding from scratch.
4. Ask questions and get answers grounded in cited source chunks.
5. See structured answer helpers like charts and hero metrics.

### 2.2 Why this product exists
Typical chat tools are convenient, but they often lose track of source truth for dense business documents. This product is designed to solve that by:
1. Keeping all processing local.
2. Persisting graph workspaces for reuse.
3. Forcing citation-aware answer behavior.
4. Giving users operational controls over build quality, speed, and storage lifecycle.

### 2.3 Core product promise
"Turn source material into a reusable local knowledge workspace and query it with grounded answers."

---

## 3. Personas And Jobs To Be Done
### 3.1 Personas
1. **Business Analyst**
Uses investor decks and annual reports. Cares about reliable numeric answers with citations.
2. **Research Operator**
Manages multiple graph workspaces. Cares about build controls, persistence, import/export, and disk hygiene.
3. **Non-technical stakeholder**
Needs confidence and clarity without code. Cares about understandable controls and visible progress.
4. **Power user**
Tunes quality vs speed with advanced settings. Cares about chunking, vision routing, and retrieval behavior.

### 3.2 Jobs to be done
1. Build a high-density graph from one or more files/URLs.
2. Add new sources into an existing graph later (augmentation).
3. Persist and manage graph assets in storage.
4. Ask business questions and get sourced answers.
5. Validate whether answers are grounded and trustworthy.

---

## 4. Product Goals, Non-goals, And Constraints
### 4.1 Goals
1. High fact density in graph extraction for text plus visual artifacts.
2. Practical build speed on local Mac hardware.
3. Better retrieval recall for factual/numeric questions.
4. Strict grounded answering with low hallucination risk.
5. Complete graph lifecycle management in UI.

### 4.2 Non-goals
1. Cloud hosting or external model APIs.
2. Deep web crawling by default.
3. Multi-tenant enterprise IAM features.
4. Strong backward compatibility guarantees for legacy graph formats.

### 4.3 Hard constraints
1. Local-only model runtime through Ollama.
2. URL ingestion is explicit and depth `0` only.
3. Streamlit rerun model drives UI execution behavior.
4. Mid-size local models require careful token/context budgets.

---

## 5. System Overview (Plain Language First)
### 5.1 What happens when a graph is built
1. The app reads source content (PDF pages or selected URLs).
2. Content is split into chunks optimized for extraction and retrieval.
3. Text model extracts structured facts.
4. Vision model is used only when needed (charts/images/low-text pages).
5. Facts are cleaned, merged, and deduplicated.
6. Retrieval indexes are built.
7. Graph and indexes are saved to disk as v2 artifacts.
8. Quality report is generated.

### 5.2 What happens when a user asks a question
1. Retrieval finds evidence from chunks and graph neighbors.
2. App computes confidence and allowed citations.
3. If confidence is too low, app returns "not found".
4. If confidence is sufficient, reasoning model drafts answer.
5. Artifacts are parsed into visuals and hero figures.
6. Rule-based grounding check and lightweight judge validate output.

### 5.3 Model role specialization
1. **Scanner:** `llama3.2:latest`
Used for fast text extraction and lightweight judging.
2. **Vision:** `llama3.2-vision:latest`
Used for selective chart/image/table fact extraction.
3. **Brain:** `deepseek-r1:14b`
Used for final answer synthesis.
4. **Embeddings:** `all-MiniLM-L6-v2`
Used for fast local semantic retrieval.

---

## 6. End-to-End User Journeys
### 6.1 Build a new graph from PDF
1. Choose `PDF` source type.
2. Upload file.
3. Configure build mode and vision routing (optional).
4. Click `Build`.
5. Observe progress, throughput, ETA, and logs.
6. On completion, graph is auto-saved and ready for chat.

### 6.2 Build a new graph from URL(s)
1. Choose `Web URL` source type.
2. Provide primary URL and optional additional URLs.
3. Optional `Preview URL` for sanity check.
4. Click `Build`.
5. App processes only provided URLs (no crawl depth expansion).

### 6.3 Augment an existing graph
1. Load an existing graph.
2. Switch build target to `Augment loaded graph`.
3. Add new source (PDF or URL).
4. Click `Augment`.
5. Graph is updated and saved with merged source lineage.

### 6.4 Chat on active graph
1. Open `Neural Chat`.
2. Ask question.
3. App retrieves evidence and plans response.
4. User sees structured answer with tabs: visuals, key insights and figures, sources.
5. Chat history persists by session.

### 6.5 Storage lifecycle
1. Load graph into RAM.
2. Unload RAM without deleting disk artifacts.
3. Export graph archive.
4. Import graph archive.
5. Delete graph from disk when no longer needed.

---

## 7. Complete UI Feature Catalog (Every Visible Feature)
This section is the functional inventory of user-visible controls and panels.

### 7.1 Sidebar: `PAST SESSIONS`
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| `+ New Session` | Creates a fresh chat session and makes it active | Lets users separate conversations by topic | More sessions increase list management overhead | Multi-threaded analysis on same graph |
| Session button (title) | Opens a specific saved session | Quick context switching | Streamlit rerun on switch can feel momentary | Resume earlier question threads |
| Session title ellipsis | Truncates long titles for compact display | Keeps list readable in narrow sidebar | Full title only visible via tooltip/help | High-density session navigation |
| Session delete `X` | Deletes one session | Prevents clutter | Deletion is destructive for that session | Clean, focused session list |
| Auto fallback session | Creates a new session if user deletes last one | Prevents empty-state lock | Adds one default session automatically | Always-ready chat workspace |

### 7.2 Tab 1: `Graph Nexus` layout
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Two-column split (`1 : 2.5`) | Control deck on left, activity/graph on right | Keeps operations and output visible together | Less width for left-side forms | Better operational ergonomics |
| Module cards and deck titles | Groups controls by workflow stage | Reduces confusion for non-technical users | Requires more UI discipline to maintain | Guided interaction path |

### 7.3 `SOURCE DATA` module
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Source type selector (`PDF` / `Web URL`) | Switches ingest mode | Single entry point for two workflows | Different validation logic per mode | Flexible source onboarding |
| PDF uploader | Loads PDF bytes and extracts page text | Main path for enterprise reports | Extraction quality depends on PDF text layer | Rapid document ingestion |
| URL primary input | Captures main URL | Supports web-only workflows | Depends on URL availability/content type | Web-to-graph ingestion |
| Additional URLs textarea | Adds multiple URLs line-by-line | Batch ingest without deep crawling | Manual list curation required | Multi-page domain bundles |
| `Preview URL` | Fetches and previews parse summary | Avoids costly failed full builds | Preview is limited (not full graph build) | Early quality check |
| `Clear Preview` | Removes preview state | Prevents stale UI confusion | None | Cleaner operator loop |
| Preview stats (title/segments/images) | Shows parse readiness signal | Builds trust in source parsing | Heuristic quality indicator, not final guarantee | Better build decision-making |

### 7.4 `GRAPH OPERATIONS` module
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Saved graph selector | Chooses existing graph workspace | Entry point for lifecycle actions | Large lists can become long | Workspace switching |
| `Load` | Loads selected graph from disk to RAM | Needed for chat and augmentation | Uses RAM proportional to graph size | Immediate reuse of past builds |
| `Unload RAM` | Clears loaded graph from memory only | Memory control on local machine | Requires re-load for chat | Safe memory management |
| `Delete Disk` | Permanently removes graph directory | Storage governance | Destructive action | Reclaim disk space |
| `Export Archive` | Exports graph as `.opengrwm.zip` | Backup and transfer | Export requires complete required files | Portability and restore |
| Import uploader + `Import Archive` | Imports archived graph | Recovery and cross-machine continuity | Archive format validation required | Long-lived knowledge continuity |
| `Auto-load after import` | Optionally loads imported graph immediately | Reduces clicks | Might auto-switch active workspace | Faster restoration flow |

### 7.5 `ADVANCED CONFIGURATION` expander
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Build target (`Create new` / `Augment`) | Controls whether pipeline creates or merges | Enables incremental graph growth | Augment requires active loaded graph | Continuous workspace expansion |
| Graph name input | Sets workspace display name and slug source | Human-readable catalog management | Name collisions resolve by unique suffix | Organized graph library |
| Build mode (`Fast/Balanced/Thorough`) | Controls chunk and extraction budgets | Operator control over latency vs quality | Faster modes may reduce density; thorough is slower | Tunable build behavior |
| `Use Vision Routing` | Enables selective vision tasks | Better fact recall for visual-heavy pages | Adds runtime and model cost | Visual fact capture |
| `Vision text limit (chars)` | Routes low-text pages to vision | Smarter page-level routing | Too low misses visuals; too high adds overhead | Better routing precision |
| `Max vision artifacts per source` | Hard cap on vision tasks | Runtime predictability | Cap may skip lower-ranked artifacts | Build-time cost control |
| `Vision detail level` | Render scale for vision extraction | Better readability for chart OCR | Higher scale increases compute cost | Better visual extraction quality |
| `Load most recent graph on startup` | Auto-loads last graph at app launch | Reduces repetitive setup | Can load stale context if user forgot | Faster session resume |

### 7.6 `EXECUTION` module
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| `Build` / `Augment` | Starts pipeline based on target mode | Primary production action | Streamlit rerun loop means staged updates | End-to-end graph creation |
| `Stop` | Requests graceful stop | Prevents long unwanted runs | Stops at safe boundary, not hard kill | Controlled interruption |
| `Clear RAM` | Clears runtime graph/chat/build memory state | Recovery path from bad state | Does not delete disk artifacts | Fast reset without data loss |

### 7.7 Right deck: `SYSTEM ACTIVITY`
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Progress bar | Shows processed tasks over total | Visibility into long builds | Dependent on correct task accounting | Build transparency |
| Throughput and ETA | Shows tasks/sec and estimated remaining time | Operational planning | ETA is estimate, not guarantee | Better user expectation management |
| Terminal log panel | Rolling operation logs in console style | Supports debugging and trust | Verbose logs can overwhelm non-technical users | Action traceability |
| Status bubbles | Short-lived success/warn/processing signals | Immediate action feedback | Ephemeral by design (~2.2s) | Confident interaction feedback |

### 7.8 Right deck: `ACTIVE KNOWLEDGE LATTICE`
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Interactive graph widget | Displays node-edge network | Lets users inspect relationship structure | Dense graphs can look visually busy | Concept map navigation |
| Data pills (`Entities`, `Relations`, `Saved Graphs`) | Quick graph stats at glance | Fast sanity checks | Counts do not guarantee quality alone | Health snapshot |
| `Quality Report` expander | Shows build eval metrics JSON | Build quality transparency | Raw JSON can feel technical | Evidence-based quality review |

### 7.9 Tab 2: `Neural Chat`
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Chat title with active graph name | Shows current workspace context | Prevents querying wrong graph | If no graph loaded, chat is blocked | Context-aware chat |
| `Clear Current Session` | Clears only current chat history | Clean restart without deleting graph | Action applies only to active session | Session hygiene |
| User bubble (right aligned) | Displays user question | Clear role separation | None | Readable conversation flow |
| AI container (left aligned) | Displays assistant answer block | Consistent output framing | More structured layout complexity | Rich answer UI |
| Internal answer tabs | Organizes answer output (`Visuals`, `Key Insights & Figures`, `Sources`) | Avoids one long wall of text | Depends on robust artifact parsing | Scannable analytical output |
| Live reasoning expander | Shows model thinking trace when available | Better waiting experience and transparency | Large traces need truncation caps | Improved trust during latency |
| Chat input | Sends a new prompt to retrieval + reasoning pipeline | Primary user interaction | Requires loaded graph | Graph-grounded Q and A |

### 7.10 Answer block internals
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Main summary extraction | Shows concise direct answer | Fast comprehension | Depends on answer parser quality | Executive-friendly response |
| Visual renderer | Renders parsed visualization JSON as chart/table | Converts model output to readable visuals | Falls back when payload invalid | Immediate trend comprehension |
| Hero figures as metrics | Displays top KPI cards | Highlights headline numbers | Needs correctly parsed `label/value/delta` | KPI-first review |
| Insights list extraction | Shows bullet insights from answer body | Improves readability | Quality tied to model output format | Decision-ready synopsis |
| Sources tab | Lists extracted citations | Makes grounding visible | Missing/invalid citations trigger stricter handling | Auditability |

### 7.11 Visual design and interaction layer
| Feature | What it does | Why important | Constraint / Tradeoff | Unlock |
|---|---|---|---|---|
| Dark, high-contrast theme | Improves readability for dense text | Low eye strain in long sessions | Requires consistent contrast governance | Better sustained usage |
| Glow accents and motion | Emphasizes active states and events | Faster status recognition | Too much motion can distract if overused | Better action awareness |
| Compact session tabs + ellipsis | Reduces sidebar waste | Fits many sessions cleanly | Full title hidden without hover | High-density history list |

---

## 8. Functional Requirements (FR IDs)
### 8.1 Source and ingestion
- `FR-SRC-001`: App shall support source mode selection between PDF and Web URL.
- `FR-SRC-002`: App shall parse PDF text page-by-page for build input.
- `FR-SRC-003`: App shall normalize URL input and reject invalid URLs.
- `FR-SRC-004`: App shall support multiple explicit URLs via textarea list.
- `FR-SRC-005`: App shall provide URL preview metadata before build.
- `FR-SRC-006`: URL ingest shall remain depth `0` only.

### 8.2 Build and augmentation
- `FR-BLD-001`: App shall create new graph workspace from source input.
- `FR-BLD-002`: App shall augment an active loaded graph with additional sources.
- `FR-BLD-003`: App shall auto-save graph artifacts after successful build.
- `FR-BLD-004`: App shall maintain source lineage across augment operations.
- `FR-BLD-005`: App shall expose build mode presets for speed/quality tradeoffs.
- `FR-BLD-006`: App shall support stop and resume-safe checkpoint behavior.

### 8.3 Vision routing and extraction
- `FR-VSN-001`: App shall run text extraction for all eligible text chunks.
- `FR-VSN-002`: App shall route vision extraction selectively using heuristics and limits.
- `FR-VSN-003`: App shall allow user-configured vision text threshold and artifact cap.
- `FR-VSN-004`: App shall support vision detail scale control.

### 8.4 Graph lifecycle and storage
- `FR-STO-001`: App shall load saved graphs into RAM.
- `FR-STO-002`: App shall unload RAM state without deleting disk assets.
- `FR-STO-003`: App shall delete selected graph directories from disk.
- `FR-STO-004`: App shall export graph workspaces as portable archive files.
- `FR-STO-005`: App shall import valid archives and assign unique graph IDs.
- `FR-STO-006`: App shall persist v2 artifacts under `rag_store/<rag_id>/v2`.

### 8.5 Retrieval and grounded answer policy
- `FR-RET-001`: App shall build hybrid retrieval evidence (dense, sparse, graph context).
- `FR-RET-002`: App shall compute retrieval confidence before final answer generation.
- `FR-RET-003`: App shall use strict grounded policy: return not-found when support is insufficient.
- `FR-RET-004`: App shall enforce citation constraints against retrieved evidence.
- `FR-RET-005`: App shall run deterministic intent enrichment (no latency-heavy enrichment LLM path).

### 8.6 Chat and session management
- `FR-CHT-001`: App shall persist chat sessions on disk (`chat_sessions.json`).
- `FR-CHT-002`: App shall support create/switch/delete session operations in sidebar.
- `FR-CHT-003`: App shall preserve user and assistant messages per session.
- `FR-CHT-004`: App shall render assistant output in structured subtabs.
- `FR-CHT-005`: App shall expose reasoning trace when available.

### 8.7 Quality and evaluation
- `FR-EVL-001`: App shall generate build eval report at completion.
- `FR-EVL-002`: Eval report shall include quality metrics and pass/fail checks.
- `FR-EVL-003`: App shall display quality report in Graph Nexus.

### 8.8 UX feedback and observability
- `FR-OBS-001`: App shall show progress, throughput, ETA, and rolling logs while building.
- `FR-OBS-002`: App shall show short-lived status bubbles for key actions.
- `FR-OBS-003`: App shall display active graph summary stats in data pills.

---

## 9. Choice Rationale, Constraints, Tradeoffs, And Unlocks
### 9.1 Major architecture choices
| Choice | Why this choice | Constraint | Tradeoff | Unlock |
|---|---|---|---|---|
| Three-model specialization | Keeps extraction fast and reasoning quality high | Requires multiple local models installed | More orchestration complexity | Better speed-quality balance |
| Dual chunk tracks (extraction vs retrieval) | Different tasks need different context sizes | More metadata and storage | More pipeline complexity | Better extraction density and retrieval recall |
| Two-pass text extraction + numeric boost | Increases fact density in financial documents | Extra extraction work per chunk | Added runtime in quality mode | Richer graph for downstream retrieval |
| Selective vision routing | Uses vision only when likely useful | Heuristic misses are possible | May miss some visual facts if capped | Big speed gains vs full-page vision |
| Hybrid retrieval + graph expansion | Increases recall and context coverage | Needs multiple indexes | More moving parts to tune | More reliable grounded answers |
| Strict grounded confidence gate | Reduces hallucination | More "not found" outcomes | Lower perceived recall when evidence weak | Higher trust in factual responses |
| Deterministic enrichment | Prevents latency spikes | Less creativity than freeform enrichment | Simpler planning output | Stable low-latency query planning |
| Lightweight judge on scanner model | Keeps eval loop cheap and local | Judge quality < large dedicated judge model | False retries possible | Practical quality guardrail on local hardware |
| v2 artifact persistence + cache + checkpoints | Enables reuse and resumability | More files to manage | Storage overhead | Long-term graph lifecycle control |

### 9.2 Local runtime constraints (Mac M1)
1. CPU/GPU shared memory budget limits context and throughput.
2. Vision extraction is the largest latency multiplier.
3. Streamlit rerun cycle can create brief UI lock moments.
4. Long answers may require continuation handling (`done=length` cases).

### 9.3 Legacy compatibility policy
Legacy handling is not the primary target. v2 behavior is normative. Legacy compatibility is best-effort, low-priority, and non-blocking for v2 evolution.

---

## 10. Data Model And Storage Contracts
### 10.1 Persistent storage root
`/Users/shantanurastogi/Documents/New project/rag_store`

### 10.2 v2 graph artifact layout
For each graph ID `<rag_id>`:

```text
rag_store/
  <rag_id>/
    meta.json
    v2/
      manifest.json
      graph.json
      facts.jsonl
      entities.jsonl
      chunks.jsonl
      embeddings.npy
      bm25_index.pkl
      eval_report.json
      extract_cache.sqlite
      build_checkpoint.json   # only while build is in-progress or paused
```

### 10.3 Typed contracts (conceptual)
#### Fact
```json
{
  "subject": "string",
  "predicate": "string",
  "object": "string",
  "value": "string",
  "unit": "string",
  "timeframe": "string",
  "source_ref": "[Chunk ...] or [Vision ...]",
  "confidence": 0.0,
  "source_id": "string",
  "chunk_id": "string"
}
```

#### ChunkRecord
```json
{
  "source_id": "string",
  "page_or_doc_idx": 0,
  "chunk_id": "string",
  "section_title": "string",
  "char_start": 0,
  "char_end": 100,
  "is_vision": false,
  "created_at": "ISO-8601",
  "text": "string"
}
```

#### RetrievalEvidence
```json
{
  "chunk_ids": [1, 5, 9],
  "chunk_lines": ["[Chunk ...] ..."],
  "graph_lines": ["A -[rel]-> B [Chunk ...]"],
  "confidence": 0.0,
  "entity_hits": ["..."],
  "metric_hits": ["..."],
  "allowed_citations": ["[Graph]", "[Chunk ...]"],
  "debug": {"dense_candidates": 0.0}
}
```

#### QueryPlan
```json
{
  "intents": ["direct_lookup"],
  "timeframe": "2025",
  "entities": ["Maersk"],
  "metrics": ["ebitda"]
}
```

### 10.4 Manifest contract (`manifest.json`)
Required keys:
1. `format_version` (expected `2.0`)
2. `rag_id`
3. `graph_name`
4. `created_at`, `updated_at`
5. `build_profile`
6. `models` (`scanner`, `vision`, `brain`, `embed`)
7. `source_files`
8. `timings`
9. `quality_metrics`
10. `counts` (`nodes`, `edges`, `facts`, `retrieval_chunks`, `extraction_chunks`)

### 10.5 Chat persistence contract
File: `rag_store/chat_sessions.json`

Each session stores:
1. `id`
2. `title`
3. `created_at`
4. `updated_at`
5. `messages[]`

Each message may include:
1. `role`
2. `content`
3. `viz` (optional structured visualization payload)
4. `hero_figures` (optional KPI cards payload)
5. `thinking` (optional reasoning trace)

---

## 11. Retrieval And Grounding Policy
### 11.1 Policy statement
The app is strict-grounded by default. If evidence is insufficient, it should say so explicitly instead of fabricating an answer.

### 11.2 Retrieval strategy
1. Dense retrieval over embedding index.
2. Sparse retrieval over BM25 index.
3. Entity and metric priors.
4. Graph neighborhood expansion.
5. MMR selection for diversity and coverage.

### 11.3 Confidence and citation enforcement
1. Confidence score combines retrieval consistency and evidence coverage.
2. Allowed citations are explicitly enumerated.
3. Rule checks reject answers with invalid/missing citations.
4. Retry path attempts correction once.
5. If still unsupported, final output is `not found`.

### 11.4 Why this matters for non-technical users
This design favors trustworthy answers over confident-sounding guesses.

---

## 12. Performance And Quality Requirements
### 12.1 Build profiles
| Profile | Intended behavior | Typical use |
|---|---|---|
| Fast | Prioritize speed, smaller budgets | Quick exploratory builds |
| Balanced | Default compromise | Everyday usage |
| Thorough | Prioritize extraction density and quality | Final-quality build runs |

### 12.2 Practical expectations on local M1
1. Build speed is highly affected by number of chunks and vision artifacts.
2. Vision-heavy sources are slower than text-heavy sources.
3. Throughput and ETA are estimates, not strict guarantees.

### 12.3 Quality metrics tracked
1. `edges_per_page`
2. `none_node_rate`
3. `citation_validity_rate`
4. `numeric_support_rate`
5. `retrieval_recall_at_8_proxy`
6. performance counters (`build_total_sec`, `extract_sec`, `facts_per_sec`)

### 12.4 Default quality checks
1. `edges_per_page >= 6`
2. `none_node_rate == 0`
3. `citation_validity_rate >= 0.95`
4. `retrieval_recall_at_8_proxy >= 0.85`

---

## 13. Error Handling And Recovery
### 13.1 Ingestion errors
1. Invalid URL format: user-facing validation warning.
2. Unsupported content type: build blocked with error message.
3. PDF read errors: user-facing error, no crash.

### 13.2 Build-time issues
1. No tasks generated: build stops with warning status bubble.
2. Stop requested: graceful halt at safe boundary.
3. Partial build interruption: checkpoint retained for resumability.

### 13.3 Chat-time issues
1. No graph loaded: user prompted to load/build graph first.
2. Ollama timeout or read failure: surfaced as explicit error string.
3. Low confidence retrieval: deterministic `not found` response.
4. Invalid model artifact JSON: parser sanitizes and falls back.

### 13.4 Artifact render fallbacks
1. If chart payload invalid, render table or nothing with caption.
2. If hero figures invalid, hide metrics and show fallback caption.
3. If PyVis render fails, show warning and continue app flow.

---

## 14. Security, Privacy, And Local Runtime Constraints
### 14.1 Security model
1. Local-first processing; no mandatory external inference API.
2. Import path validation blocks unsafe archive traversal paths.
3. Data deletion is explicit and user-controlled.

### 14.2 Privacy model
1. Source documents and chat sessions persist locally.
2. No multi-user account separation in current scope.
3. Operators are responsible for local machine security posture.

### 14.3 Operational constraints
1. Local disk usage increases with graph count and cache size.
2. Model availability depends on installed Ollama models.
3. Long-running local tasks require resource monitoring on laptop hardware.

---

## 15. Acceptance Criteria And Deterministic Scenario Matrix
### 15.1 Scenario matrix
| Scenario ID | Preconditions | User action | Expected result | Fail condition | FR coverage |
|---|---|---|---|---|---|
| `AC-001` New PDF build | Valid PDF uploaded | Click `Build` | Build runs, graph saved, stats visible | Build completes without persisted graph | FR-SRC-001, FR-BLD-001, FR-BLD-003 |
| `AC-002` URL build depth 0 | Valid URL provided | Click `Build` | Only provided URLs processed, graph saved | App crawls beyond provided URL list | FR-SRC-003, FR-SRC-004, FR-SRC-006 |
| `AC-003` Augment graph | Graph loaded in RAM | Set target `Augment`, click `Augment` | Existing graph gains new source lineage | Augment starts without loaded graph | FR-BLD-002, FR-BLD-004 |
| `AC-004` Stop build | Build in progress | Click `Stop` | Build pauses gracefully with logs/checkpoint | App hard-freezes or loses all progress | FR-BLD-006 |
| `AC-005` Load/unload | Saved graph exists | Click `Load`, then `Unload RAM` | RAM state toggles, disk remains intact | Unload deletes disk artifacts | FR-STO-001, FR-STO-002 |
| `AC-006` Delete graph | Saved graph exists | Click `Delete Disk` | Graph directory removed | Metadata remains listed as active graph | FR-STO-003 |
| `AC-007` Export/import | Saved graph exists | Export then import archive | Graph restored with valid ID and metadata | Import accepts invalid archive structure | FR-STO-004, FR-STO-005 |
| `AC-008` Low confidence question | Weak retrieval evidence | Ask unsupported question | App returns grounded `not found` | Hallucinated unsupported answer returned | FR-RET-003, FR-RET-004 |
| `AC-009` Citation enforcement | Retrieved citations known | Ask numeric question | Answer contains valid citations or not found | Invalid chunk citations pass unchecked | FR-RET-004 |
| `AC-010` Session persistence | Active graph loaded | Ask question, restart app | Session appears in sidebar and can reopen | Chat history lost unexpectedly | FR-CHT-001, FR-CHT-002, FR-CHT-003 |
| `AC-011` Answer subtabs | Assistant response includes artifacts | Open answer block tabs | Visuals/Insights/Sources render in tabs | Raw JSON leaks into main answer body | FR-CHT-004 |
| `AC-012` Quality report | Build completed | Open `Quality Report` expander | Eval metrics displayed | No eval output available after successful build | FR-EVL-001, FR-EVL-002, FR-EVL-003 |

### 15.2 PRD quality gates
1. Every visible control in UI has a documented function and user value.
2. Every major architecture choice has explicit constraint and tradeoff.
3. Every core behavior maps to FR IDs and acceptance rows.

---

## 16. Risks, Limitations, And Future Roadmap
### 16.1 Current risks and limitations
1. Visual extraction quality depends on image clarity and chart complexity.
2. Strict grounding may increase "not found" outcomes for borderline context.
3. Streamlit rerun model can briefly feel non-linear during long actions.
4. Very large graphs can challenge readability and local memory limits.

### 16.2 Planned improvement themes
1. Better graph density tuning by domain-aware extraction templates.
2. Stronger citation parser hardening for malformed model payloads.
3. More deterministic visualization schema handling.
4. Improved retrieval observability for debugging recall gaps.

---

## 17. Glossary (Simple Language)
1. **Graph workspace**: A saved knowledge map plus indexes for one named project.
2. **Chunk**: A small text segment used for extraction or retrieval.
3. **Fact**: A structured record like subject-predicate-object with source reference.
4. **Vision artifact**: An image/table/chart passed to the vision model.
5. **Grounded answer**: An answer supported by retrieved source evidence.
6. **Not found**: Safe fallback when evidence is not strong enough.
7. **Build profile**: Speed-vs-quality preset (`Fast`, `Balanced`, `Thorough`).
8. **Augment**: Add new sources into an existing graph workspace.
9. **Checkpoint**: Saved build progress used for resume safety.
10. **Evidence bundle**: Retrieved chunks plus graph context passed to answer generation.

---

## Appendix A: Public Interface Changes In This PRD Update
This deliverable is documentation-only.

1. Runtime API changes: `None`.
2. Storage format changes: `None` introduced by this doc update.
3. Product behavior changes: `None` introduced by this doc update.

---

## Appendix B: Scope Notes
1. This PRD documents current active implementation behavior.
2. Legacy compatibility is intentionally low priority and non-normative.
3. Roadmap items are informational and not contractual requirements.
