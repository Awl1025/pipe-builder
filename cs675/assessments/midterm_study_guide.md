# CS 675 — Midterm Study Guide

The midterm covers material from **Weeks 1–7**, with emphasis on the three major lecture topics: **Agent Memory**, **Knowledge Graphs & GraphRAG**, and **Learning in Agentic Systems**. Earlier material (neural network basics, agent fundamentals, embeddings, RAG, LangChain) provides the foundation — you should understand those concepts, but the exam focuses on the later material built on top of them.

This guide is organized by topic. Use it alongside your lecture slides — the slides have diagrams and code examples that will help the concepts click.

---

## Agent Foundations (Weeks 1–4)

These are building blocks. You won't be tested deeply on them in isolation, but they show up as part of bigger questions.

### What Makes Something an Agent

- An agent is more than a chatbot. A chatbot answers questions; an agent **takes actions** in an environment, uses **tools**, and works toward **goals**.
- Key capabilities that make something agentic: tool use, planning, memory, and the ability to act on the real world (not just generate text).
- An LLM alone cannot check a database, call an API, or read a file. **Tools** bridge this gap — they give the agent capabilities the model doesn't have natively.
- When a tool fails (API down, file not found), a well-designed agent needs a **graceful failure strategy** — retry, fallback, or escalate to a human. Crashing silently is the worst option.

### Embeddings and RAG Basics

- **Embeddings** convert text into vectors (lists of numbers) that capture meaning. Similar text → similar vectors.
- **RAG (Retrieval-Augmented Generation)**: instead of stuffing everything into the prompt, you store documents in a vector database, retrieve the most relevant chunks at query time, and feed those to the LLM as context.
- Why RAG instead of just using a huge context window? Even with large windows, **more context = more noise = worse performance**. RAG retrieves only what's relevant.
- The RAG pipeline: chunk documents → embed chunks → store in vector DB → user query → embed query → find similar chunks → feed to LLM → generate answer.

### LangChain and Tool Use

- LangChain provides abstractions for building agent pipelines: chains, tools, prompts, memory.
- Tools are functions the agent can call. Each tool has a **name**, **description** (critical — the LLM reads this to decide when to use it), and **parameters**.
- The LLM doesn't execute tools — it **proposes** tool calls, and the system executes them and returns results.

---

## Agent Memory (Weeks 6–7) ★ Major Topic

This is one of the deepest topics on the exam. Know it well.

### The Statelessness Problem

- Every LLM API call starts from a **blank slate**. The model has zero memory of prior turns. This is a property of the transformer architecture, not a bug that will be patched.
- The simplest "memory" is **message replay**: store all messages in a list, paste the entire list into every prompt. The model sees prior turns as input and responds accordingly.
- This works for prototypes but has production problems: no durability (crash = lost), no atomic updates, no control over what's stored.

### State Design and Reducers

- A **state schema** separates data shape from logic. You define what data looks like (a TypedDict in Python); nodes read state, do work, return updates.
- **Reducers** control how updates merge with existing state. Two fundamental modes:
  - **Append**: new data is added to existing data (e.g., `add_messages` appends new messages to the conversation list)
  - **Overwrite**: new data replaces existing data entirely
- **Append is essential for conversation history.** If you overwrite messages, you lose everything except the last turn. Overwrite is fine for simple values like a status flag or a counter.
- The key insight: nodes don't manage history directly. They return updates, and the reducer (defined at the schema level) handles the merge. Separation of concerns.

### Checkpointing and Threads

- **Checkpointing** saves state after every step, making it durable. The cycle: fetch last saved state → merge with new input → execute nodes → save result.
- Without checkpointing, a crash loses all state. With it, the agent picks up exactly where it left off.
- Three backends on a production-readiness ladder: **MemorySaver** (in-memory, prototyping only), **SQLite** (local apps), **Postgres** (production scale). Same interface — swap one line to upgrade.
- **Thread IDs** namespace state for isolation. Same thread = shared conversation. Different thread = completely independent. Without threads, two users on the same system would see each other's messages.

### Context Window Management

- LLMs have a **finite context window** (a maximum number of tokens). Two failure modes when you exceed it:
  - **Hard failure**: prompt gets truncated
  - **Soft failure**: too much context causes hallucination even before the hard limit
- **Trimming** enforces a token budget. Strategy `'last'` keeps recent messages, drops older ones (recency bias). `include_system=True` preserves the system prompt. `start_on='human'` ensures you don't have an orphaned AI response.
- **Filtering** selects messages by content, not count. Useful for removing specific actors or message types (e.g., remove tool call messages before sending to a summary agent).
- **The filter-merge-trim pipeline**: the order matters.
  1. **Filter first** — remove irrelevant messages
  2. **Merge second** — consolidate consecutive same-type messages
  3. **Trim last** — enforce the token budget on clean, relevant messages
  - If you trim first, you waste your budget keeping irrelevant messages and dropping relevant ones.

### Memory Types (from cognitive science)

- **Episodic memory**: records of specific events — what was said, when. Our entire message-based system is purely episodic.
- **Semantic memory**: extracted facts and general knowledge — "the user's name is Jack" (no timestamp needed). Not implemented in basic message systems.
- **Procedural memory**: learned skills and behaviors — "when the user asks for translations, also provide pronunciation." Not implemented in basic message systems.
- **Consolidation** trades fidelity for durability. Instead of just trimming (discarding old messages entirely), you can compress them:
  - Raw messages → summaries → extracted facts → discarded
  - Trimming is the crudest form of consolidation — zero fidelity preserved.

---

## Knowledge Graphs & GraphRAG (Week 7) ★ Major Topic

### Knowledge Graph Fundamentals

- A knowledge graph stores facts as **triplets**: (head entity, relation, tail entity). Example: (Napoleon, BornIn, Ajaccio).
- **Nodes** = real-world entities (people, places, concepts). **Edges** = semantic relationships. Both can carry properties.
- KGs are **directed, labeled graphs** with semantic edges.
- Graph types to know: undirected, directed, weighted, labeled, multigraph. KG extensions: hierarchical, temporal, multimodal.

### Graphs vs. Tables

- Tables have rigid schemas — missing values create nulls everywhere for sparse data.
- Graphs are flexible — you just add triplets. No nulls. No schema migration needed for new facts.
- Merging two KGs is trivial (combine triplet sets). Merging tables requires join logic and schema alignment.
- Graphs enable **graph algorithms**: shortest path, centrality, community detection. Tables don't.
- KGs represent **relationships explicitly** — embeddings capture similarity but not structured relationships like prerequisites, causes, or part-of.

### Building a Knowledge Graph

- Five-stage pipeline: **Creation** (gather sources, extract entities/relations) → **Assessment** (accuracy, completeness) → **Cleaning** (fix errors) → **Enrichment** (fill gaps) → **Deployment** (host in graph DB, enable queries).
- **LLMs can extract entities and relations** from text without labeled training data — Named Entity Recognition (NER) and Relation Extraction (RE) in a single prompt.
- LangChain's `LLMGraphTransformer` does this in a few lines of code.

### GraphRAG vs. Vector RAG

- **Vector RAG** retrieves text chunks by similarity. Problem: it ignores relationships between chunks, includes redundant noise, and misses global/structural knowledge.
- **GraphRAG** searches the knowledge graph instead. Retrieves entities, relationships, paths, or subgraphs — structured knowledge, not text chunks.
- **Retrieval granularity** matters:
  - **Nodes**: single entities (fast, targeted, may miss context)
  - **Triplets**: entity + relationship + entity (relational)
  - **Paths**: chains connecting entities (multi-hop reasoning)
  - **Subgraphs**: full neighborhoods (rich but risk of noise)
- Simple queries → low granularity. Complex queries → high granularity. Best systems adapt.

### HybridRAG

- Neither vector RAG nor GraphRAG is a silver bullet. **HybridRAG** uses both.
- A **router** analyzes the query and decides: factual lookup → KG. Abstractive/open-ended question → vector store. Complex → both, then fuse results.
- GraphRAG loses text nuance; vector RAG loses structure. Combining them gets the best of both.

---

## Learning in Agentic Systems (Week 6) ★ Major Topic

### The Problem

- Most agents are **stateless and amnesiac** — they make the same mistakes every run because there's no feedback loop from failure to action.
- Agent learning is about closing that loop.

### Nonparametric Learning (change the prompt, not the weights)

Three techniques, in order of complexity:

1. **Exemplar Learning**: Store successful examples, retrieve the most relevant ones at runtime via embeddings. Dynamic retrieval beats fixed few-shot because not all examples are relevant to every input. **Data needed**: a collection of successful interactions.

2. **Reflexion**: After a failure, the agent writes a self-critique — what went wrong, what to try next. Reflections accumulate in a memory buffer and are injected into future prompts. The model coaches itself. **Data needed**: only its own failure logs — no curated examples required.

3. **ExpeL (Experiential Learning)**: Extracts general insights from success/failure *pairs*. Maintains a dynamic rule set — promote, demote, edit, or remove rules over time. Insights transfer across different tasks, not just the one that generated them. **Data needed**: both successful and unsuccessful attempts to compare.

### Parametric Learning (change the model weights)

Three techniques, in order of complexity:

1. **SFT (Supervised Fine-Tuning)**: Train on (prompt, ideal response) pairs. The model learns to replicate exact behavior. LoRA adapters keep it efficient. **Best for**: precise tool calls, structured output, consistent format.

2. **DPO (Direct Preference Optimization)**: Show both preferred and rejected responses per prompt. Model learns quality *ranking*, not just imitation. **Best for**: tone, style, summarization quality — where there's no single "right" answer but clear quality differences.

3. **RLVR (Reinforcement Learning with Verifiable Rewards)**: Generate candidates, grade them with any metric you can build, optimize the model toward higher scores. Most general approach. **Best for**: complex reasoning in high-stakes domains.

### Decision Framework

- **Start nonparametric**, move to parametric only when you have the data, compute, and maintenance capacity.
- The key question: **what data do you have?**
  - Only successful examples → exemplar learning
  - Only failure logs → Reflexion
  - Both successes and failures → ExpeL
  - Hundreds of curated (prompt, response) pairs → SFT
  - Ranked preferences → DPO
  - Any measurable grading metric → RLVR
- These techniques **compose** — you can use exemplar learning alongside a fine-tuned model.
- Don't fine-tune until you've confirmed that prompt engineering and retrieval fall short.

---

## How to Study

1. **Start with the three major topics** — Agent Memory, Knowledge Graphs, and Learning. These are the core of the exam.
2. **Go back to the lecture slides** — the diagrams (especially the memory dependency chain, the GraphRAG comparison, and the learning technique table) are worth studying.
3. **Practice explaining concepts out loud** — if you can explain filter-merge-trim ordering or why Reflexion beats exemplar learning when you only have failure logs, you're in good shape.
4. **Connect the dots** — the exam rewards students who can see how these pieces fit together. Memory enables learning. Knowledge graphs enable better retrieval. Tools enable action. It's all one system.

Good luck!
