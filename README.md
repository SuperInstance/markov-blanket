# markov-blanket

> **Where does an agent end and the world begin? The Markov blanket knows.**

[![crates.io](https://img.shields.io/crates/v/markov-blanket.svg)](https://crates.io/crates/markov-blanket)
[![docs.rs](https://docs.rs/markov-blanket/badge.svg)](https://docs.rs/markov-blanket)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust library for computing Markov blankets in Bayesian networks — the statistical boundary that separates an agent from its environment. Implements blanket identification, conditional independence testing via d-separation, boundary detection, information permeability analysis, belief propagation, and KL divergence computation.

---

## Table of Contents

- [What is a Markov Blanket?](#what-is-a-markov-blanket)
- [Why Does This Matter?](#why-does-this-matter)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Related Crates](#related-crates)
- [License](#license)

---

## What is a Markov Blanket?

In probability theory, the **Markov blanket** of a node in a Bayesian network is the minimal set of nodes that shields it from the rest of the network. Once you know the values of the blanket nodes, the target node becomes conditionally independent of everything else.

Formally, for a node X in a Bayesian network, its Markov blanket consists of:

- **Parents**: nodes that directly influence X
- **Children**: nodes that X directly influences
- **Co-parents**: other parents of X's children (they explain away competing causes)

This concept is central to the **Free Energy Principle** (Karl Friston, 2006), where the blanket separates an agent's internal states from external states. Everything an agent can know about the world must come through its Markov blanket — sensory states (what it observes) and active states (what it does).

```
┌─────────────────────────────────────────────┐
│                External States               │
│         (the environment, hidden)            │
├─────────────────────────────────────────────┤
│         ┌─────────┐   ┌─────────┐           │
│         │ Sensory │   │ Active  │           │
│         │ States  │   │ States  │ ← Blanket │
│         └────┬────┘   └────┬────┘           │
├──────────────┼─────────────┼────────────────┤
│              │             │                │
│         ┌────▼─────────────▼────┐           │
│         │    Internal States    │           │
│         │   (the agent's model) │           │
│         └───────────────────────┘           │
└─────────────────────────────────────────────┘
```

## Why Does This Matter?

**For AI researchers**: The Markov blanket formalizes the agent-environment boundary. If you're building agents that reason about the world, you need to know what they can and cannot perceive.

**For multi-agent systems**: When agents share information, their blankets overlap. Understanding blanket structure tells you which agents can influence each other and through what channels.

**For causality**: d-Separation (implemented here) is the graph-theoretic test for conditional independence — the foundation of causal reasoning. If two variables are d-separated by a conditioning set, they are independent given that set.

**For cognitive science**: The Free Energy Principle treats living organisms as systems that minimize surprise within their Markov blanket. This library makes that computational.

## Architecture

```
markov-blanket
│
├── Graph / Edge           ← Directed Bayesian network representation
│
├── blanket module         ← Core Markov blanket computation
│   ├── markov_blanket()       Single-node blanket
│   ├── markov_blanket_set()   Multi-node blanket
│   ├── d_separated()          Conditional independence via d-separation
│   └── minimal_separating_set() Smallest d-separator
│
├── AgentBoundary          ← Agent-environment boundary detection
│   ├── from_internal()        Construct boundary from internal nodes
│   ├── permeability()         Information flow across boundary
│   └── optimal_boundary()     Find best boundary under constraints
│
├── Factor / BeliefState   ← Probabilistic inference
│   ├── belief_propagation()   Message passing on factor graph
│   ├── kl_divergence()        KL[p || q] computation
│   └── surprise()             Negative log-probability
│
└── Utility
    ├── information_flow()      Directional information measure
    ├── effective_connectivity() Network integration measure
    └── find_paths()            Path enumeration between nodes
```

## Quick Start

```rust
use markov_blanket::{Graph, blanket::markov_blanket, AgentBoundary};

// Build a simple Bayesian network:
//   A → B → C
//   D → B
//   C → E
let mut g = Graph::new(5);
g.add_directed_edge(0, 1); // A → B
g.add_directed_edge(3, 1); // D → B
g.add_directed_edge(1, 2); // B → C
g.add_directed_edge(2, 4); // C → E

// Compute Markov blanket of node B (index 1)
let blanket = markov_blanket(&g, 1);
// Blanket = {A, D, C} (parents + children + co-parents of children)

// Check if two nodes are conditionally independent
use markov_blanket::blanket::d_separated;
let independent = d_separated(&g, 0, 4, &[2]); // A ⊥ E | C?
// true: A and E are d-separated given C

// Detect an agent boundary
let internal = vec![1, 2]; // B, C are internal
let boundary = AgentBoundary::from_internal(&g, &internal);
println!("Boundary permeability: {:.3}", boundary.permeability());
println!("Is node 0 internal? {}", boundary.is_internal(0));
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Graph` | Directed Bayesian network with weighted edges |
| `Edge` | Directed edge with source, target, and weight |
| `AgentBoundary` | Partition of nodes into internal, boundary, and external |
| `Factor` | Probability factor over a set of variables |
| `BeliefState` | Current belief distribution over all nodes |

### Graph Operations

| Method | Returns | Description |
|--------|---------|-------------|
| `Graph::new(n)` | `Graph` | Create graph with `n` nodes |
| `g.add_edge(from, to, w)` | `()` | Add weighted directed edge |
| `g.parents(node)` | `Vec<NodeId>` | Get parent nodes |
| `g.children(node)` | `Vec<NodeId>` | Get child nodes |
| `g.neighbors(node)` | `Vec<NodeId>` | Get all adjacent nodes |

### Blanket & Independence

| Function | Returns | Description |
|----------|---------|-------------|
| `markov_blanket(&g, node)` | `Vec<NodeId>` | Parents + children + co-parents |
| `markov_blanket_set(&g, nodes)` | `Vec<NodeId>` | Blanket for a set of internal nodes |
| `d_separated(&g, x, y, z)` | `bool` | Are X and Y d-separated given Z? |
| `minimal_separating_set(&g, x, y)` | `Vec<NodeId>` | Smallest set d-separating X from Y |
| `all_separated_pairs(&g, z)` | `Vec<(NodeId,NodeId)>` | All pairs d-separated by Z |

### Boundary Analysis

| Method | Returns | Description |
|--------|---------|-------------|
| `AgentBoundary::from_internal(&g, nodes)` | `AgentBoundary` | Construct from internal node set |
| `boundary.permeability()` | `f64` | Information flow across boundary |
| `boundary.is_internal(n)` | `bool` | Is node inside the agent? |
| `optimal_boundary(&g, max)` | `AgentBoundary` | Best boundary under size constraint |

### Belief & Inference

| Function | Returns | Description |
|----------|---------|-------------|
| `belief_propagation(&g, factors)` | `BeliefState` | Run message passing |
| `kl_divergence(p, q)` | `f64` | KL[p ‖ q] divergence |
| `information_flow(&g, src, tgt)` | `f64` | Directional information measure |

## Mathematical Background

### Markov Blanket Definition

For a node X in Bayesian network G, the Markov blanket MB(X) is:

```
MB(X) = Pa(X) ∪ Ch(X) ∪ {Y : ∃Z ∈ Ch(X), Y ∈ Pa(Z) ∧ Y ≠ X}
```

where Pa(X) = parents, Ch(X) = children. The key property is:

```
X ⊥ V \ ({X} ∪ MB(X)) | MB(X)
```

X is conditionally independent of all non-blanket nodes, given its blanket.

### d-Separation

Two nodes X and Y are **d-separated** by a set Z if every undirected path between X and Y is blocked by Z. A path is blocked if it contains a node W where either:

1. W is in Z and the path passes through W as a chain (→ W →) or fork (← W →)
2. W is a collider (→ W ←) and neither W nor any descendant of W is in Z

This is the graph-theoretic criterion for conditional independence in Bayesian networks (Pearl, 1988).

### Free Energy Principle

Friston's FEP reformulates survival as minimizing variational free energy:

```
F = -ln p(s|m) + KL[q(z) || p(z|s,m)]
  = complexity - accuracy
```

The Markov blanket is the interface through which sensory data `s` arrives and active states are expressed. The agent can never directly access external states — only through the blanket.

### Information Flow

Information flow from source S to target T through a graph:

```
I(S→T) = Σ_paths  Π_(edges in path)  w(e)
```

Weighted by edge strengths, this approximates the mutual information I(S; T) under Gaussian assumptions.

## Installation

```bash
cargo add markov-blanket
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
markov-blanket = "0.1"
```

## Related Crates

Part of the **SuperInstance Exocortex** ecosystem:

- **[free-energy](https://github.com/SuperInstance/free-energy)** — Variational free energy computation (F = complexity - accuracy)
- **[active-inference](https://github.com/SuperInstance/active-inference)** — Action selection via surprise minimization
- **[signal-transduction](https://github.com/SuperInstance/signal-transduction)** — Biological signal cascading for agents
- **[morphogenesis](https://github.com/SuperInstance/morphogenesis)** — Turing pattern formation for agent development
- **[cortex-bus-protocol](https://github.com/SuperInstance/cortex-bus-protocol)** — CQRS event bus for inter-agent messaging

## License

MIT © [SuperInstance](https://github.com/SuperInstance)

Part of the [Exocortex](https://github.com/SuperInstance/exocortex) project.
