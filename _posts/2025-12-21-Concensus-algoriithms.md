---
layout: post
title: Consensus Algorithms, How Distributed Systems Agree in the Presence of Failure
date: 2025-12-20
tags: [system-design]
---

Distributed systems are fundamentally about **coordination under uncertainty**. Machines fail, networks partition, clocks drift, and messages arrive late—or never. Yet systems like Kubernetes, distributed databases, and cloud platforms must still **agree on critical decisions**.

This is where **consensus algorithms** come in.

In this article, we cover:
- What consensus is and why it’s hard  
- Core guarantees provided by consensus  
- Failure models and assumptions  
- Major consensus algorithms (Paxos, Raft)  
- How consensus is used in real systems  
- Practical tradeoffs engineers make  

---

## What Is Consensus?

**Consensus** is the problem of getting a group of distributed nodes to **agree on a single value or sequence of values**, even in the presence of failures.

Systems need consensus to agree on:
- Who the **leader** is
- The **order of operations** in a log
- Whether a transaction is **committed**
- The current **cluster configuration**

> Consensus enables reliable replicated state machines on unreliable infrastructure.

---

## Why Is Consensus Hard?

Consensus is difficult due to fundamental properties of distributed systems.

### 1. No Shared Memory
Nodes communicate only through message passing. Messages can be:
- Delayed
- Reordered
- Duplicated
- Lost

### 2. Partial Failures
Some nodes may fail while others continue running. The system cannot easily distinguish between:
- A slow node
- A dead node

### 3. No Perfect Clocks
Clock skew and drift prevent reliable global ordering based on timestamps alone.

### 4. FLP Impossibility Result
The **FLP theorem** proves that in a fully asynchronous system with even one faulty process, deterministic consensus cannot guarantee termination.

**Practical implication**: Real systems rely on timeouts, retries, and assumptions about eventual stability.

---

## Core Properties of Consensus

A correct consensus algorithm guarantees:

### Safety (Never Wrong)
- Nodes never decide different values
- Once a value is chosen, it is final

### Liveness (Eventually Progress)
- The system eventually reaches a decision (under reasonable conditions)

### Agreement
- All non-faulty nodes decide on the same value

### Validity
- The decided value must be one that was proposed

> Safety is absolute. Liveness is conditional.

---

## Failure Models

Most production consensus systems assume:
- **Crash-stop failures**: nodes stop permanently
- **Crash-recovery failures**: nodes restart and rejoin
- **Non-Byzantine behavior**: nodes are not malicious

Byzantine fault tolerance exists but is significantly more complex and costly.

---

## Consensus as Replicated State Machines

Modern systems use consensus to replicate a **log of commands**:

1. Clients send commands to a leader
2. Commands are appended to a log
3. Consensus ensures the log is replicated consistently
4. Each node applies commands in order

This is known as **State Machine Replication (SMR)**.

---

## Paxos: The Foundation

### Overview
Paxos is a family of algorithms that solve consensus with strong theoretical guarantees. It is:
- Provably correct
- Extremely subtle
- Difficult to implement

### Roles
- **Proposers**: suggest values
- **Acceptors**: vote on proposals
- **Learners**: learn the chosen value

### Phases
1. **Prepare phase**: proposer asks acceptors to promise
2. **Accept phase**: acceptors vote on a value

A value is chosen once a **quorum (majority)** agrees.

### Why Paxos Is Rarely Used Directly
- Hard to reason about
- Difficult to extend
- Easy to implement incorrectly

---

## Raft: Consensus for Practitioners

Raft was designed to make consensus:
- Easier to understand
- Easier to implement
- Easier to reason about operationally

Raft breaks consensus into:
- Leader election
- Log replication
- Safety guarantees

---

## Raft Architecture

Each node is in one of three states:
- **Leader**
- **Follower**
- **Candidate**

At any time:
- At most one leader exists
- All writes go through the leader

---

## Leader Election in Raft

1. Followers start an election if they stop hearing from the leader
2. They become candidates and request votes
3. A candidate with a majority becomes leader

Randomized timeouts reduce split votes.

---

## Log Replication in Raft

1. Leader appends entries to its log
2. Entries are replicated to followers
3. Once a majority acknowledges, entries are **committed**
4. Nodes apply entries to their state machines

---

## Safety Guarantees in Raft

Raft ensures:
- Logs remain consistent across leaders
- Committed entries are never lost
- Leaders cannot overwrite committed entries

This provides **linearizable writes**.

---

## Quorums: The Key Insight

Most consensus algorithms rely on **majorities**.

A quorum is typically:
[N/2] + 1

Why it works:
- Any two quorums intersect
- At least one node always has the latest committed state

---

## Consensus vs Replication

- **Replication** copies data
- **Consensus** decides *what* to replicate and *in what order*

Strong consistency requires consensus.

---

## Performance Tradeoffs

Consensus introduces several performance costs that systems must account for.

| Dimension    | Impact of Consensus |
|--------------|---------------------|
| **Latency**  | Requires multiple network round trips for each write |
| **Throughput** | Leader can become a bottleneck, limiting parallelism |
| **Availability** | A majority of nodes must be reachable; progress can halt during network partitions |

As a result, many systems restrict the use of consensus to **metadata and control planes** rather than high-volume data paths.

---

## Real-World Use Cases

Consensus is commonly used for:
- Leader election
- Configuration management
- Metadata replication
- Distributed locks
- Cluster membership

Examples:
- Kubernetes (etcd)
- CockroachDB
- Consul
- ZooKeeper

---

## When Not to Use Consensus

Avoid consensus when:
- Eventual consistency is acceptable
- Latency is critical
- Workloads are high-volume and write-heavy

Examples:
- Caches
- Metrics pipelines
- Log ingestion systems

---

## Final Thoughts

Consensus algorithms are the **foundation of reliable distributed systems**. They allow systems to function predictably in the face of failures—but they are not free.

A strong engineer understands:
- How consensus works
- What guarantees it provides
- When its tradeoffs are worth paying

> Distributed systems fail by default. Consensus is how we agree on what happens next.
