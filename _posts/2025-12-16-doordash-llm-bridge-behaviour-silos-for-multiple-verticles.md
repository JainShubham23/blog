---
layout: post
title: How DoorDash Uses LLMs to Bridge Behavioral Silos in Recommendations
---

DoorDash recently published a great article on how they use **Large Language Models (LLMs)** to improve recommendations across multiple business verticals like restaurants, groceries, and retail. Instead of replacing traditional recommender systems, they use LLMs as a **semantic bridge** between otherwise disconnected user behaviors.

> **Original article:**  
> https://careersatdoordash.com/blog/doordash-llms-bridge-behavioral-silos-in-multi-vertical-recommendations/

This post is my detailed breakdown of what DoorDash is doing, why it matters, and what we can learn from it.

---

## The core problem: behavioral silos

DoorDash operates across multiple **verticals**:
- Restaurant delivery
- Grocery delivery
- Convenience and retail

A major challenge is that **user behavior does not transfer cleanly across these verticals**.

For example:
- A user may order restaurant food frequently
- But have little or no grocery order history
- Or their grocery behavior may look completely different

Traditional recommender systems rely heavily on **historical interaction signals** (clicks, orders, ratings). When users haven’t interacted with a vertical, these systems struggle with:
- Cold start
- Sparse data
- Poor cross-vertical generalization

This creates *behavioral silos* where each vertical operates almost independently.

---

## Why traditional recommenders fall short

Classic recommendation pipelines typically look like this:

1. Candidate retrieval (based on past behavior)
2. Ranking (using learned relevance models)
3. Post-processing (business rules, diversity, etc.)

These models work well **within a single domain**, but fail when:
- User intent must transfer across domains
- Behavioral signals are sparse or indirect
- Item vocabularies differ significantly

Ordering sushi doesn’t obviously map to buying groceries — at least not in a way traditional models can easily encode.

---

## DoorDash’s key insight: model intent, not actions

Instead of focusing purely on *what* users did, DoorDash focuses on *why* they did it.

This is where LLMs come in.

LLMs can:
- Understand user behavior at a **semantic level**
- Encode patterns like preferences, routines, and intent
- Generalize across different item types and vocabularies

For example:
- Frequent late-night food orders
- Consistent healthy meal choices
- Large family-sized orders on weekends

These patterns can be expressed as **high-level intent signals**, which transfer more naturally across verticals.

---

## How LLMs are used (high-level architecture)

Importantly, DoorDash does **not** use LLMs as an end-to-end recommender.

Instead, LLMs act as an **augmentation layer**:

1. **Input signals**
   - User behavior (orders, searches, browsing)
   - Context (time, location, device)
   - Item metadata

2. **LLM-based representation**
   - Generate rich semantic embeddings or features
   - Capture cross-domain intent
   - Reduce sparsity

3. **Downstream recommendation models**
   - Retrieval models
   - Ranking models
   - Business logic

The final recommendations are still produced by **traditional scalable ML systems**, but with better features.

---

## Bridging verticals with shared representations

The most interesting part is how LLMs help create **shared representations** across verticals.

Instead of learning separate user models for:
- Restaurants
- Groceries
- Retail

DoorDash can:
- Represent user intent in a unified embedding space
- Allow signals from one vertical to inform another
- Improve recommendations even when direct interaction data is missing

This significantly improves:
- Cold-start performance
- Cross-sell opportunities
- Overall recommendation relevance

---

## Why this approach works in production

Several aspects make this approach practical:

- **LLMs are not in the critical serving path**  
  They enrich features, not handle real-time ranking.

- **Scalability is preserved**  
  Existing retrieval and ranking systems remain intact.

- **Risk is controlled**  
  If LLM features degrade, the system falls back to traditional signals.

- **Interpretability improves**  
  High-level intent features are easier to reason about than raw clicks.

This is a great example of **incremental LLM adoption** rather than a full rewrite.

---

## Key takeaways

1. LLMs shine at **semantic understanding**, not brute-force ranking
2. Modeling **intent** is more transferable than modeling **actions**
3. LLMs work best when paired with classical recommender systems
4. Cross-domain recommendation is a natural fit for language models
5. Production ML systems benefit most from *hybrid architectures*

---

## Why this matters beyond DoorDash

This pattern applies to many domains:
- Job marketplaces (job search vs applications)
- E-commerce (browsing vs purchasing)
- Media platforms (reading vs watching)
- Fintech (transactions vs intent)

Anywhere behavior is fragmented across domains, **LLMs can act as the connective tissue**.

---

## Final thoughts

What makes this DoorDash article compelling is its pragmatism. LLMs are not presented as magic replacements, but as **tools that complement existing systems** by solving a very specific problem: bridging behavioral silos.

This is a strong blueprint for how LLMs should be introduced into mature ML systems — carefully, incrementally, and where they add the most value.

---

**Original article:**  
https://careersatdoordash.com/blog/doordash-llms-bridge-behavioral-silos-in-multi-vertical-recommendations/
