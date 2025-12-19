## Summary: Building the Intent Engine at Instacart

I recently read a deep engineering blog from Instacart that explains how they rebuilt their **query understanding system** using **large language models (LLMs)** to improve search across millions of users. The core challenge they were solving was understanding what users *mean*, not just what they type — especially for ambiguous, noisy, or long-tail queries.

Instacart’s legacy query understanding stack relied on many task-specific ML models (classification, rewriting, entity extraction, etc.). This approach struggled with sparse training data, long-tail queries, and system complexity. Because search happens before conversion, high-quality labeled data was limited, and maintaining many independent models slowed iteration and increased operational overhead.

---

### Why LLMs Made Sense

LLMs offered several advantages:
- Strong general language understanding and world knowledge  
- Better handling of ambiguity and rare queries  
- The ability to consolidate multiple query understanding tasks into a single backbone  
- Reduced feature engineering and faster iteration cycles  

Instead of treating LLMs as a drop-in replacement, Instacart designed a **layered strategy** to make them reliable and scalable in production.

---

### The Intent Engine Architecture

Instacart’s Intent Engine is built around three progressively more specialized techniques:

#### 1. Context Engineering (RAG)
Rather than relying on raw prompting, Instacart injects domain-specific context into prompts. This includes product catalog signals, popular conversions, and taxonomy information. This grounds the LLM’s outputs in Instacart’s business reality instead of generic language patterns.

#### 2. Guardrails and Post-Processing
LLM outputs are validated against structured constraints like category hierarchies and allowed attributes. This step reduces hallucinations and ensures the model’s predictions align with Instacart’s internal schemas.

#### 3. Fine-Tuning
For tasks that require deep domain expertise and low latency, Instacart fine-tunes smaller models using proprietary data. This moves domain knowledge from prompts into the model weights themselves, improving consistency and efficiency at scale.

This progression reflects a spectrum from general-purpose prompting to fully specialized models.

---

### How the System Is Used

#### Query Category Classification
Instacart maintains a large hierarchical product taxonomy. The new system retrieves candidate categories based on historical signals and uses an LLM to rerank them. Additional semantic filtering removes irrelevant results, leading to better precision and recall compared to legacy classifiers.

#### Query Rewriting
Query rewrites improve recall by generating alternate expressions of user intent. Instacart uses specialized prompts for different rewrite types, such as broader queries, substitutes, and synonyms. Outputs are filtered to ensure semantic relevance and correctness. This structured approach significantly improves coverage while maintaining high precision.

#### Semantic Role Labeling (SRL)
SRL extracts structured information from queries, such as product type, brand, and attributes. Instacart uses a hybrid system:
- An offline pipeline generates high-quality annotations for common queries and caches the results.
- A lightweight, fine-tuned model handles real-time inference for long-tail queries.

The offline pipeline also generates training data for the real-time model, allowing it to approximate the quality of larger models at a fraction of the cost.

---

### Production Engineering and Scale

Running LLMs in real time required careful optimization:
- Adapter merging and hardware upgrades reduced latency significantly  
- Caching handled the majority of high-frequency queries  
- Autoscaling helped control GPU costs during off-peak traffic  
- Accuracy was prioritized over aggressive quantization when tradeoffs emerged  

The system improved search quality for rare queries, reduced user friction, and significantly lowered search-related complaints.

---

### Key Takeaways

- Domain-specific context is more valuable than raw model size  
- Offline pipelines are effective for bootstrapping data and reducing online costs  
- LLMs can replace many narrow ML models when paired with guardrails  
- Production success depends as much on systems engineering as model quality  

Overall, this post is a strong example of how LLMs can be thoughtfully integrated into a real-world, high-traffic system by balancing accuracy, cost, latency, and maintainability.

**Original article:**  
https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac
