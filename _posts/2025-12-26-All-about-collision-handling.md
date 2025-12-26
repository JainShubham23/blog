---
layout: post
title: Understanding Hash Table Collision Handling
date: 2025-12-26
tags: [system-design]
---

Hash tables are one of the most widely used data structures in computer science because they offer near constant-time lookup, insertion, and deletion. However, their performance relies heavily on how **collisions** are handled—situations where multiple keys map to the same index in the underlying array.

This article explains the **first seven fundamental collision handling techniques**, how they work, and where each one fits in real-world systems.

---

## 1. Separate Chaining

Separate chaining is the most straightforward collision handling technique.

### How It Works
Each slot in the hash table points to a secondary data structure—traditionally a linked list—that stores all elements hashing to the same index. When a collision occurs, the new key-value pair is added to this structure.

### Performance Characteristics
Lookup time depends on the length of the chain. With a good hash function, this remains close to constant time, but in the worst case it can degrade to linear time.

### Strengths
- Simple to implement  
- Supports load factors greater than 1  
- Deletions are easy  

### Weaknesses
- Extra memory overhead for pointers  
- Poor cache locality as chains grow  
- Vulnerable to hash flooding without protections  

### Real-World Usage
Many systems use optimized chaining. For example, Java’s `HashMap` converts long chains into balanced trees to avoid worst-case behavior.

---

## 2. Open Addressing (Conceptual Foundation)

Open addressing takes a different approach from chaining.

### How It Works
All entries are stored directly in the hash table array. When a collision occurs, the algorithm probes other slots until it finds an empty one.

### Key Properties
- The table must always have free slots  
- Load factor must remain below 1  
- Deletions require tombstones to preserve probe sequences  

Open addressing is a family of techniques, including linear probing, quadratic probing, and double hashing.

---

## 3. Linear Probing

Linear probing is the simplest form of open addressing.

### How It Works
When a collision occurs, the algorithm checks the next slot in sequence until an empty slot is found.

### Performance Characteristics
Linear probing benefits from excellent cache locality but suffers from **primary clustering**, where long runs of occupied slots form and slow down operations.

### Strengths
- Simple and fast in low-load scenarios  
- Excellent cache locality  
- Minimal overhead  

### Weaknesses
- Severe performance degradation at higher load factors  
- Primary clustering  

### Usage Considerations
Linear probing works best when the load factor is kept well below 0.7.

---

## 4. Quadratic Probing

Quadratic probing improves on linear probing by spreading out probe locations.

### How It Works
Instead of checking consecutive slots, the algorithm probes positions at increasing quadratic distances from the original hash index.

### Performance Characteristics
This reduces primary clustering but still suffers from **secondary clustering**, where keys with the same initial hash follow identical probe sequences.

### Strengths
- Better distribution than linear probing  
- Reduced clustering  

### Weaknesses
- More complex than linear probing  
- Requires careful table sizing  
- Secondary clustering still exists  

### Usage Considerations
Quadratic probing is less common in modern systems due to its complexity and limitations.

---

## 5. Double Hashing

Double hashing is one of the most effective open addressing strategies.

### How It Works
A second hash function determines the probe step size, creating a key-dependent probe sequence.

### Performance Characteristics
Double hashing greatly reduces both primary and secondary clustering, resulting in near-uniform probe distributions.

### Strengths
- Excellent performance under moderate load  
- Minimal clustering  
- Predictable behavior  

### Weaknesses
- Requires two hash computations  
- More complex implementation  

### Real-World Usage
Many high-performance hash tables use variants of double hashing.

---

## 6. Robin Hood Hashing

Robin Hood hashing focuses on reducing variance in lookup times.

### How It Works
Each entry tracks how far it has been displaced from its original position. During insertion, entries with shorter probe distances may be displaced by those that have traveled farther.

### Performance Characteristics
This equalizes probe lengths across entries, leading to more predictable lookup times.

### Strengths
- Reduced tail latency  
- Consistent performance  
- Good cache locality  

### Weaknesses
- More expensive insertions  
- Complex deletion handling  

### Real-World Usage
Robin Hood hashing is used in modern libraries such as Rust’s `HashMap` and Google’s Abseil containers.

---

## 7. Cuckoo Hashing

Cuckoo hashing uses multiple hash functions to provide fast lookups.

### How It Works
Each key has multiple possible locations. If all are occupied, an existing key is evicted and reinserted elsewhere, potentially causing a chain of relocations.

### Performance Characteristics
Lookups are guaranteed to be constant time, but insertions may trigger expensive rehashing.

### Strengths
- O(1) worst-case lookup  
- Extremely fast reads  
- Cache-efficient  

### Weaknesses
- Complex insertion logic  
- Possible insertion failures  
- Rehashing overhead  

### Real-World Usage
Cuckoo hashing is commonly used in systems where read performance is critical, such as networking and in-memory databases.

---

## Conclusion

Collision handling is the core determinant of hash table performance. Separate chaining offers simplicity and flexibility, while open addressing techniques prioritize memory efficiency and cache locality. Among probing strategies, double hashing and Robin Hood hashing offer strong general-purpose performance, while cuckoo hashing excels in read-heavy workloads.

Understanding these techniques and their trade-offs enables better design decisions when building or choosing hash table implementations.
