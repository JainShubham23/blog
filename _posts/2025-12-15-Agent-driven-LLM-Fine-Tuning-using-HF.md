---
layout: post
title: Trying Agent-Driven LLM Fine-Tuning with Hugging Face Skills
tags: [machine-learning]
---

I recently tried out Hugging Face’s new **Skills-based fine-tuning workflow**, where a coding agent (like Claude Code) doesn’t just *write* training scripts, but actually **runs end-to-end model training jobs** on Hugging Face infrastructure. The experience was surprisingly smooth and changed how I think about fine-tuning large language models.

> *Based on this article:*  
> https://huggingface.co/blog/hf-skills-training

## What I tried

The core idea is simple: instead of manually setting up GPUs, scripts, and monitoring, you give a **natural language instruction** like:

> Fine-tune Qwen3-0.6B on the open-r1/codeforces-cots dataset.

From there, the agent handles everything:
- Validates the dataset format
- Chooses the right GPU based on model size
- Configures training (SFT, LoRA, etc.)
- Monitors progress using Trackio
- Pushes the trained model back to the Hub

All I had to do was review the configuration and approve it.

## What stood out

The biggest surprise was **how production-ready this feels**. This isn’t a toy demo. The workflow supports:
- Supervised Fine-Tuning (SFT)
- Reinforcement learning methods like GRPO
- Models from small (~0.6B) up to mid-sized ranges using LoRA
- Automatic cost and hardware estimation before you commit

For a small test run, the entire fine-tuning process cost only a few cents and finished in minutes.

## Monitoring and iteration

Once the job was running, I could ask the agent things like:
- *How’s the training going?*
- *What’s the current loss?*
- *Is anything wrong?*

The agent would fetch logs, summarize metrics, and even suggest fixes if something failed (for example, dataset format issues or GPU memory limits). This makes experimentation much safer and cheaper.

## What I learned

The most important takeaway is that **LLM fine-tuning is becoming conversational**. Instead of stitching together scripts, cloud configs, and dashboards, you can focus on *intent*:
- What model?
- What data?
- What behavior?

Everything else becomes automation.

This lowers the barrier significantly for developers who understand models and data, but don’t want to spend hours on infrastructure setup, and also for fast setup fast fail kind of experiments. 

## When I would use this

I’d absolutely use this approach for:
- Prototyping domain-specific models
- Instruction tuning on custom datasets
- Preference alignment experiments
- Small to medium production fine-tuning jobs

For very large models or highly customized training loops, you might still need more manual control — but for most practical cases this workflow is more than capable.

## Final thoughts

Hugging Face Skills plus agentic coding tools feel like a glimpse of the future of ML workflows. Fine-tuning no longer feels like a specialized infrastructure task — it feels like *giving instructions to a capable collaborator*.

That shift alone makes this worth trying.
