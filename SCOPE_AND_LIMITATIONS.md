# Scope and Limitations

This document clarifies the **intent, scope, and limitations** of the code contained in this repository.

The purpose of this repository is **conceptual demonstration**, not software engineering completeness.

---

## Purpose of This Software

This software was developed **purely as a proof of principle**.

Its primary goal is to demonstrate that **Takens’ delay-coordinate embedding can be used as a viable structural replacement for attention mechanisms in language models**, and that language sequences can be meaningfully treated as trajectories in a reconstructed phase space.

The implementation is intended to support:
- conceptual clarity,
- experimental validation,
- and theoretical discussion.

It is **not intended** to serve as a production-grade system, research benchmark, or general-purpose language modelling library.

---

## Development Context

The author is **not a professional software engineer**.

The code was developed with the assistance of **Large Language Models (LLMs)** and was deliberately constrained to the **minimum complexity required to test the core idea**. Design decisions prioritised:

- transparency over optimisation,
- readability over abstraction,
- and conceptual fidelity over performance.

As such, the code should be understood as a **minimal experimental testbed**, not a polished framework.

---

## Examples and Supporting Material

The repository includes:

- Worked examples demonstrating model training and inference
- Example datasets used to validate the approach
- Pre-trained models corresponding to those examples

These are documented in:

- `examples.md` — description and usage of example models and datasets
- `architecture.md` — architectural overview and module-level breakdown

These materials are provided to ensure the approach can be **inspected, reproduced, and extended**.

---

## Current Capabilities

In its present form, the code supports:

- Training small language models using **Takens-based delay embeddings**
- Continuous text modelling
- Simple question–answer datasets
- An initial “question → bridge → answer” format to demonstrate structured sequence transitions

These capabilities are sufficient to demonstrate the **core phase-space embedding concept**, but are intentionally limited.

---

## Known Limitations

The current implementation has **deliberate and acknowledged limitations**, including but not limited to:

- Simplified text encoding and tokenisation
- Minimal vocabulary handling
- Limited embedding and sequence management strategies
- Absence of advanced optimisation, batching, or scaling features
- No guarantees of numerical stability, efficiency, or robustness

The code has **not** been designed to meet any formal software standard, performance benchmark, or production reliability requirement.

---

## Future Extensions (Out of Scope)

Any serious extension of this work would require, for example:

- More comprehensive text encoding and tokenisation strategies
- Improved handling of long and heterogeneous sequences
- Refined embedding and memory management
- More systematic training and evaluation protocols

These extensions are **intentionally left out** of the present repository in order to keep the focus on the underlying conceptual contribution.

---

## Intellectual and Philosophical Context

This repository is shared in the spirit of **academic openness and public commitment to advancing understanding** in the following areas:

- the nature of language and meaning in AI systems,
- the mathematical structure underlying transformer architectures,
- and the interpretation of language models as dynamical systems.

It also serves as a practical bridge between:
- the **technical mechanics of AI and language models**, and
- the **philosophical framework of Geofinitism**, as developed by the author, Kevin R. Haylett.

The intent is not to assert final answers, but to **open a coherent and testable line of inquiry**.

---

## Summary

In short:

- This code demonstrates **that the idea works**
- It does not claim to show **how far the idea can be pushed**
- It is offered as a **starting point, not a destination**

Readers and developers are encouraged to engage with the work in that spirit.
