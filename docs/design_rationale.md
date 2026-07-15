# Design Rationale for the Modular Arithmetic Challenge Repository

## 1. Context and Goals

The Modular Arithmetic Challenge asks participants to train models that organically learn $(a \times b) \pmod{p}$ without relying on hard-coded programming primitives (like modulo operators) or classic symbolic reductions (like Montgomery/Barrett). 

Because this is a deeply experimental research track focusing on phenomena like **grokking**, **representation learning**, and **routing**, the repository cannot follow a standard software engineering template. It must be designed as a **Long-Term Research Laboratory**.

## 2. Core Design Principles

### A. Research Precedes Implementation
The repository structure places `research/`, `literature/`, and `docs/` at the forefront. Code (`models/`, `training/`) exists to test the hypotheses formulated in the research documentation. 

### B. Modularity of Experiments
Grokking and algorithmic learning require hundreds of ablations (e.g., character-level vs. digit-level vs. byte-level tokenization). An `experiments/` directory is designed to house compartmentalized trials, complete with configuration files, run logs, and visualization scripts, ensuring that failed experiments are cleanly preserved without polluting the main model architectures.

### C. The Separation of Inference and Sandbox
The competition defines a strict inference contract and sandbox environment. To mirror this constraint locally, we establish separate `sandbox/` and `evaluation_pipeline/` directories. This ensures that models are tested locally under the exact same zero-fallback constraints they will face on the competition server.

### D. Hugging Face Preparedness
The repository structure explicitly allocates a `huggingface/` sub-hierarchy with `model_cards/` and `manifest/` placeholders. This ensures that when an architecture successfully groks the modular arithmetic, exporting it to the Hugging Face Hub is a seamless, localized operation rather than a messy refactor.

### E. Visual Hierarchy
Taking cues from the Mathematics Distillation repository, visual identity is treated as a first-class citizen in the `branding/` and `assets/` directories, setting a professional standard for open-source AI research.

## 3. Divergence from the Reference Repository
Unlike the Lean 4 formal verification repository (which had distinct `stage1/` and `stage2/` for prompt engineering vs deterministic solvers), this repository is structured around the **machine learning lifecycle**:
- **Theory** (`literature/`, `research/`)
- **Data** (`datasets/`)
- **Design** (`architectures/`, `tokenization/`)
- **Execution** (`training/`, `experiments/`)
- **Validation** (`evaluation_pipeline/`, `sandbox/`)
- **Publication** (`submission/`, `huggingface/`)

This lifecycle dictates the directory tree detailed in our `architecture.md`.
