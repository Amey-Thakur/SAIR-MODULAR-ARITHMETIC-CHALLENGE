# Proposed Repository Architecture

The following directory tree is designed to support months of iterative research, large-scale transformer training, and rigorous evaluation for the SAIR Modular Arithmetic Challenge.

```text
/
├── README.md                      # Hero document outlining the project and reading order
├── LICENSE                        # Open source license (e.g., MIT or Apache 2.0)
│
├── docs/                          # Core repository documentation
│   ├── repository_analysis.md
│   ├── design_rationale.md
│   └── architecture.md            # This file
│
├── competition/                   # Analysis and rules of the challenge
│   ├── analysis.md
│   └── recommendations.md
│
├── literature/                    # Summaries of relevant research papers
│   └── review.md
│
├── research/                      # Theoretical derivations and knowledge base
│   ├── knowledge_base.md
│   └── open_questions.md
│
├── planning/                      # Strategic roadmaps
│   ├── experiment_roadmap.md
│   ├── model_roadmap.md
│   └── milestones.md
│
├── datasets/                      # Data generation and processing
│   ├── dataset_roadmap.md
│   ├── generators/                # Scripts to generate synthetic primes and operands
│   └── loaders/                   # PyTorch dataset/dataloader implementations
│
├── architectures/                 # Neural network designs (Implementation Pending)
│   ├── baselines/                 # Standard Transformer, MLP
│   ├── hybrid/                    # Routing, Scratchpad models
│   └── custom/                    # Novel architectures addressing modular learning
│
├── tokenization/                  # Tokenization strategies (Digit, Char, Byte, Mixed)
│
├── training/                      # Training infrastructure (Implementation Pending)
│   ├── configs/                   # YAML/JSON hyperparameters
│   ├── loops/                     # PyTorch/Lightning training loops
│   └── callbacks/                 # Checkpoint saving, early stopping
│
├── experiments/                   # Isolated experimental runs and ablations
│   ├── archive/                   # Deprecated/failed experiments
│   └── active/                    # Ongoing trials
│
├── evaluation_pipeline/           # Local mimic of the competition judge
│   ├── metrics/                   # Exact match accuracy, bit-wise error
│   ├── behavioral_checks/         # Scripts ensuring no banned primitives are used
│   └── static_analysis/           # AST parsers to verify the inference contract
│
├── sandbox/                       # Isolated environment for model testing
│
├── submission/                    # Artifacts specifically for submission
│   └── submission_roadmap.md
│
├── huggingface/                   # Artifacts for the Hugging Face Hub (Placeholders)
│   ├── integration_plan.md
│   ├── model_cards/               # Templates for final model cards
│   ├── weights/                   # Output destination for final checkpoints
│   └── manifest/                  # Competition manifest files
│
└── branding/                      # Visual identity
    ├── branding_plan.md
    ├── logos/                     # SVG/PNG logos
    └── diagrams/                  # Markdown/Mermaid/SVG architecture diagrams
```

## Directory Responsibilities

- **`research/` vs `literature/`**: `literature/` contains summaries of *external* papers. `research/` contains our *internal* theoretical findings and deductions.
- **`architectures/` vs `experiments/`**: `architectures/` contains clean, reusable model classes. `experiments/` contains specific run scripts binding a model, dataset, and hyperparameter config together.
- **`evaluation_pipeline/` vs `sandbox/`**: `evaluation_pipeline/` computes metrics on model outputs. `sandbox/` strictly ensures the model executes without violating competition rules (e.g. no `%` operator).
