<div align="center">

# Documentation

**The reasoning behind the laboratory, in the order it is worth reading.**

[Back to the repository](../README.md) &nbsp;·&nbsp;
[Laboratory](../src/README.md) &nbsp;·&nbsp;
[Competition](https://competition.sair.foundation/competitions/modular-arithmetic-challenge/overview)

</div>

---

Read these in order and the design decisions in `src/` stop looking arbitrary.
Each one answers a question the next one depends on.

## 1. Why the repository is shaped this way

| Document | Question it answers |
| --- | --- |
| [repository_analysis.md](repository_analysis.md) | What a sibling research repository got right, and what was worth carrying over |
| [design_rationale.md](design_rationale.md) | Why research comes before implementation here, and why inference is kept apart from the sandbox |
| [architecture.md](architecture.md) | What each directory is responsible for, and what it is not allowed to know about |

## 2. What the competition actually demands

| Document | Question it answers |
| --- | --- |
| [competition/analysis.md](competition/analysis.md) | The inference contract, the sandbox, the difficulty tiers, and what static analysis will look for |
| [competition/recommendations.md](competition/recommendations.md) | Where the marks are: routing, tokenisation, hardware emulation, curriculum, weight decay |

> [!NOTE]
> These two were written from the rules as published at the time. The official
> interface later settled into three per-argument preprocessing hooks and a
> `predict_digits` call that returns base-B digits, with the decoder owned by
> the pipeline. The [root README](../README.md) carries the final shape.

## 3. The mathematics and the prior work

| Document | Question it answers |
| --- | --- |
| [research/knowledge_base.md](research/knowledge_base.md) | Why this is hard for a network at all: discrete logarithms, Fourier representations, Horner emulation, tokenisation |
| [literature/review.md](literature/review.md) | Grokking, mechanistic interpretability, in-context algorithmic reasoning, length generalisation, routing |
| [research/open_questions.md](research/open_questions.md) | What is still unsettled, including the optimal scratchpad radix and whether grokking survives scale |
| [paper_summary.md](paper_summary.md) | The whole line of work, written as a short paper |

## 4. From theory to something submittable

| Document | Question it answers |
| --- | --- |
| [../src/datasets/dataset_roadmap.md](../src/datasets/dataset_roadmap.md) | How the primes, the curriculum and the scratchpad traces are generated, and in what order |
| [../src/submission/submission_roadmap.md](../src/submission/submission_roadmap.md) | The banned operations, the sandbox wrapper, the decoding loop, and local validation |
| [../src/huggingface/integration_plan.md](../src/huggingface/integration_plan.md) | Checkpoint conversion, the model card, and the custom inference handler |

## The one idea to take away

Everything here follows from a single observation. A transformer that is told
*where* a digit sits learns something that stops being true the moment an
operand gets longer. A model told what a digit is *worth* learns something
that stays true at any length.

Place value is not a trick for this task. It is the only description of a
digit that survives the thing the task does to inputs.

**[Back to the repository](../README.md)** &nbsp;·&nbsp;
**[On to the laboratory](../src/README.md)**
