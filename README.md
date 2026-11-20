# Fundamentals of Natural Language Processing — Assignment 4

## Contents

| Path | Contents |
| --- | --- |
| `part1/` | BERT sentiment classification, text transformations, and data augmentation |
| `part2/` | T5 fine-tuning, from-scratch training, inference, and SQL evaluation |
| `scripts/` | Download and verify the course data from a pinned starter commit |
| `tests/` | Scratch-training continuation and export checks |
| [docs/RUNNING.md](docs/RUNNING.md) | Environment setup and execution commands |
| [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) | Checkpoint restoration instructions |
| [Course starter](https://github.com/gregdurrett/nyu-dsga1011-a4/tree/e67bb11) | Original starter code and supplied data |

## Recorded results

| Part 1 model | Original accuracy (%) | Transformed accuracy (%) |
| --- | ---: | ---: |
| Original training data | 92.412 | 87.196 |
| Original + 5,000 transformed training reviews | 92.964 | 89.900 |

| Part 2 model | Selected epoch | Dev Record F1 (%) | Test Record F1 (%) |
| --- | ---: | ---: | ---: |
| Fine-tuned T5-small | 18 | 85.58 | 79.26 |
| T5-small initialized from scratch | 35 | 61.08 | 55.75 |

Fine-tuned dev F1 includes timeout rechecks with unchanged SQL. Test scores are
reported by Gradescope.
