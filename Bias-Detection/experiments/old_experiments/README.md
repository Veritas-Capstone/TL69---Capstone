Initial experiments to improve on the volf baseline model. These experiments tested different approaches to explainability, article-level prediction, and OOD evaluation using the baseline Volf model before we looked into building our own training pipeline.

### Some findings from these experiments

- Using attention to get explainability gave us poor results and not meaningful words, Integrated Gradients was too slow to use.
- Volf's baseline generalizes pretty poorly, got 47% accuracy on allsides articles.
