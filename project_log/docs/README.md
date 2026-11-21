# Dual-form data attribution

This project examines if the dual form of a neural network gives better data attribution than methods on the trained weights. Under plain SGD, each linear layer is exactly its initial weights plus one rank-1 write per training example and step. The project stores these writes and uses them to find the training examples that cause a prediction. The current work uses an MLP on MNIST. Mentor: Omri Uzan. Collaborator: Amir Zur.

## Docs
- [motivation.md](motivation.md) — the research question and its reason.
- [codebase.md](codebase.md) — the scripts, the outputs, and the commands.
- [experiments/](experiments/) — one file per experiment.
- [findings.md](findings.md) — the current conclusions and their evidence.
- [open_questions.md](open_questions.md) — risks and next-step options.
- [../entries/](../entries/) — the history, in order.

## Terms
- **Write:** the change that one training example makes to one layer at one SGD step: `-lr * e k^T`.
- **Key (k):** the input of the layer for that training example.
- **Error (e):** the gradient of the loss with respect to the layer output. The value is `v = -e`.
- **Query (q):** the input of the layer for a test example, from the final model.
- **Dual form:** `W_final = W_init - sum over writes of lr * e k^T`.
- **Attention:** `k^T q`, the similarity of a key and a query (Irie et al. 2022).
- **Attention × error:** `lr * (k^T q) * (e^T e_q)`. A positive value means that the write lowers the test loss.
- **TracIn:** the sum over steps of `lr * (gradient of the test loss) · (gradient of the training loss)`, both at the model of that step.
- **Margin:** the logit of the true class minus the highest other logit.
- **Proponent / opponent:** a training example that lowers / increases the test loss.
