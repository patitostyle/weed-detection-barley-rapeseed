# Source checkpoint

Exact retained barley initialization: `cebada_best_model.pth`.

SHA256: `728e1ffe2e6647a251aca57f8cc7fdc416db3ad14a384d52efae294b86e9e1e4`

This file is supplied separately in `release_assets/` for upload as a GitHub Release asset. Download it to `revision_controlada/models/cebada_best_model.pth`, or set `WEED_BASE_MODEL`. Size: 30,881,832 bytes. The source checkpoint was selected using Barley 2 validation. It is not a target-domain checkpoint. Use `torch.load(..., weights_only=True)` as in the scripts.

The 18 target checkpoints are retained locally but are not included in the repository package. They can be regenerated with `--train`; the archived histories and confusion matrices are included. Re-evaluating already trained target models requires their checkpoints; histories alone do not contain their weights.
