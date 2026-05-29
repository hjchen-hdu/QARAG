# Data Files

Only lightweight dataset identifier files were migrated from `SubgraphRAG`.

The following generated artifacts were intentionally not copied:

- `processed/*.pkl`
- `emb/*/*.pth`
- `triple_scores/*.pth`
- `gpt_triples.pth`

To run the baseline unchanged, regenerate or copy the required artifacts into
this directory. For QARAG development, keep generated artifacts out of source
control and treat this folder as runtime data.

