"""
This file contains the full end-to-end pipeline of the project:
- Create and train a Fully Connected Network (FCN)
- Apply magnitude-based weight pruning
- Convert pruning masks into permanent zero weights
- Compress the pruned model
- Decompress layers on-the-fly and run inference
- Report accuracy, sparsity, and memory-related results

The file only orchestrates the workflow.
All core logic is implemented in separate modules.
"""

