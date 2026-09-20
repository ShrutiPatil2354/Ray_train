# Limitations

- Only one physical NVIDIA GPU was available.
- Actual multi-GPU and multi-node Ray execution was not performed.
- Gloo was used on Windows; NCCL was not assumed.
- The Iris dataset is small and intended for an academic demonstration.
- No distributed speedup benchmark is claimed.
- No CI/CD, serving, monitoring, governance, or external experiment tracker was selected or implemented for this focused CCA workflow.
- DVC has local database tracking metadata but no remote storage.
- Checkpoint state loading and inference were tested; resume-training was not.