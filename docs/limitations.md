# Limitations

- Only one physical NVIDIA GPU was available, so actual multi-GPU and multi-node Ray execution was not performed.
- Gloo was used on Windows; NCCL was not assumed.
- The UCI Bike Sharing day-level dataset is appropriate for coursework but is not an enterprise-scale benchmark.
- No distributed speedup is claimed.
- `casual` and `registered` are deliberately excluded to avoid target leakage.
- No CI/CD, serving API, monitoring stack, or external experiment tracker was selected for this focused CCA workflow.
- DVC locally tracks database metadata; no DVC remote storage is configured.
- Checkpoint loading and test inference were executed; resume-training was not.