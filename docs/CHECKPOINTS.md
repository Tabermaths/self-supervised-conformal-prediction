# Checkpoint policy

Large checkpoints are excluded from the repository.

Recommended local layout:

```text
outputs/<experiment>/checkpoints/<method>/ckp_*.pth.tar
```

Examples:

```text
outputs/gaussian_denoising_div2k/checkpoints/sure/ckp_099.pth.tar
outputs/poisson_denoising_div2k/checkpoints/pure/ckp_399.pth.tar
```

## Public release recommendation

For a public release, host final checkpoints externally, for example on an institutional server, Zenodo, Hugging Face, or Google Drive, and update this file with:

- checkpoint name;
- experiment config;
- URL;
- SHA256 checksum;
- expected command to reproduce the corresponding figure.

## Missing Poisson deblurring checkpoint

The Poisson deblurring experiment used a specialised SwinIR/P4IP-style estimator. The uploaded material did not include a compact public checkpoint, so the current builder raises `NotImplementedError` for this experiment. This is intentional: silently substituting a different model would be misleading.
