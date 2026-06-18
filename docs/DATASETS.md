# Dataset policy

## Do not commit datasets

Do not commit DIV2K images, generated HDF5 files, downloaded DeepInv datasets, or cached Hugging Face files. These files are intentionally ignored by `.gitignore`.

## Expected layout

The paper scripts expect generated measurement datasets under the configured experiment root, for example:

```text
outputs/gaussian_denoising_div2k/dinv_dataset0.h5
outputs/poisson_denoising_div2k/dinv_dataset_gain_0.25.h5
```

The exact HDF5 naming follows the legacy DeepInv code. If a dataset is not found, the builders raise a clear `FileNotFoundError` instead of silently downloading or regenerating data.

## Public release recommendation

For a public GitHub release:

1. commit only code and lightweight reference arrays;
2. provide dataset-generation commands;
3. publish checksums for generated HDF5 files if they are needed for exact reproduction;
4. keep raw DIV2K licensing separate and direct users to the official source.

## Small built-in datasets

The package contains synthetic toy datasets in `sscp.data.synthetic`. They are only for installation tests and API demonstrations; they are not paper benchmarks.
