# Public release checklist

Before making the repository public:

- [ ] Replace `<OWNER>/<REPO>` badges in `README.md`.
- [ ] Confirm the license with all co-authors.
- [ ] Update `CITATION.cff` with the final repository URL and DOI.
- [ ] Decide whether checkpoints will be hosted externally.
- [ ] Add checkpoint URLs and SHA256 checksums to `docs/CHECKPOINTS.md`.
- [ ] Confirm whether the Poisson deblurring SwinIR/P4IP weights can be shared.
- [ ] Run `make test` on a clean machine.
- [ ] Run `make reference-figures` and inspect the output figures.
- [ ] Ensure no private paths, W&B logs, `.git` folders, HDF5 datasets, or checkpoints are committed.
- [ ] Create a GitHub release and archive it on Zenodo if desired.

- [ ] Confirm that `paper_assets/` contains only figures that can be publicly shared.
- [ ] Replace placeholder GitHub owner/repository badge URLs in `README.md`.
