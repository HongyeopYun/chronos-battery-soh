# Chronos battery workspace

- Keep all Chronos-related generated files inside this repository. Do not create build, cache, transfer, or temporary output folders in `Desktop/Lab`, a ChatGPT project mirror, or the user's Desktop.
- Use `artifacts/presentation-builds/<work-name>/` for slide build sources, intermediate files, renders and draft outputs; use `presentations/published/` for delivered slide decks.
- Use `artifacts/remote-transfer/<date>/` for SSH/tunnel packages and downloaded run bundles, and `artifacts/exports/` for exported result archives.
- Use `artifacts/maintenance/<date>-<task>/` for filesystem inventories, path maps, backups and verification reports. Never discard a file merely because it is untracked or ignored by Git.
- Keep `src/`, `scripts/`, `configs/`, `tests/`, `data/`, `models/` and completed `results/` at their established paths. Saved experiment protocols reference paths and source hashes. Preserve immutable result records; write new experiments to a new run directory.
- Start with `docs/workspace_layout.md` and the script/result indexes when locating an experiment. Record new runnable entry points and result reports there.
- Preserve shared raw datasets and unrelated models outside the repository. Synced ChatGPT `sources/` are read-only references and must never be moved or edited.
- Some raw datasets and archived weights are hardlinked to identical retained files (see `artifacts/maintenance/20261015-storage-cleanup/plan.json`). Do not modify shared file contents in place. If an authorized change is required, write a new file and replace the intended path so other aliases retain their original content.
- Ask the user before changing a model's internal architecture or adding/changing a task head or its feature dimensions. Training settings and filesystem cleanup do not authorize architectural changes.
