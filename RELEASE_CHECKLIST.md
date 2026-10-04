# Public Release Checklist

## Before pushing to GitHub

- [ ] Confirm the repository name and description.
- [x] Add `LICENSE` using Apache License 2.0.
- [x] Remove third-party photographs whose redistribution rights are not verified.
- [x] Keep `images/` as an input directory with a licensing notice.
- [x] Document third-party asset policy in `docs/third_party_assets.md`.
- [ ] Confirm that every image added later can legally be redistributed.
- [ ] Create a clean Python 3.10+ virtual environment.
- [ ] Run `pip install -r requirements.txt`.
- [ ] Run `pytest`.
- [ ] Run `cd src && python main.py` on at least one suitable, legally usable image.
- [ ] Inspect `results/results.csv`.
- [ ] Verify that no private images, credentials, virtual environments,
      caches, or generated files are committed.
- [ ] Review README claims and keep them consistent with measured validation.
