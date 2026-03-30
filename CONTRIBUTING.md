# Contributing to PSE vcipher Collapse

Thanks for contributing to `pse-vcipher-collapse`.

This repository is a focused prototype for POWER8-oriented LLM inference work, so the best contributions are usually small, reviewable changes to headers, benchmark code, or documentation.

## Good Contribution Targets

- Fix correctness issues in the header implementations
- Improve benchmark clarity or reproducibility in `bench_vcipher_collapse.c`
- Tighten POWER8-specific build or compatibility notes in `README.md`
- Add small documentation updates that explain the PSE / vcipher / vec_perm workflow more clearly

## Before You Open a PR

1. Read [README.md](README.md) to understand the current scope and build assumptions
2. Keep changes narrow and easy to review
3. If you change behavior, update the relevant benchmark or documentation alongside the code change
4. Avoid large unrelated refactors

## Development Notes

- This repository is centered on C and header-only integration snippets
- POWER8-specific intrinsics and compile flags matter here; preserve the existing style unless you are intentionally improving it
- If you add or change macros, document the intended usage in `README.md`

## Suggested Validation

Use the lightest checks that fit your change:

```bash
git diff --check
cc -fsyntax-only -mcpu=power8 -mvsx -maltivec -mcrypto bench_vcipher_collapse.c
```

If your environment does not provide a POWER8-capable toolchain, say so clearly in the PR and describe what you were still able to verify.

## Pull Request Checklist

- Fork the repository and create a topic branch
- Explain what changed and why
- Include the exact validation commands you ran
- Link any related issue or bounty in the PR body
- Keep the diff limited to the files needed for the contribution

## Documentation Expectations

When a change affects one of these areas, update the matching file in the same PR:

- build assumptions or compile flags -> `README.md`
- benchmark methodology -> `bench_vcipher_collapse.c`
- integration surface or macro behavior -> relevant `ggml-*.h` header

## License

By contributing, you agree that your contribution will be released under the [MIT License](LICENSE).
