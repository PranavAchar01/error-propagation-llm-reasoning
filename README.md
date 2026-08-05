# Error Propagation in Multi-Step LLM Reasoning

Does forcing a language model to commit to explicit, machine-checkable intermediate
steps improve multi-step logical reasoning accuracy — and does it reduce *error
propagation*, the tendency for one wrong intermediate step to corrupt every step
after it?

**Status: apparatus complete, experiments not yet run.** See
[HYPOTHESIS.md](HYPOTHESIS.md) for the pre-registered primary metric, and
[paper/report.md](paper/report.md) for the write-up. Headline numbers will be
filled in here once the pilot and full runs execute — this file will state the
effect size, n, and 95% CI, or state plainly that no effect was detected.

This is independent work. It is not affiliated with, endorsed by, or conducted
under any research lab.

## Quick start

```bash
make setup && make data && make test
```

## License

MIT for code. Benchmark datasets retain their original licenses; see
[data/README.md](data/README.md).
