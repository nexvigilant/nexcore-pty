# nexcore-pty

Part of the [NexVigilant](https://nexvigilant.com) pharmacovigilance platform.

## About NexVigilant

NexVigilant makes pharmacovigilance accessible. We build open computation tools for drug safety signal detection, causality assessment, and regulatory intelligence — because patient safety knowledge should be available to everyone willing to learn.

**Live tools:** [mcp.nexvigilant.com](https://mcp.nexvigilant.com) — 193 MCP tools for AI-powered pharmacovigilance, free to connect.

## Installation

```toml
[dependencies]
nexcore-pty = { git = "https://github.com/nexvigilant/nexcore-pty" }
```

> **Note:** This crate was developed as part of the [nexcore](https://github.com/nexvigilant) workspace. Some dependencies may reference workspace-level configuration. See individual `Cargo.toml` for details.

## License

**Personal, non-commercial use only.** See [LICENSE](LICENSE) for full terms.

Organizations of any kind must have explicit written permission for use.
Contact [matthew@nexvigilant.com](mailto:matthew@nexvigilant.com) for licensing.

## Contributing

Contributions are welcome under the following terms:

1. **Fork & PR.** Fork this repository, make your changes, and submit a pull request.
2. **CLA.** By submitting a pull request, you agree that your contributions become the property of NexVigilant LLC under the same license terms.
3. **Code quality.** All Rust code must pass `cargo clippy -- -D warnings` and `cargo fmt --check`.
4. **Tests.** New functionality should include tests. Run `cargo test --lib` before submitting.

For questions or discussion, open an issue or reach out at [matthew@nexvigilant.com](mailto:matthew@nexvigilant.com).

---

Built by [NexVigilant LLC](https://nexvigilant.com) — Pharmacovigilance for NexVigilants.
