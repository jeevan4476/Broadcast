# Agent Instructions

## Package manager and JS runtime

Use **Bun** everywhere in this repository. Never use `npm`, `npx`, `yarn`, or `pnpm`.

- Install UI dependencies with `cd ui && bun install` (never `npm install`).
- Run scripts with `bun run <script>` (never `npm run`).
- Run package binaries with `bunx <bin>` (never `npx`).
- Keep only `bun.lock` if a lockfile is generated. Never create or commit `package-lock.json`, `yarn.lock`, or `pnpm-lock.yaml`.
