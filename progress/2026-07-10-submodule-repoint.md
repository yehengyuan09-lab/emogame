# Submodule Repointing — Move 3rd-party submodules to owned forks

## Problem

Several git submodules were cloned from repos I do **not** own, so I cannot
push the parent `emogame` repo (submodule refs would point at commits living
only on someone else's remote).

Current (third-party) URLs in `.gitmodules`:

- `hero-skin-image` → `https://github.com/yansheng836/hero-skin-image.git`
- `weiboSpider`      → `https://github.com/dataabc/weiboSpider.git`
- `minimind`        → `https://github.com/jingyaogong/minimind.git` (leave as-is)

## Decision

- Repoint only the two I own to my forks:
  - `hero-skin-image` → `git@github.com:mzhyui/hero-skin-image.git`
  - `weiboSpider`     → `git@github.com:mzhyui/weiboSpider.git`
- **Leave `minimind` pointing at the upstream third-party repo** — it is not
  in my owned list and stays read-only.
- Parent repo pushes to `git@github.com:mzhyui/emogame.git`.

## Critical gotcha

The local submodule checkouts are at commits that are **not yet present** in
my forks:

- `hero-skin-image` local HEAD `461c4a9` (v1.0.5-7) — NOT in `mzhyui/hero-skin-image`
- `weiboSpider` local HEAD `00e1235` (heads/master) — NOT in `mzhyui/weiboSpider`

So the recorded submodule SHAs would dangle unless the commits are pushed to
the forks first. **Push before repointing/committing the parent.**

## Steps (to run when ready)

```bash
# 1. Push each submodule's current branch to MY fork.
git -C hero-skin-image remote add mine git@github.com:mzhyui/hero-skin-image.git
git -C hero-skin-image push mine HEAD:main        # fork default branch is main

git -C weiboSpider remote add mine git@github.com:mzhyui/weiboSpider.git
git -C weiboSpider push mine HEAD:master

# 2. Repoint submodule URLs in the parent repo.
git submodule set-url hero-skin-image git@github.com:mzhyui/hero-skin-image.git
git submodule set-url weiboSpider git@github.com:mzhyui/weiboSpider.git

# 3. Commit + push the parent.
git add .gitmodules
git commit -m "chore: repoint submodules to owned forks"
git push origin main        # origin = git@github.com:mzhyui/emogame.git
```

## Verification

- `git submodule status` shows both as clean (no `-`/ `+` prefix drift) after push.
- `git ls-remote git@github.com:mzhyui/<sub>` contains the local HEAD SHA.
- `git submodule sync --recursive` from a fresh clone resolves both forks.
