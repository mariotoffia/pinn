# PINN Learning Path
#
#   make generate   -> build dist/index.html from content/*.md + src/*.ts + src/*.js
#
# The build has ZERO dependencies: plain Node, no npm install, no network. esbuild is used
# automatically if it happens to be resolvable, otherwise the bundled type stripper runs and
# the result is validated with `node --check`, so a broken build fails loudly.

SHELL      := /bin/bash
NODE       ?= node
PY         ?= python3
OUT        ?= dist/index.html
PORT       ?= 8000
CONTENT    := $(wildcard content/*.md)
SOURCES    := $(wildcard src/*.ts) $(wildcard src/*.js) $(wildcard src/*.css)
TOOLS      := $(wildcard tools/*.mjs)

.DEFAULT_GOAL := help
.PHONY: help generate watch serve open check ci typecheck dev setup run clean stats links

## generate : build the single-file hub into dist/ from all markdown + TS/JS
generate: $(OUT)

$(OUT): $(CONTENT) $(SOURCES) $(TOOLS)
	@command -v $(NODE) >/dev/null || { echo "error: node not found. Install Node 18+ and retry."; exit 1; }
	@echo "building $(OUT)"
	@$(NODE) tools/build.mjs --out $(OUT)
	@echo "open it with:  make open"

## watch    : rebuild whenever a chapter or source file changes (Ctrl-C to stop)
watch:
	@echo "watching content/ src/ tools/ - Ctrl-C to stop"
	@last=""; while true; do \
	  now=$$(ls -l $(CONTENT) $(SOURCES) $(TOOLS) 2>/dev/null | md5sum 2>/dev/null || ls -l $(CONTENT) $(SOURCES) $(TOOLS) | md5); \
	  if [ "$$now" != "$$last" ]; then $(MAKE) --no-print-directory generate; last="$$now"; fi; \
	  sleep 1; \
	done

## serve    : build, then serve on http://localhost:$(PORT) (nicer than file:// for testing)
serve: generate
	@echo "serving http://localhost:$(PORT)/  (from $(dir $(OUT)))"
	@cd $(dir $(OUT)) && $(PY) -m http.server $(PORT)

## open     : build and open the hub in your default browser
open: generate
	@(command -v open >/dev/null && open $(OUT)) \
	  || (command -v xdg-open >/dev/null && xdg-open $(OUT)) \
	  || echo "open $(PWD)/$(OUT) in a browser"

## check    : syntax-check the build tools, build once, and verify the result is usable
check:
	@for f in $(TOOLS); do $(NODE) --check $$f && echo "ok  $$f"; done
	@$(NODE) tools/build.mjs --out /tmp/pinn-check.html --quiet && echo "ok  build produces valid output"
	@$(NODE) tools/verify.mjs /tmp/pinn-check.html
	@rm -f /tmp/pinn-check.html

## ci       : what CI runs - syntax-check the tools, then prove the build is reproducible
ci: check
	@echo "checking the build is byte-reproducible"
	@$(NODE) tools/build.mjs --out /tmp/pinn-ci-a.html --date 2026-01-01 --quiet
	@$(NODE) tools/build.mjs --out /tmp/pinn-ci-b.html --date 2026-01-01 --quiet
	@cmp -s /tmp/pinn-ci-a.html /tmp/pinn-ci-b.html \
	  && echo "ok  two builds are identical" \
	  || { echo "error: build is not reproducible"; exit 1; }
	@rm -f /tmp/pinn-ci-a.html /tmp/pinn-ci-b.html

## typecheck: run the TypeScript compiler over src/ (needs npx + typescript; optional)
typecheck:
	@npx --no-install tsc --noEmit --strict --target es2020 --moduleResolution bundler \
	     --module esnext --lib es2020,dom src/*.ts \
	  || echo "(skipped: install with 'npm i -D typescript' to type-check src/)"

## stats    : word, link and chapter counts
stats:
	@echo "chapters : $(words $(CONTENT))"
	@echo "words    : $$(cat $(CONTENT) | wc -w)"
	@echo "links    : $$(grep -oh 'https\?://[^ )>]*' $(CONTENT) | wc -l)"
	@echo "unique   : $$(grep -oh 'https\?://[^ )>]*' $(CONTENT) | sed 's/[.,;:]$$//' | sort -u | wc -l)"

## links    : print every unique external URL (pipe to a link checker if you like)
links:
	@grep -oh 'https\?://[^ )>]*' $(CONTENT) | sed 's/[.,;:]$$//' | sort -u

## dev      : create .venv at the repo root (starter kit + marimo + ruff) for your editor
dev:
	@if command -v uv >/dev/null 2>&1; then \
	  echo "using uv"; \
	  uv venv .venv && uv pip install --python .venv -e ./starter marimo ruff; \
	else \
	  echo "uv not found - falling back to venv + pip"; \
	  $(PY) -m venv .venv && .venv/bin/python -m pip install -q -U pip \
	    && .venv/bin/python -m pip install -q -e ./starter marimo ruff; \
	fi
	@echo ""
	@echo "  .venv is ready."
	@echo "  VS Code: reload the window and it will pick it up (Python: Select Interpreter -> .venv)."
	@echo "  shell  : source .venv/bin/activate"

## setup    : create the starter-kit virtualenv (uv if present, venv otherwise)
setup:
	@cd starter && ( command -v uv >/dev/null \
	  && uv venv && uv pip install -e . \
	  || ( $(PY) -m venv .venv && .venv/bin/pip install -q -e . ) )
	@echo "starter kit ready. next:  make run"

## run      : run the starter-kit environment check (then work through scripts/ in order)
run:
	@cd starter && ( [ -x .venv/bin/python ] && .venv/bin/python scripts/00_check_environment.py \
	  || $(PY) scripts/00_check_environment.py )

## clean    : remove the generated hub
clean:
	@rm -rf $(dir $(OUT))
	@echo "removed $(dir $(OUT))"

help:
	@echo ""
	@echo "  PINN Learning Path"
	@echo ""
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  make /'
	@echo ""
	@echo "  $(words $(CONTENT)) chapters in content/, $(words $(SOURCES)) source files in src/"
	@echo ""
