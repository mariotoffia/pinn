package main

import (
	"embed"
	"fmt"
)

// The payload staged by `make pinn-assets`: the built hub (index.html), the
// starter kit and the marimo labs. In git, assets/ holds only .gitkeep; the
// real files are copied in at build time so the repository never commits
// build artifacts. A binary built without staging fails with a clear message
// instead of shipping an empty shell.
//
//go:embed all:assets
var assets embed.FS

const notStaged = "this binary was built without its payload - build it with `make pinn` (which stages cmd/pinn/assets first), not with plain `go build`"

// hubPage returns the embedded single-file hub.
func hubPage() ([]byte, error) {
	b, err := assets.ReadFile("assets/index.html")
	if err != nil || len(b) == 0 {
		return nil, fmt.Errorf("%s", notStaged)
	}
	return b, nil
}
