// Command pinn is the single-binary host and launcher for the PINN learning
// path: it serves the embedded hub (the same dist/index.html that GitHub
// Pages publishes), extracts the starter kit and the marimo labs into a local
// work folder, and runs both through uv - which it bootstraps automatically
// when it is not already installed.
//
// The binary embeds everything this repository authors. The one thing it
// cannot embed is PyTorch: on the first `pinn run` or `pinn lab`, uv resolves
// the CPU-only wheels once (the labs' PEP 723 headers and the starter kit's
// pyproject pin the CPU index) and caches them; after that, everything is
// offline.
package main

import (
	"fmt"
	"os"
)

var version = "dev" // stamped by -ldflags "-X main.version=..."

func usage() {
	fmt.Fprintf(os.Stderr, `pinn %s - the PINN learning path in one binary

Usage:
  pinn serve [-addr host:port] [-no-open]     serve the hub locally and open it
  pinn init  [-dir DIR]                       extract starter kit + labs to DIR
  pinn run   [-dir DIR] <script> [-- args]    run a starter script via uv
  pinn lab   [-dir DIR] [-dry-run] <name>     open a marimo lab (sandboxed)
  pinn version                                print the version

Examples:
  pinn serve                # read the learning path at http://127.0.0.1:8000
  pinn init                 # materialise ./pinn-work
  pinn run 00               # environment check (proves the float64 story)
  pinn run 06 -- --plain    # Burgers baseline; later: -- --all
  pinn lab 02               # autodiff lab in your browser (CPU torch auto-installed)

The work folder defaults to ./pinn-work (override with -dir or PINN_WORKDIR).
uv is taken from PATH or ~/.pinn/bin, or downloaded there on first need.
`, version)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "serve":
		err = cmdServe(os.Args[2:])
	case "init":
		err = cmdInit(os.Args[2:])
	case "run":
		err = cmdRun(os.Args[2:])
	case "lab":
		err = cmdLab(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Println("pinn", version)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "pinn: unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "pinn:", err)
		os.Exit(1)
	}
}
