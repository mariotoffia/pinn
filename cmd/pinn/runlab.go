package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

func cmdRun(args []string) error {
	fl := flag.NewFlagSet("run", flag.ExitOnError)
	dir := workdirFlag(fl)
	dry := fl.Bool("dry-run", false, "print the command instead of running it")
	if err := fl.Parse(args); err != nil {
		return err
	}
	if fl.NArg() < 1 {
		return fmt.Errorf("usage: pinn run [-dir DIR] <script> [-- script args]")
	}
	if err := ensureWorkdir(*dir); err != nil {
		return err
	}
	name, err := resolveOne(filepath.Join(*dir, "starter", "scripts"), fl.Arg(0), "starter script")
	if err != nil {
		return err
	}
	uv, err := ensureUV()
	if err != nil {
		return err
	}
	extra := fl.Args()[1:]
	if len(extra) > 0 && extra[0] == "--" {
		extra = extra[1:]
	}
	// uv run inside starter/ resolves the kit's pyproject (CPU torch pinned
	// for Linux/Windows via [tool.uv.sources]) into a cached project env.
	cargs := append([]string{"run", filepath.Join("scripts", name)}, extra...)
	return execTool(uv, cargs, filepath.Join(*dir, "starter"), *dry)
}

func cmdLab(args []string) error {
	fl := flag.NewFlagSet("lab", flag.ExitOnError)
	dir := workdirFlag(fl)
	dry := fl.Bool("dry-run", false, "print the command instead of running it")
	if err := fl.Parse(args); err != nil {
		return err
	}
	if fl.NArg() != 1 {
		return fmt.Errorf("usage: pinn lab [-dir DIR] <02|03|04|name>")
	}
	if err := ensureWorkdir(*dir); err != nil {
		return err
	}
	name, err := resolveOne(filepath.Join(*dir, "notebooks"), fl.Arg(0), "lab")
	if err != nil {
		return err
	}
	uv, err := ensureUV()
	if err != nil {
		return err
	}
	if !*dry {
		fmt.Printf("pinn: opening %s - the first run builds its sandbox (PyTorch download, once; then cached)\n", name)
	}
	// The lab's PEP 723 header names its dependencies; marimo --sandbox has uv
	// build the environment from it (uvx == `uv tool run`). On Linux/Windows,
	// point that resolution at PyTorch's CPU-only wheel index: PyPI's default
	// Linux torch bundles CUDA (a multi-GB download). unsafe-best-match keeps
	// every other package coming from PyPI, and macOS needs neither.
	var env []string
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		env = append(env,
			"UV_INDEX=https://download.pytorch.org/whl/cpu",
			"UV_INDEX_STRATEGY=unsafe-best-match")
	}
	cargs := []string{"tool", "run", "marimo", "edit", "--sandbox", filepath.Join("notebooks", name)}
	return execTool(uv, cargs, *dir, *dry, env...)
}

// execTool runs bin with args in dir, terminal wired through, and with bin's
// folder prepended to PATH (marimo --sandbox shells back out to uv, which
// must be findable when we bootstrapped it into ~/.pinn/bin).
func execTool(bin string, args []string, dir string, dry bool, extraEnv ...string) error {
	if dry {
		fmt.Printf("cd %s && %s %s\n", dir, bin, strings.Join(args, " "))
		return nil
	}
	c := exec.Command(bin, args...)
	c.Dir = dir
	c.Stdin, c.Stdout, c.Stderr = os.Stdin, os.Stdout, os.Stderr
	c.Env = append(os.Environ(), "PATH="+filepath.Dir(bin)+string(os.PathListSeparator)+os.Getenv("PATH"))
	c.Env = append(c.Env, extraEnv...)
	return c.Run()
}
