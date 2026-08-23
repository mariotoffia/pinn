package main

import (
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const defaultWorkdir = "pinn-work"

func workdirFlag(fl *flag.FlagSet) *string {
	def := os.Getenv("PINN_WORKDIR")
	if def == "" {
		def = defaultWorkdir
	}
	return fl.String("dir", def, "work folder (also settable via PINN_WORKDIR)")
}

const workdirReadme = `# PINN learning path - work folder

Created by "pinn init". Contents:

- starter/    the runnable Python kit (scripts 00..07) - see starter/README.md
- notebooks/  the interactive marimo labs           - see notebooks/README.md

Run things through the pinn binary from the parent folder (or pass -dir):

    pinn run 00           # environment check
    pinn run 06 -- --plain
    pinn lab 02           # opens in your browser; packages auto-install (CPU torch)

Everything also works without pinn, straight from this folder:

    cd starter && uv run scripts/00_check_environment.py
    uvx marimo edit --sandbox notebooks/02_autodiff_lab.py

The full learning path (the hub): pinn serve
Repository: https://github.com/mariotoffia/pinn
`

// extractWorkdir writes the embedded starter/ and notebooks/ into dir.
// Existing files are never overwritten - a reader's edits are theirs.
func extractWorkdir(dir string) (written, kept int, err error) {
	for _, top := range []string{"starter", "notebooks"} {
		sub, serr := fs.Sub(assets, "assets/"+top)
		if serr != nil {
			return written, kept, serr
		}
		werr := fs.WalkDir(sub, ".", func(p string, d fs.DirEntry, e error) error {
			if e != nil {
				return e
			}
			dst := filepath.Join(dir, top, filepath.FromSlash(p))
			if d.IsDir() {
				return os.MkdirAll(dst, 0o755)
			}
			if _, statErr := os.Stat(dst); statErr == nil {
				kept++
				return nil
			}
			b, rerr := fs.ReadFile(sub, p)
			if rerr != nil {
				return rerr
			}
			if wferr := os.WriteFile(dst, b, 0o644); wferr != nil {
				return wferr
			}
			written++
			return nil
		})
		if werr != nil {
			if errors.Is(werr, fs.ErrNotExist) {
				return written, kept, fmt.Errorf("%s", notStaged)
			}
			return written, kept, werr
		}
	}
	if written+kept == 0 {
		return 0, 0, fmt.Errorf("%s", notStaged)
	}
	rd := filepath.Join(dir, "README.md")
	if _, statErr := os.Stat(rd); errors.Is(statErr, os.ErrNotExist) {
		if wferr := os.WriteFile(rd, []byte(workdirReadme), 0o644); wferr == nil {
			written++
		}
	}
	return written, kept, nil
}

// ensureWorkdir extracts on demand and stays quiet when nothing was missing.
func ensureWorkdir(dir string) error {
	w, k, err := extractWorkdir(dir)
	if err != nil {
		return err
	}
	if w > 0 {
		fmt.Printf("pinn: prepared %s (%d files written, %d existing kept)\n", dir, w, k)
	}
	return nil
}

func cmdInit(args []string) error {
	fl := flag.NewFlagSet("init", flag.ExitOnError)
	dir := workdirFlag(fl)
	if err := fl.Parse(args); err != nil {
		return err
	}
	if fl.NArg() > 0 { // also allow: pinn init some/dir
		*dir = fl.Arg(0)
	}
	w, k, err := extractWorkdir(*dir)
	if err != nil {
		return err
	}
	fmt.Printf("pinn: extracted into %s (%d written, %d already there - kept)\n", *dir, w, k)
	fmt.Println("next:  pinn run 00   |   pinn lab 02   |   pinn serve")
	return nil
}

// resolveOne finds exactly one .py file in dir whose name matches arg by
// prefix or substring ("02" and "autodiff" both find 02_autodiff_lab.py).
func resolveOne(dir, arg, kind string) (string, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var names []string
	for _, e := range ents {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".py") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	var hits []string
	for _, n := range names {
		if strings.HasPrefix(n, arg) || strings.Contains(n, arg) {
			hits = append(hits, n)
		}
	}
	switch len(hits) {
	case 1:
		return hits[0], nil
	case 0:
		return "", fmt.Errorf("no %s matches %q; available: %s", kind, arg, strings.Join(names, ", "))
	default:
		return "", fmt.Errorf("%q is ambiguous (%s); available: %s", arg, strings.Join(hits, ", "), strings.Join(names, ", "))
	}
}
