package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestUVTarget(t *testing.T) {
	cases := []struct{ goos, goarch, triple, ext string }{
		{"darwin", "arm64", "aarch64-apple-darwin", "tar.gz"},
		{"darwin", "amd64", "x86_64-apple-darwin", "tar.gz"},
		{"linux", "amd64", "x86_64-unknown-linux-gnu", "tar.gz"},
		{"linux", "arm64", "aarch64-unknown-linux-gnu", "tar.gz"},
		{"windows", "amd64", "x86_64-pc-windows-msvc", "zip"},
		{"windows", "arm64", "aarch64-pc-windows-msvc", "zip"},
	}
	for _, c := range cases {
		triple, ext, err := uvTarget(c.goos, c.goarch)
		if err != nil || triple != c.triple || ext != c.ext {
			t.Errorf("uvTarget(%s,%s) = %q,%q,%v; want %q,%q", c.goos, c.goarch, triple, ext, err, c.triple, c.ext)
		}
	}
	if _, _, err := uvTarget("plan9", "amd64"); err == nil {
		t.Error("expected an error for an unsupported OS")
	}
	if _, _, err := uvTarget("linux", "riscv64"); err == nil {
		t.Error("expected an error for an unsupported arch")
	}
}

func TestHubHandler(t *testing.T) {
	page, err := hubPage()
	if err != nil {
		t.Skip("assets not staged (run `make pinn-assets`):", err)
	}
	low := strings.ToLower(string(page[:200]))
	if !strings.Contains(low, "<!doctype html") {
		t.Fatal("embedded page does not start like the hub")
	}
	srv := httptest.NewServer(hubHandler(page))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(body), "Physics-Informed") {
		t.Fatalf("GET / = %d, page recognisable: %v", resp.StatusCode, strings.Contains(string(body), "Physics-Informed"))
	}

	resp, err = http.Get(srv.URL + "/definitely-not-here")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Fatalf("GET /definitely-not-here = %d, want 404", resp.StatusCode)
	}
}

func TestExtractAndResolve(t *testing.T) {
	dir := t.TempDir()
	w, _, err := extractWorkdir(dir)
	if err != nil {
		t.Skip("assets not staged (run `make pinn-assets`):", err)
	}
	if w == 0 {
		t.Fatal("nothing extracted into an empty dir")
	}
	for _, p := range []string{
		filepath.Join(dir, "starter", "pyproject.toml"),
		filepath.Join(dir, "notebooks", "02_autodiff_lab.py"),
		filepath.Join(dir, "README.md"),
	} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("missing after extract: %s", p)
		}
	}

	name, err := resolveOne(filepath.Join(dir, "notebooks"), "02", "lab")
	if err != nil || name != "02_autodiff_lab.py" {
		t.Fatalf("resolveOne(02) = %q, %v", name, err)
	}
	name, err = resolveOne(filepath.Join(dir, "notebooks"), "oscillator", "lab")
	if err != nil || name != "04_oscillator_pinn_lab.py" {
		t.Fatalf("resolveOne(oscillator) = %q, %v", name, err)
	}
	if _, err := resolveOne(filepath.Join(dir, "notebooks"), "0", "lab"); err == nil {
		t.Fatal("expected an ambiguity error for \"0\"")
	}
	if _, err := resolveOne(filepath.Join(dir, "notebooks"), "zzz", "lab"); err == nil {
		t.Fatal("expected a no-match error for \"zzz\"")
	}

	// A second extract must keep every existing file (never overwrite).
	marker := filepath.Join(dir, "notebooks", "02_autodiff_lab.py")
	if err := os.WriteFile(marker, []byte("USER EDIT"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, k, err := extractWorkdir(dir)
	if err != nil || k == 0 {
		t.Fatalf("re-extract: kept=%d err=%v", k, err)
	}
	b, _ := os.ReadFile(marker)
	if string(b) != "USER EDIT" {
		t.Fatal("re-extract overwrote a user-edited file")
	}
}
