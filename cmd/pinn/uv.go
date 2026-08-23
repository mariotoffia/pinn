package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

const maxArchive = 200 << 20 // generous cap for uv's ~35 MB archives

// ensureUV returns a usable uv binary: from PATH, from ~/.pinn/bin, or
// downloaded there from uv's official release archives.
func ensureUV() (string, error) {
	if p, err := exec.LookPath("uv"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	bin := filepath.Join(home, ".pinn", "bin", uvExe())
	if _, err := os.Stat(bin); err == nil {
		return bin, nil
	}
	fmt.Println("pinn: uv not found - fetching it once from astral-sh/uv releases...")
	if err := downloadUV(bin); err != nil {
		return "", fmt.Errorf("could not bootstrap uv: %w\ninstall it manually instead: https://docs.astral.sh/uv/getting-started/installation/", err)
	}
	fmt.Println("pinn: uv installed at", bin)
	return bin, nil
}

func uvExe() string {
	if runtime.GOOS == "windows" {
		return "uv.exe"
	}
	return "uv"
}

// uvTarget maps GOOS/GOARCH to uv's release target triple and archive type.
func uvTarget(goos, goarch string) (triple, ext string, err error) {
	arch := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[goarch]
	if arch == "" {
		return "", "", fmt.Errorf("no uv build for architecture %q", goarch)
	}
	switch goos {
	case "darwin":
		return arch + "-apple-darwin", "tar.gz", nil
	case "linux":
		return arch + "-unknown-linux-gnu", "tar.gz", nil
	case "windows":
		return arch + "-pc-windows-msvc", "zip", nil
	}
	return "", "", fmt.Errorf("no uv build for OS %q", goos)
}

func downloadUV(dst string) error {
	triple, ext, err := uvTarget(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return err
	}
	url := fmt.Sprintf("https://github.com/astral-sh/uv/releases/latest/download/uv-%s.%s", triple, ext)
	cl := &http.Client{Timeout: 15 * time.Minute}
	resp, err := cl.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxArchive))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if ext == "zip" {
		return extractOneZip(body, uvExe(), dst)
	}
	return extractOneTarGz(body, uvExe(), dst)
}

func extractOneTarGz(archive []byte, base, dst string) error {
	gz, err := gzip.NewReader(bytes.NewReader(archive))
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if h.Typeflag == tar.TypeReg && filepath.Base(h.Name) == base {
			b, err := io.ReadAll(io.LimitReader(tr, maxArchive))
			if err != nil {
				return err
			}
			return os.WriteFile(dst, b, 0o755)
		}
	}
	return fmt.Errorf("%s not found in downloaded archive", base)
}

func extractOneZip(archive []byte, base, dst string) error {
	zr, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil {
		return err
	}
	for _, f := range zr.File {
		if !f.FileInfo().IsDir() && filepath.Base(f.Name) == base {
			rc, err := f.Open()
			if err != nil {
				return err
			}
			b, err := io.ReadAll(io.LimitReader(rc, maxArchive))
			rc.Close()
			if err != nil {
				return err
			}
			return os.WriteFile(dst, b, 0o755)
		}
	}
	return fmt.Errorf("%s not found in downloaded archive", base)
}
