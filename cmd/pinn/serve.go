package main

import (
	"flag"
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"runtime"
	"time"
)

func cmdServe(args []string) error {
	fl := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fl.String("addr", "127.0.0.1:8000", "listen address")
	noOpen := fl.Bool("no-open", false, "do not open the browser")
	if err := fl.Parse(args); err != nil {
		return err
	}
	page, err := hubPage()
	if err != nil {
		return err
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		return fmt.Errorf("cannot listen on %s: %w (try -addr 127.0.0.1:0 for a free port)", *addr, err)
	}
	url := "http://" + ln.Addr().String()
	fmt.Printf("pinn: hub at %s  (Ctrl-C to stop)\n", url)
	fmt.Println("pinn: labs -> `pinn lab 02` | starter -> `pinn run 00`")
	if !*noOpen {
		go func() {
			time.Sleep(150 * time.Millisecond)
			openBrowser(url)
		}()
	}
	return http.Serve(ln, hubHandler(page))
}

// hubHandler serves the embedded page at / and nothing else.
func hubHandler(page []byte) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" && r.URL.Path != "/index.html" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		_, _ = w.Write(page)
	})
}

func openBrowser(url string) {
	var c *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		c = exec.Command("open", url)
	case "windows":
		c = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		c = exec.Command("xdg-open", url)
	}
	_ = c.Start()
}
