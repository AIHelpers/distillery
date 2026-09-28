package pdfraster_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"distillery/internal/infra/pdfraster"
)

// writeStubPython compiles a tiny Go program standing in for the real
// trainer/pdf_rasterize.py: it parses --out-dir the same way the real
// module's CLI would be invoked, then does whatever the given source says
// (write a manifest.json + PNGs, fail with a message on stderr, or produce
// nothing). This tests the Go <-> Python contract (argument shape, manifest
// parsing, page file resolution) without requiring a real Python/PyMuPDF
// install in CI.
//
// An earlier version of this stub was a "#!/bin/sh" script executed
// directly as PythonBin. That relied on the OS interpreting the shebang,
// which Windows does not do for arbitrary files — Rasterize's
// exec.CommandContext would fail to start the process at all, producing no
// output, which firstLine reports as "pdf rasterization failed: unknown
// error" (TestRasterize_Success / TestRasterize_SubprocessFailure would
// fail with exactly that message on a Windows machine even though the Go
// code under test is correct). Compiling a real executable with the
// project's own Go toolchain runs the same way on every OS `go test` runs
// on.
func writeStubPython(t *testing.T, source string) string {
	t.Helper()

	dir := t.TempDir()

	srcPath := filepath.Join(dir, "stub.go")

	err := os.WriteFile(srcPath, []byte(source), 0o600)
	if err != nil {
		t.Fatalf("write stub source: %v", err)
	}

	exeName := "python-stub"
	if runtime.GOOS == "windows" {
		exeName += ".exe"
	}

	exePath := filepath.Join(dir, exeName)

	// Built as a standalone file (no module context: t.TempDir() has no
	// go.mod ancestor), so this only ever needs whatever Go toolchain is
	// already on PATH to run `go test` — it never triggers the surrounding
	// module's own `go` directive/toolchain resolution.
	cmd := exec.Command("go", "build", "-o", exePath, srcPath)

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build stub: %v\n%s", err, out)
	}

	return exePath
}

const successStub = `package main

import (
	"os"
	"path/filepath"
)

func main() {
	outdir := ""
	args := os.Args[1:]

	for i := 0; i < len(args); i++ {
		if args[i] == "--out-dir" && i+1 < len(args) {
			outdir = args[i+1]
		}
	}

	_ = os.WriteFile(filepath.Join(outdir, "page_0001.png"), []byte("\x89PNG\r\n\x1a\ndata"), 0o600)
	_ = os.WriteFile(filepath.Join(outdir, "page_0002.png"), []byte("\x89PNG\r\n\x1a\ndata2"), 0o600)

	manifest := ` + "`" + `{"pages":[{"page":1,"file":"page_0001.png","width":827,"height":1170},{"page":2,"file":"page_0002.png","width":827,"height":1170}],"dpi":100}` + "`" + `
	_ = os.WriteFile(filepath.Join(outdir, "manifest.json"), []byte(manifest), 0o600)
}
`

func TestRasterize_Success(t *testing.T) {
	t.Parallel()

	r := &pdfraster.Rasterizer{PythonBin: writeStubPython(t, successStub), Module: "trainer.pdf_rasterize", Timeout: 30 * time.Second}

	pages, err := r.Rasterize([]byte("%PDF-1.4 fake"), 100)
	if err != nil {
		t.Fatalf("Rasterize: %v", err)
	}

	if len(pages) != 2 {
		t.Fatalf("expected 2 pages, got %d", len(pages))
	}

	if pages[0].Page != 1 || pages[0].Width != 827 || pages[0].Height != 1170 {
		t.Errorf("unexpected page 1 metadata: %+v", pages[0])
	}

	if len(pages[0].PNG) == 0 || len(pages[1].PNG) == 0 {
		t.Error("expected non-empty page image bytes")
	}
}

const failureStub = `package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "FATAL: could not open PDF: corrupt file")
	os.Exit(1)
}
`

func TestRasterize_SubprocessFailure(t *testing.T) {
	t.Parallel()

	r := pdfraster.NewRasterizer(writeStubPython(t, failureStub))
	r.Timeout = 30 * time.Second

	_, err := r.Rasterize([]byte("not a real pdf"), 100)
	if err == nil {
		t.Fatal("expected an error")
	}

	if got := err.Error(); got != "pdf rasterization failed: FATAL: could not open PDF: corrupt file" {
		t.Errorf("unexpected error message: %q", got)
	}
}

func TestRasterize_EmptyInput(t *testing.T) {
	t.Parallel()

	r := pdfraster.NewRasterizer("irrelevant")

	_, err := r.Rasterize(nil, 100)
	if err == nil {
		t.Fatal("expected an error for empty PDF bytes")
	}
}

const noManifestStub = `package main

func main() {}
`

func TestRasterize_MissingManifest(t *testing.T) {
	t.Parallel()

	r := pdfraster.NewRasterizer(writeStubPython(t, noManifestStub))
	r.Timeout = 30 * time.Second

	_, err := r.Rasterize([]byte("%PDF-1.4 fake"), 100)
	if err == nil {
		t.Fatal("expected an error when the manifest is missing")
	}
}

func TestNewRasterizer_Defaults(t *testing.T) {
	t.Parallel()

	r := pdfraster.NewRasterizer("")
	if r.PythonBin != "python" {
		t.Errorf("expected default python bin, got %q", r.PythonBin)
	}

	if r.Module != "trainer.pdf_rasterize" {
		t.Errorf("expected default module, got %q", r.Module)
	}

	if r.Timeout != 2*time.Minute {
		t.Errorf("expected default 2m timeout, got %v", r.Timeout)
	}
}
