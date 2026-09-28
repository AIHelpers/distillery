// Package pdfraster implements domain.PDFRasterizer by shelling out to the
// trainer/pdf_rasterize.py helper (PyMuPDF), the same subprocess pattern the
// local trainer and GGUF exporter use to hand heavyweight work to Python
// without adding a Go-side PDF dependency.
package pdfraster

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"distillery/internal/domain"
)

// ErrRasterizeTimeout is returned when the Python subprocess exceeds Timeout.
var ErrRasterizeTimeout = errors.New("pdf rasterization timed out")

// ErrRasterizeFailed is returned when the Python subprocess exits non-zero.
// The actual failure detail from its output is wrapped alongside it.
var ErrRasterizeFailed = errors.New("pdf rasterization failed")

// ErrNoReadablePages is returned when the rasterizer produced a manifest but
// none of its listed page images could be read back from disk.
var ErrNoReadablePages = errors.New("pdf rasterizer produced no readable pages")

// Rasterizer implements domain.PDFRasterizer.
type Rasterizer struct {
	// PythonBin is the python interpreter to invoke (default "python").
	PythonBin string
	// Module is the module path to run (default "trainer.pdf_rasterize").
	Module string
	// Timeout bounds one rasterization call (default 2 minutes).
	Timeout time.Duration
}

// NewRasterizer creates a Rasterizer, defaulting fields when empty.
func NewRasterizer(pythonBin string) *Rasterizer {
	if pythonBin == "" {
		pythonBin = "python"
	}

	return &Rasterizer{PythonBin: pythonBin, Module: "trainer.pdf_rasterize", Timeout: 2 * time.Minute}
}

// manifest is the JSON pdf_rasterize.py writes to {outDir}/manifest.json.
type manifest struct {
	Pages []struct {
		Page   int    `json:"page"`
		File   string `json:"file"`
		Width  int    `json:"width"`
		Height int    `json:"height"`
	} `json:"pages"`
}

// Rasterize implements domain.PDFRasterizer by writing pdfBytes to a
// temporary file, invoking the Python helper, and reading back the produced
// PNG pages.
func (r *Rasterizer) Rasterize(pdfBytes []byte, maxDPI int) ([]domain.RasterizedPage, error) {
	if len(pdfBytes) == 0 {
		return nil, fmt.Errorf("%w: empty PDF", domain.ErrInvalidInput)
	}

	workDir, err := os.MkdirTemp("", "distillery-pdf-*")
	if err != nil {
		return nil, err
	}

	defer os.RemoveAll(workDir)

	pdfPath := filepath.Join(workDir, "input.pdf")

	err = os.WriteFile(pdfPath, pdfBytes, 0o600)
	if err != nil {
		return nil, err
	}

	outDir := filepath.Join(workDir, "out")

	err = os.MkdirAll(outDir, 0o755)
	if err != nil {
		return nil, err
	}

	if maxDPI <= 0 {
		maxDPI = 150
	}

	timeout := r.Timeout
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	pythonBin := r.PythonBin
	if pythonBin == "" {
		pythonBin = "python"
	}

	module := r.Module
	if module == "" {
		module = "trainer.pdf_rasterize"
	}

	cmd := exec.CommandContext(ctx, pythonBin, "-m", module,
		"--input", pdfPath,
		"--out-dir", outDir,
		"--dpi", strconv.Itoa(maxDPI),
	)

	output, runErr := cmd.CombinedOutput()

	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, ErrRasterizeTimeout
	}

	if runErr != nil {
		return nil, fmt.Errorf("%w: %s", ErrRasterizeFailed, firstLine(string(output)))
	}

	manifestData, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("pdf rasterizer produced no manifest: %w", err)
	}

	var m manifest

	err = json.Unmarshal(manifestData, &m)
	if err != nil {
		return nil, fmt.Errorf("malformed rasterizer manifest: %w", err)
	}

	pages := make([]domain.RasterizedPage, 0, len(m.Pages))

	for _, p := range m.Pages {
		// filepath.Base neutralizes gosec's taint tracking on a path built
		// from the (trusted, script-generated) manifest filename.
		data, err := os.ReadFile(filepath.Join(outDir, filepath.Base(p.File)))
		if err != nil {
			continue
		}

		pages = append(pages, domain.RasterizedPage{Page: p.Page, PNG: data, Width: p.Width, Height: p.Height})
	}

	if len(pages) == 0 {
		return nil, ErrNoReadablePages
	}

	return pages, nil
}

// firstLine returns the first non-empty line of s, for a compact error
// message from a subprocess's (potentially long) combined output.
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			return line
		}
	}

	return "unknown error"
}
