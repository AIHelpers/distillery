// Package web embeds the static single-page UI so the server binary is
// fully self-contained — no separate asset directory needed at runtime,
// which keeps both the desktop and container deployment paths simple.
package web

import (
	"embed"
	"io/fs"
)

//go:embed index.html agent.html static/*
var files embed.FS

// FS returns the embedded web UI filesystem rooted at the web directory.
func FS() fs.FS {
	return files
}
