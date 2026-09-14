package code

import (
	"context"
	"regexp"
	"strings"

	"distillery/internal/domain"
)

// ParseResult is the outcome of parsing a code sample.
type ParseResult struct {
	Valid    bool    `json:"valid"`
	Language string  `json:"language"`
	Metrics  Metrics `json:"metrics"`
}

// ParseError describes a syntax problem found in a sample.
type ParseError struct {
	Line    int    `json:"line"`
	Column  int    `json:"column"`
	Message string `json:"message"`
}

// Metrics are aggregate statistics of a code sample.
type Metrics struct {
	LinesOfCode   int     `json:"lines_of_code"`
	FunctionCount int     `json:"function_count"`
	Complexity    int     `json:"complexity"`
	CommentRatio  float32 `json:"comment_ratio"`
}

// LanguageAnalyzer validates and measures code samples.
type LanguageAnalyzer interface {
	Parse(ctx context.Context, code string) (ParseResult, []ParseError)
	IsValid(code string) bool
	GetComplexity(code string) int
}

// PythonParser analyzes Python code.
type PythonParser struct{}

func (p *PythonParser) Parse(_ context.Context, code string) (ParseResult, []ParseError) {
	if !p.IsValid(code) {
		return ParseResult{
			Valid:    false,
			Language: "python",
			Metrics:  Metrics{LinesOfCode: 0, FunctionCount: 0, Complexity: 0, CommentRatio: 0},
		}, []ParseError{{Line: 1, Message: "Invalid Python syntax"}}
	}

	return ParseResult{Valid: true, Language: "python", Metrics: p.metrics(code)}, nil
}

func (p *PythonParser) IsValid(code string) bool {
	return balancedBrackets(code)
}

func (p *PythonParser) GetComplexity(code string) int {
	c := 1
	for _, kw := range []string{"if ", "elif ", "for ", "while ", "except "} {
		c += strings.Count(code, kw)
	}

	return c
}

func (p *PythonParser) metrics(code string) Metrics {
	lines := strings.Split(code, "\n")
	m := Metrics{LinesOfCode: len(lines), Complexity: p.GetComplexity(code), FunctionCount: 0, CommentRatio: 0}
	re := regexp.MustCompile(`^\s*def\s+`)
	comments := 0

	for _, l := range lines {
		if re.MatchString(l) {
			m.FunctionCount++
		}

		if strings.Contains(l, "#") {
			comments++
		}
	}

	if len(lines) > 0 {
		m.CommentRatio = float32(comments) / float32(len(lines))
	}

	return m
}

// GoParser analyzes Go code.
type GoParser struct{}

func (p *GoParser) Parse(_ context.Context, code string) (ParseResult, []ParseError) {
	if !p.IsValid(code) {
		return ParseResult{
			Valid:    false,
			Language: "go",
			Metrics:  Metrics{LinesOfCode: 0, FunctionCount: 0, Complexity: 0, CommentRatio: 0},
		}, []ParseError{{Line: 1, Message: "Invalid Go syntax"}}
	}

	return ParseResult{Valid: true, Language: "go", Metrics: p.metrics(code)}, nil
}

func (p *GoParser) IsValid(code string) bool {
	return balancedBrackets(code)
}

func (p *GoParser) GetComplexity(code string) int {
	c := 1
	for _, kw := range []string{"if ", "for ", "switch ", "range "} {
		c += strings.Count(code, kw)
	}

	return c
}

func (p *GoParser) metrics(code string) Metrics {
	lines := strings.Split(code, "\n")
	m := Metrics{LinesOfCode: len(lines), Complexity: p.GetComplexity(code), FunctionCount: 0, CommentRatio: 0}
	re := regexp.MustCompile(`^\s*func\s+`)
	comments := 0

	for _, l := range lines {
		if re.MatchString(l) {
			m.FunctionCount++
		}

		if strings.Contains(l, "//") {
			comments++
		}
	}

	if len(lines) > 0 {
		m.CommentRatio = float32(comments) / float32(len(lines))
	}

	return m
}

// JavaScriptParser analyzes JavaScript code.
type JavaScriptParser struct{}

func (p *JavaScriptParser) Parse(_ context.Context, code string) (ParseResult, []ParseError) {
	if !p.IsValid(code) {
		return ParseResult{
			Valid:    false,
			Language: "javascript",
			Metrics:  Metrics{LinesOfCode: 0, FunctionCount: 0, Complexity: 0, CommentRatio: 0},
		}, []ParseError{{Line: 1, Message: "Invalid JavaScript syntax"}}
	}

	return ParseResult{Valid: true, Language: "javascript", Metrics: p.metrics(code)}, nil
}

func (p *JavaScriptParser) IsValid(code string) bool {
	return balancedBrackets(code)
}

func (p *JavaScriptParser) GetComplexity(code string) int {
	c := 1
	for _, kw := range []string{"if ", "for ", "while ", "switch ", "catch "} {
		c += strings.Count(code, kw)
	}

	c += strings.Count(code, "&&")
	c += strings.Count(code, "||")

	return c
}

func (p *JavaScriptParser) metrics(code string) Metrics {
	lines := strings.Split(code, "\n")
	m := Metrics{LinesOfCode: len(lines), Complexity: p.GetComplexity(code), FunctionCount: 0, CommentRatio: 0}
	re := regexp.MustCompile(`(\bfunction\b|=>)`)
	comments := 0

	for _, l := range lines {
		if re.MatchString(l) {
			m.FunctionCount++
		}

		if strings.Contains(l, "//") || strings.Contains(l, "/*") {
			comments++
		}
	}

	if len(lines) > 0 {
		m.CommentRatio = float32(comments) / float32(len(lines))
	}

	return m
}

// NewParser returns an analyzer for a language, or nil if unsupported.
func NewParser(lang domain.ProgrammingLanguage) LanguageAnalyzer {
	switch lang {
	case domain.LangPython:
		return &PythonParser{}
	case domain.LangGo:
		return &GoParser{}
	case domain.LangJavaScript, domain.LangTypeScript:
		return &JavaScriptParser{}
	case domain.LangJava, domain.LangCSharp, domain.LangRust, domain.LangCpp:
		return nil
	default:
		return nil
	}
}

// balancedBrackets checks that (), [], {} pairs are balanced.
func balancedBrackets(code string) bool {
	stack := []rune{}
	open := map[rune]rune{')': '(', ']': '[', '}': '{'}

	for _, ch := range code {
		switch ch {
		case '(', '[', '{':
			stack = append(stack, ch)
		case ')', ']', '}':
			if len(stack) == 0 || stack[len(stack)-1] != open[ch] {
				return false
			}

			stack = stack[:len(stack)-1]
		}
	}

	return len(stack) == 0
}
