package code_test

import (
	"context"
	"testing"

	"distillery/internal/domain"
	"distillery/internal/infra/code"
)

func TestNewParser_Python(t *testing.T) {
	p := code.NewParser(domain.LangPython)
	if p == nil {
		t.Fatal("expected non-nil parser for python")
	}
}

func TestNewParser_Go(t *testing.T) {
	p := code.NewParser(domain.LangGo)
	if p == nil {
		t.Fatal("expected non-nil parser for go")
	}
}

func TestNewParser_JavaScript(t *testing.T) {
	p := code.NewParser(domain.LangJavaScript)
	if p == nil {
		t.Fatal("expected non-nil parser for javascript")
	}
}

func TestNewParser_Unsupported(t *testing.T) {
	p := code.NewParser(domain.ProgrammingLanguage("cobol"))
	if p != nil {
		t.Fatal("expected nil parser for unsupported language")
	}
}

func TestPythonParser_IsValid_Balanced(t *testing.T) {
	p := code.NewParser(domain.LangPython)
	valid := p.IsValid("def foo():\n    return [1, 2, (3)]")
	if !valid {
		t.Error("expected balanced brackets to be valid")
	}
}

func TestPythonParser_IsValid_Unbalanced(t *testing.T) {
	p := code.NewParser(domain.LangPython)
	valid := p.IsValid("def foo():\n    return [1, 2")
	if valid {
		t.Error("expected unbalanced brackets to be invalid")
	}
}

func TestPythonParser_GetComplexity(t *testing.T) {
	p := code.NewParser(domain.LangPython)
	code := "if x:\n    for y in z:\n        while a:\n            pass"
	complexity := p.GetComplexity(code)
	if complexity != 4 { // 1 base + 1 if + 1 for + 1 while
		t.Errorf("expected complexity 4, got %d", complexity)
	}
}

func TestPythonParser_Parse_Valid(t *testing.T) {
	p := code.NewParser(domain.LangPython)
	sample := "def foo():\n    return 1\n# comment\ndef bar():\n    return 2"
	result, errs := p.Parse(context.Background(), sample)
	if !result.Valid {
		t.Error("expected valid parse result")
	}
	if len(errs) != 0 {
		t.Errorf("expected no parse errors, got %d", len(errs))
	}
	if result.Language != "python" {
		t.Errorf("expected language 'python', got %q", result.Language)
	}
	if result.Metrics.LinesOfCode != 5 {
		t.Errorf("expected 5 lines, got %d", result.Metrics.LinesOfCode)
	}
	if result.Metrics.FunctionCount != 2 {
		t.Errorf("expected 2 functions, got %d", result.Metrics.FunctionCount)
	}
	if result.Metrics.CommentRatio <= 0 {
		t.Error("expected positive comment ratio")
	}
}

func TestPythonParser_Parse_Invalid(t *testing.T) {
	p := code.NewParser(domain.LangPython)
	result, errs := p.Parse(context.Background(), "def foo(:\n    return 1")
	if result.Valid {
		t.Error("expected invalid parse result")
	}
	if len(errs) == 0 {
		t.Error("expected parse errors")
	}
}

func TestGoParser_IsValid_Balanced(t *testing.T) {
	p := code.NewParser(domain.LangGo)
	valid := p.IsValid("func main() {\n    x := []int{1, 2}\n}")
	if !valid {
		t.Error("expected balanced brackets to be valid")
	}
}

func TestGoParser_IsValid_Unbalanced(t *testing.T) {
	p := code.NewParser(domain.LangGo)
	valid := p.IsValid("func main() {\n    x := []int{1, 2}")
	if valid {
		t.Error("expected unbalanced brackets to be invalid")
	}
}

func TestGoParser_GetComplexity(t *testing.T) {
	p := code.NewParser(domain.LangGo)
	complexity := p.GetComplexity("if x {\n    for i := range y {\n        switch z {\n        }\n    }\n}")
	// 1 base + 1 if + 1 for + 1 range + 1 switch = 5
	if complexity != 5 {
		t.Errorf("expected complexity 5, got %d", complexity)
	}
}

func TestGoParser_Parse_Valid(t *testing.T) {
	p := code.NewParser(domain.LangGo)
	sample := "package main\n\nfunc foo() int {\n    return 1\n}\n\nfunc bar() int {\n    return 2\n}"
	result, errs := p.Parse(context.Background(), sample)
	if !result.Valid {
		t.Error("expected valid parse result")
	}
	if len(errs) != 0 {
		t.Errorf("expected no parse errors, got %d", len(errs))
	}
	if result.Language != "go" {
		t.Errorf("expected language 'go', got %q", result.Language)
	}
	if result.Metrics.FunctionCount != 2 {
		t.Errorf("expected 2 functions, got %d", result.Metrics.FunctionCount)
	}
}

func TestGoParser_Parse_Invalid(t *testing.T) {
	p := code.NewParser(domain.LangGo)
	result, errs := p.Parse(context.Background(), "func main( {\n    return 1\n}")
	if result.Valid {
		t.Error("expected invalid parse result")
	}
	if len(errs) == 0 {
		t.Error("expected parse errors")
	}
}

func TestJavaScriptParser_IsValid_Balanced(t *testing.T) {
	p := code.NewParser(domain.LangJavaScript)
	valid := p.IsValid("function foo() {\n    return [1, 2];\n}")
	if !valid {
		t.Error("expected balanced brackets to be valid")
	}
}

func TestJavaScriptParser_IsValid_Unbalanced(t *testing.T) {
	p := code.NewParser(domain.LangJavaScript)
	valid := p.IsValid("const x = {a: 1, b: [2, 3}")
	if valid {
		t.Error("expected unbalanced brackets to be invalid")
	}
}

func TestJavaScriptParser_GetComplexity(t *testing.T) {
	p := code.NewParser(domain.LangJavaScript)
	complexity := p.GetComplexity("if (x) {\n    for (let i = 0; i < y; i++) {\n        if (a && b) {\n        }\n    }\n}")
	// 1 base + 2 if + 1 for + 1 && = 5
	if complexity != 5 {
		t.Errorf("expected complexity 5, got %d", complexity)
	}
}

func TestJavaScriptParser_Parse_Valid(t *testing.T) {
	p := code.NewParser(domain.LangJavaScript)
	sample := "function foo() {\n    return 1;\n}\n\nconst bar = () => {\n    return 2;\n}"
	result, errs := p.Parse(context.Background(), sample)
	if !result.Valid {
		t.Error("expected valid parse result")
	}
	if len(errs) != 0 {
		t.Errorf("expected no parse errors, got %d", len(errs))
	}
	if result.Language != "javascript" {
		t.Errorf("expected language 'javascript', got %q", result.Language)
	}
	if result.Metrics.FunctionCount != 2 {
		t.Errorf("expected 2 functions, got %d", result.Metrics.FunctionCount)
	}
}

func TestJavaScriptParser_Parse_Invalid(t *testing.T) {
	p := code.NewParser(domain.LangJavaScript)
	result, errs := p.Parse(context.Background(), "function foo( {\n    return [1, 2;")
	if result.Valid {
		t.Error("expected invalid parse result")
	}
	if len(errs) == 0 {
		t.Error("expected parse errors")
	}
}

func TestPythonParser_Metrics_CommentRatio(t *testing.T) {
	p := code.NewParser(domain.LangPython)
	sample := "def foo():\n    return 1\n# comment1\n# comment2\nx = 'just a string'"
	result, _ := p.Parse(context.Background(), sample)
	// 5 lines total. Comment lines: line 3 (# comment1), line 4 (# comment2).
	// Line 5 "x = 'just a string'" contains no #, so commentRatio = 2/5 = 0.4
	if result.Metrics.CommentRatio != 0.4 {
		t.Errorf("expected comment ratio 0.4, got %f", result.Metrics.CommentRatio)
	}
}
