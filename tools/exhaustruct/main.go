// Command exhaustruct rewrites every composite literal of a struct type
// annotated with //exhaustruct:enforce so that all fields are present,
// inserting zero values for the missing ones (adding imports as needed).
//
// Usage: go run tools/exhaustruct.
package main

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// fieldDef maps field name -> zero-value Go source.
type fieldDef struct {
	Name string
	Zero string
}

const (
	domainImport = "distillery/internal/domain"
	codeImport   = "distillery/internal/infra/code"
)

var errNoExpr = errors.New("no expr parsed")

// zerotype returns a fully-qualified key used by zeroval; empty if scalar.
func zeroval(fullType string) string {
	switch fullType {
	case "string":
		return `""`
	case "int", "int64", "uint", "float64", "float32", "time.Duration":
		return "0"
	case "bool":
		return "false"
	case "time.Time":
		return "nil"
	case "domain.ExampleSource", "domain.TaskType", "domain.TrainingStatus",
		"domain.DeploymentStatus", "domain.ProgrammingLanguage", "domain.SkillCategory",
		"domain.RequestStatus", "domain.JobStatus", "domain.OptimizationAction",
		"domain.MetricType":
		return "nil" // these are string-typed consts; "" handled below.
	case "[2]int":
		return "[2]int{}"
	default:
		if strings.HasPrefix(fullType, "*") || strings.HasPrefix(fullType, "[]") ||
			strings.HasPrefix(fullType, "map[") || strings.HasPrefix(fullType, "func") ||
			fullType == "interface{}" {
			return "nil"
		}
		// Unknown value type; fall back to nil is a compile error. Signal it.
		return "\x00UNHANDLED:" + fullType
	}
}

func zeroValue(fullType string, fset *token.FileSet, f *ast.File, pkgName string) ast.Expr {
	zv := zeroval(fullType)

	// String-typed const types: zero is "".
	switch fullType {
	case "domain.ExampleSource", "domain.TaskType", "domain.TrainingStatus",
		"domain.DeploymentStatus", "domain.ProgrammingLanguage", "domain.SkillCategory",
		"domain.RequestStatus", "domain.JobStatus", "domain.OptimizationAction",
		"domain.MetricType":
		return &ast.BasicLit{Kind: token.STRING, Value: `""`}
	}

	// Value structs with no enforced fields: emit a bare complete literal.
	// Handle before the generic nil check so time.Time etc. become a real
	// composite literal rather than nil.
	switch fullType {
	case "time.Time", "domain.CheckpointInfo", "domain.ValidationParameters":
		pkg, ty := "time", "Time"
		if strings.HasPrefix(fullType, "domain.") {
			pkg, ty = "domain", strings.TrimPrefix(fullType, "domain.")
		}

		if fullType == "time.Time" && pkgName != "time" {
			addImport(f, "time", "time")
		}

		return typeComposite(fullType, pkg, ty, f, pkgName, nil)
	}

	// Nil-able zero-value types.
	if zv == "nil" {
		return &ast.Ident{Name: "nil"}
	}

	// Named composite type present in FIELDS → build a complete zero literal
	// (recursively zeroing each sub-field). Ensures no partial nested literal
	// triggers exhaustruct again.
	if nested, ok := FIELDS2[fullType]; ok && strings.Contains(fullType, ".") {
		var elts []ast.Expr
		for _, fd := range nested {
			elts = append(elts, &ast.KeyValueExpr{
				Key:   ast.NewIdent(fd.Name),
				Value: zeroValue(fd.Zero, fset, f, pkgName),
			})
		}

		pkg, ty := splitType(fullType)

		return typeComposite(fullType, pkg, ty, f, pkgName, elts)
	}

	// [2]int or similar fixed arrays.
	if zv == "[2]int{}" {
		return &ast.CompositeLit{Type: arrayTypeExpr(fullType)}
	}

	// Build an AST from the zero-value source snippet.
	expr, err := parseExpr(fset, zv)
	if err != nil {
		return &ast.Ident{Name: "nil"}
	}

	return expr
}

func splitType(full string) (pkg, name string) {
	idx := strings.LastIndex(full, ".")
	pkg = full[:idx]
	name = full[idx+1:]

	return
}

func arrayTypeExpr(fullType string) ast.Expr {
	// fullType is like [2]int.
	return &ast.ArrayType{
		Len: ast.NewIdent(strings.TrimSuffix(strings.TrimPrefix(fullType, "["), "]int")),
		Elt: ast.NewIdent("int"),
	}
}

// FIELDS2 mirrors FIELDS but keyed by plain "domain.X"/"code.X" names for
// nested zero-literal construction.
var FIELDS2 = FIELDS

// typeComposite builds a composite literal type expression for fullType
// (like "domain.ToolCall" or "code.CodeMetrics"), emitting an unqualified
// identifier when the target package is the file's own package.
func typeComposite(fullType, pkg, ty string, f *ast.File, pkgName string, elts []ast.Expr) ast.Expr {
	var typeExpr ast.Expr
	if pkgName == pkg {
		// Same package: unqualified identifier (avoid self-import).
		typeExpr = ast.NewIdent(ty)
	} else {
		if strings.HasPrefix(fullType, "domain.") {
			addImport(f, domainImport, "domain")
		} else if strings.HasPrefix(fullType, "code.") {
			addImport(f, codeImport, "code")
		}

		typeExpr = &ast.SelectorExpr{
			X:   ast.NewIdent(pkg),
			Sel: ast.NewIdent(ty),
		}
	}

	return &ast.CompositeLit{Type: typeExpr, Elts: elts}
}

func parseExpr(fset *token.FileSet, src string) (ast.Expr, error) {
	f, err := parser.ParseFile(fset, "", "package p\nvar _ = "+src, parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}

	for _, d := range f.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			continue
		}

		for _, s := range gd.Specs {
			vs, ok := s.(*ast.ValueSpec)
			if !ok || len(vs.Values) == 0 {
				continue
			}

			return vs.Values[0], nil
		}
	}

	return nil, errNoExpr
}

// addImport ensures a package is imported under the given local name (or default).
func addImport(f *ast.File, path, name string) {
	for _, imp := range f.Imports {
		if imp.Path.Value == `"`+path+`"` {
			return // already imported.
		}
	}
	// Insert into the first import decl (or create one).
	imp := &ast.ImportSpec{
		Path: &ast.BasicLit{Kind: token.STRING, Value: `"` + path + `"`},
	}
	if name != "" {
		imp.Name = ast.NewIdent(name)
	}

	for _, d := range f.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.IMPORT {
			continue
		}

		gd.Specs = append(gd.Specs, imp)
		f.Imports = append(f.Imports, imp) // keep f.Imports in sync.

		return
	}
	// No import block yet; append a new GenDecl after package clause.
	gd := &ast.GenDecl{Tok: token.IMPORT, Specs: []ast.Spec{imp}}
	f.Decls = append([]ast.Decl{gd}, f.Decls...)
	f.Imports = append(f.Imports, imp) // keep f.Imports in sync.
}

// FIELDS maps fully-qualified linter type => ordered fields (name, go type).
var FIELDS = map[string][]fieldDef{
	"domain.ToolInput":  {{"Params", "map[string]interface{}"}},
	"domain.ToolOutput": {{"Result", "interface{}"}, {"Error", "string"}},
	"domain.ToolCall": {
		{"ID", "string"},
		{"ToolName", "string"},
		{"Input", "map[string]interface{}"},
		{"Status", "string"},
		{"Output", "domain.ToolOutput"},
		{"Error", "string"},
	},
	"domain.Message": {
		{"Role", "string"},
		{"Content", "string"},
		{"ToolCalls", "[]domain.ToolCall"},
		{"Timestamp", "time.Time"},
	},
	"domain.AgentState": {
		{"ID", "string"},
		{"Status", "string"},
		{"CurrentTask", "string"},
		{"Progress", "int"},
		{"ToolCalls", "[]domain.ToolCall"},
		{"History", "[]domain.Message"},
		{"Error", "string"},
		{"LastActivity", "time.Time"},
		{"Metadata", "map[string]string"},
	},
	"domain.AgentConfig": {
		{"Name", "string"},
		{"SystemPrompt", "string"},
		{"MaxIterations", "int"},
		{"Timeout", "time.Duration"},
		{"TemperatureHint", "float32"},
		{"TopPHint", "float32"},
	},
	"domain.Example": {
		{"ID", "string"},
		{"TaskID", "string"},
		{"Input", "string"},
		{"Output", "string"},
		{"Source", "domain.ExampleSource"},
		{"Flagged", "bool"},
		{"FlagNote", "string"},
		{"Duplicate", "bool"},
		{"CreatedAt", "time.Time"},
	},
	"domain.DatasetStats": {
		{"TaskID", "string"},
		{"Total", "int"},
		{"Duplicates", "int"},
		{"Flagged", "int"},
		{"Synthetic", "int"},
		{"UserProvided", "int"},
		{"Feedback", "int"},
		{"UsableCount", "int"},
		{"LabelBalance", "map[string]int"},
		{"ReadyToTrain", "bool"},
		{"ReadinessReason", "string"},
	},
	"domain.TrainingParameters": {
		{"Epochs", "int"},
		{"BatchSize", "int"},
		{"LearningRate", "float64"},
		{"WarmupSteps", "int"},
		{"MaxSequenceLength", "int"},
		{"GradientAccumSteps", "int"},
		{"WeightDecay", "float64"},
		{"SchedulerType", "string"},
		{"OptimizerType", "string"},
		{"PreserveSyntax", "bool"},
		{"ContextWindow", "int"},
		{"BalancedSampling", "bool"},
	},
	"domain.MetricPoint": {
		{"Step", "int"}, {"Epoch", "int"}, {"Value", "float64"}, {"Timestamp", "time.Time"},
	},
	"domain.FineTuneRequest": {
		{"ID", "string"},
		{"Name", "string"},
		{"Description", "string"},
		{"Language", "domain.ProgrammingLanguage"},
		{"Skill", "domain.SkillCategory"},
		{"BaseModel", "string"},
		{"DatasetID", "string"},
		{"TrainingParams", "domain.TrainingParameters"},
		{"ValidationParams", "domain.ValidationParameters"},
		{"Owner", "string"},
		{"Status", "domain.RequestStatus"},
		{"CreatedAt", "time.Time"},
		{"UpdatedAt", "time.Time"},
	},
	"domain.FineTuneJob": {
		{"ID", "string"},
		{"RequestID", "string"},
		{"Status", "domain.JobStatus"},
		{"Progress", "float64"},
		{"CurrentEpoch", "int"},
		{"TotalEpochs", "int"},
		{"CurrentStep", "int"},
		{"TotalSteps", "int"},
		{"Loss", "[]domain.MetricPoint"},
		{"ValidationLoss", "[]domain.MetricPoint"},
		{"LearningRate", "[]domain.MetricPoint"},
		{"CustomMetrics", "map[string][]domain.MetricPoint"},
		{"FinalMetrics", "map[string]float64"},
		{"Error", "string"},
		{"BestCheckpoint", "domain.CheckpointInfo"},
		{"OutputModelID", "string"},
		{"ComputeCost", "float64"},
		{"GPUHours", "float64"},
		{"CreatedAt", "time.Time"},
		{"StartedAt", "*time.Time"},
		{"CompletedAt", "*time.Time"},
	},
	"domain.TrainedModel": {
		{"ID", "string"},
		{"Name", "string"},
		{"Language", "domain.ProgrammingLanguage"},
		{"Skill", "domain.SkillCategory"},
		{"BaseModel", "string"},
		{"FineTuneJobID", "string"},
		{"Version", "string"},
		{"Status", "string"},
		{"Checkpoint", "string"},
		{"HuggingFaceURL", "string"},
		{"Quantized", "bool"},
		{"QuantBits", "int"},
		{"TrainingMetrics", "map[string]float64"},
		{"BenchmarkScore", "float64"},
		{"Owner", "string"},
		{"IsPublic", "bool"},
		{"Downloads", "int"},
		{"Rating", "float64"},
		{"CreatedAt", "time.Time"},
	},
	"domain.DatasetQuality": {
		{"OverallScore", "float64"},
		{"ValidityRate", "float64"},
		{"ComplexityScore", "float64"},
		{"Issues", "[]string"},
	},
	"domain.DatasetInfo": {
		{"ID", "string"},
		{"Name", "string"},
		{"Language", "domain.ProgrammingLanguage"},
		{"FileCount", "int"},
		{"TotalSize", "int64"},
		{"TotalLines", "int"},
		{"TotalTokens", "int64"},
		{"SampleCount", "int"},
		{"Quality", "domain.DatasetQuality"},
		{"Status", "string"},
		{"Owner", "string"},
		{"CreatedAt", "time.Time"},
	},
	"domain.DatasetAnalysis": {
		{"FileCount", "int"},
		{"TotalSize", "int64"},
		{"TotalTokens", "int64"},
		{"LanguageCoverage", "map[string]int"},
		{"ComplexityRange", "[2]int"},
		{"AverageComplexity", "float64"},
		{"SyntaxValidityRate", "float64"},
		{"Recommendations", "[]string"},
		{"ReadyForTraining", "bool"},
		{"QualityScore", "float64"},
	},
	"domain.HyperparameterRecommendationReq": {
		{"Language", "domain.ProgrammingLanguage"},
		{"Skill", "domain.SkillCategory"},
		{"DatasetSize", "int64"},
		{"AvailableGPU", "int"},
	},
	"domain.TrainingOptimization": {
		{"CurrentStep", "int"},
		{"CurrentLoss", "float64"},
		{"LossDirection", "string"},
		{"Suggestion", "string"},
		{"Action", "domain.OptimizationAction"},
		{"Confidence", "float64"},
	},
	"domain.QualityReport": {
		{"TrainingComplete", "bool"},
		{"FinalLoss", "float64"},
		{"BestMetrics", "map[string]float64"},
		{"CodeExecutability", "float64"},
		{"SyntaxValidity", "float64"},
		{"OverallQuality", "string"},
		{"Issues", "[]string"},
		{"Recommendations", "[]string"},
	},
	"domain.AgentInsights": {
		{"Phase", "string"},
		{"Summary", "string"},
		{"Metrics", "map[string]interface{}"},
		{"Warnings", "[]string"},
		{"NextSteps", "[]string"},
		{"EstimatedQuality", "string"},
	},
	"domain.Task": {
		{"ID", "string"},
		{"Name", "string"},
		{"Description", "string"},
		{"Type", "domain.TaskType"},
		{"CreatedAt", "time.Time"},
		{"UpdatedAt", "time.Time"},
	},
	"domain.BaseModel": {
		{"Name", "string"}, {"ParamsBillions", "float64"}, {"Family", "string"},
	},
	"domain.TrainingMetrics": {
		{"FinalLoss", "float64"}, {"EvalAccuracy", "float64"}, {"Epochs", "int"}, {"TrainExamples", "int"},
	},
	"domain.TrainingJob": {
		{"ID", "string"},
		{"TaskID", "string"},
		{"Version", "int"},
		{"BaseModel", "domain.BaseModel"},
		{"Status", "domain.TrainingStatus"},
		{"Progress", "int"},
		{"Metrics", "*domain.TrainingMetrics"},
		{"Error", "string"},
		{"CreatedAt", "time.Time"},
		{"StartedAt", "*time.Time"},
		{"CompletedAt", "*time.Time"},
	},
	"domain.Deployment": {
		{"ID", "string"},
		{"TaskID", "string"},
		{"TrainingJobID", "string"},
		{"Endpoint", "string"},
		{"Autoscale", "bool"},
		{"Status", "domain.DeploymentStatus"},
		{"RequestCount", "int"},
		{"APIKeyHash", "string"},
		{"CreatedAt", "time.Time"},
	},
	"domain.Misprediction": {
		{"ID", "string"},
		{"TaskID", "string"},
		{"Input", "string"},
		{"ActualOutput", "string"},
		{"ExpectedOutput", "string"},
		{"Resolved", "bool"},
		{"CreatedAt", "time.Time"},
	},
	"code.ParseResult": {
		{"Valid", "bool"}, {"Language", "string"}, {"Metrics", "code.CodeMetrics"},
	},
	"code.CodeMetrics": {
		{"LinesOfCode", "int"}, {"FunctionCount", "int"}, {"Complexity", "int"}, {"CommentRatio", "float32"},
	},
}

// typeKey resolves a composite literal's type expression to a FIELDS key.
func typeKey(expr ast.Expr, pkgName string) string {
	switch t := expr.(type) {
	case *ast.Ident:
		if pkgName == "domain" && strings.HasPrefix(t.Name, "ToolCall") {
			return "domain." + t.Name
		}

		if pkgName == "domain" {
			if _, ok := FIELDS["domain."+t.Name]; ok {
				return "domain." + t.Name
			}
		}

		if pkgName == "code" {
			if _, ok := FIELDS["code."+t.Name]; ok {
				return "code." + t.Name
			}
		}

		return ""
	case *ast.SelectorExpr:
		id, ok := t.X.(*ast.Ident)
		if !ok {
			return ""
		}

		full := id.Name + "." + t.Sel.Name
		if _, ok := FIELDS[full]; ok {
			return full
		}
		// Handle alias: package imported under a different local name.
		return ""
	}

	return ""
}

func processFile(_, pkgName string, fset *token.FileSet, f *ast.File) (changed bool) {
	ast.Inspect(f, func(n ast.Node) bool {
		cl, ok := n.(*ast.CompositeLit)
		if !ok || cl.Type == nil {
			return true
		}

		key := typeKey(cl.Type, pkgName)
		if key == "" {
			return true
		}

		fields := FIELDS[key]
		present := map[string]bool{}

		for _, el := range cl.Elts {
			kv, ok := el.(*ast.KeyValueExpr)
			if !ok {
				continue
			}

			id, ok := kv.Key.(*ast.Ident)
			if !ok {
				continue
			}

			present[id.Name] = true
		}
		// Compute missing fields.
		missing := []fieldDef{}

		for _, fd := range fields {
			if !present[fd.Name] {
				missing = append(missing, fd)
			}
		}

		if len(missing) == 0 {
			return true
		}

		var inserted []ast.Expr
		for _, fd := range missing {
			inserted = append(inserted, &ast.KeyValueExpr{
				Key:   ast.NewIdent(fd.Name),
				Value: zeroValue(fd.Zero, fset, f, pkgName),
			})
		}

		cl.Elts = append(cl.Elts, inserted...)
		changed = true

		return true
	})

	return changed
}

func main() {
	root := "internal"

	var filepaths []string

	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}

		filepaths = append(filepaths, path)

		return nil
	})

	total := 0

	for _, path := range filepaths {
		fset := token.NewFileSet()

		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			fmt.Fprintln(os.Stderr, "parse:", path, err)
			continue
		}

		if processFile(path, f.Name.Name, fset, f) {
			// Format and write back.
			var buf bytes.Buffer

			err = printer.Fprint(&buf, fset, f)
			if err != nil {
				fmt.Fprintln(os.Stderr, "print:", path, err)
				continue
			}

			formatted, err := format.Source(buf.Bytes())
			if err != nil {
				fmt.Fprintln(os.Stderr, "format:", path, err)
				continue
			}

			err = os.WriteFile(path, formatted, 0o644)
			if err != nil {
				fmt.Fprintln(os.Stderr, "write:", path, err)
				continue
			}

			total++

			fmt.Println("rewrote", path)
		}
	}

	fmt.Printf("rewrote %d files\n", total)
}
