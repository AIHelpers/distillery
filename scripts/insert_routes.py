import io
import sys

PATH = "internal/delivery/http/router.go"

ROUTES = '''\tmux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/asr", func(w http.ResponseWriter, r *http.Request) {
\t\th.Dataset.AddASRAudioExample(w, r, r.PathValue("taskID"))
\t})
\tmux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/import-asr-zip", func(w http.ResponseWriter, r *http.Request) {
\t\th.Dataset.ImportASRZIP(w, r, r.PathValue("taskID"))
\t})
\tmux.HandleFunc("GET /api/v1/tasks/{taskID}/examples/asr-stats", func(w http.ResponseWriter, r *http.Request) {
\t\th.Dataset.ASRStats(w, r, r.PathValue("taskID"))
\t})
\tmux.HandleFunc("POST /api/v1/inference/{id}/transcribe", func(w http.ResponseWriter, r *http.Request) {
\t\th.Inference.Transcribe(w, r, r.PathValue("id"))
\t})
\tmux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/vision", func(w http.ResponseWriter, r *http.Request) {
'''

ANCHOR = '''\tmux.HandleFunc("POST /api/v1/tasks/{taskID}/examples/vision", func(w http.ResponseWriter, r *http.Request) {
'''


def main():
    with io.open(PATH, "r", encoding="utf-8", newline="") as f:
        content = f.read()

    if "import-asr-zip" in content:
        print("SKIP: routes already present")
        return

    if ANCHOR not in content:
        print("ANCHOR NOT FOUND")
        sys.exit(1)

    content = content.replace(ANCHOR, ROUTES, 1)

    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    print("INSERTED routes into", PATH)


if __name__ == "__main__":
    main()