import io
import sys

PATH = "internal/delivery/http/router.go"

BAD = '''\tmux.HandleFunc("POST /api/v1/inference/{id}/transcribe", func(w http.ResponseWriter, r *http.Request) {
\t\th.Inference.Transcribe(w, r, r.PathValue("id"))
\t})
'''

GOOD = '''\tmux.HandleFunc("POST /api/v1/inference/{deploymentID}/transcribe", func(w http.ResponseWriter, r *http.Request) {
\t\th.Deployment.Transcribe(w, r, r.PathValue("deploymentID"))
\t})
'''


def main():
    with io.open(PATH, "r", encoding="utf-8", newline="") as f:
        content = f.read()

    if GOOD in content:
        print("SKIP: already fixed")
        return

    if BAD not in content:
        print("BAD ROUTE NOT FOUND")
        sys.exit(1)

    content = content.replace(BAD, GOOD, 1)

    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    print("FIXED transcribe route")


if __name__ == "__main__":
    main()