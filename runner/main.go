// // file: cli.go
// package main

// import (
// 	"bufio"
// 	"context"
// 	"flag"
// 	"fmt"
// 	"os"
// 	"os/exec"
// 	"strings"
// 	"time"
// )

// // result model
// type TestResult struct {
// 	ID, Status, Out, Err string
// }

// // store
// type Store struct {
// 	data map[string]*TestResult
// }

// func NewStore() *Store {
// 	return &Store{data: make(map[string]*TestResult)}
// }

// // file writer
// func writeFile(r *TestResult) {
// 	os.MkdirAll("results", 0755)

// 	f, err := os.Create("results/" + r.ID + ".txt")
// 	if err != nil {
// 		fmt.Println("file error:", err)
// 		return
// 	}
// 	defer f.Close()

// 	fmt.Fprintf(f, "ID: %s\nSTATUS: %s\n\n", r.ID, r.Status)

// 	fmt.Fprintln(f, "=== STDOUT ===")
// 	fmt.Fprintln(f, r.Out)

// 	fmt.Fprintln(f, "\n=== STDERR ===")
// 	fmt.Fprintln(f, r.Err)
// }

// // failure detector (important)
// func hasFailure(out string) bool {
// 	return strings.Contains(out, "checks_failed") ||
// 		strings.Contains(out, "http_req_failed") ||
// 		strings.Contains(out, "ERRO")
// }

// // worker
// type Worker struct {
// 	ID, Script, Dur string
// 	VUs             int
// 	store           *Store
// }

// func (w *Worker) Run(ctx context.Context) {

// 	res := &TestResult{ID: w.ID, Status: "running"}
// 	w.store.data[w.ID] = res

// 	cmd := exec.CommandContext(ctx, "k6", "run",
// 		"--vus", fmt.Sprintf("%d", w.VUs),
// 		"--duration", w.Dur,
// 		w.Script,
// 	)

// 	stdout, _ := cmd.StdoutPipe()
// 	stderr, _ := cmd.StderrPipe()

// 	var out, errStr string

// 	// stdout → success stream
// 	go func() {
// 		sc := bufio.NewScanner(stdout)
// 		for sc.Scan() {
// 			line := sc.Text()
// 			fmt.Println("[OUT]", line)
// 			out += line + "\n"
// 		}
// 	}()

// 	// stderr → error stream
// 	go func() {
// 		sc := bufio.NewScanner(stderr)
// 		for sc.Scan() {
// 			line := sc.Text()
// 			fmt.Println("[ERR]", line)
// 			errStr += line + "\n"
// 		}
// 	}()

// 	start := time.Now()

// 	if err := cmd.Start(); err != nil {
// 		fmt.Println("cmd start error:", err)
// 		return
// 	}

// 	err := cmd.Wait()

// 	// result assignment
// 	res.Out = out
// 	res.Err = errStr

// 	// smarter failure logic
// 	if err != nil || errStr != "" || hasFailure(out) {
// 		res.Status = "FAILED"
// 	} else {
// 		res.Status = "SUCCESS"
// 	}

// 	fmt.Println("Finished in:", time.Since(start))
// 	writeFile(res)
// }

// // main
// func main() {

// 	// CLI args
// 	script := flag.String("script", "", "path to k6 script")
// 	vus := flag.Int("vus", 2, "number of VUs")
// 	dur := flag.String("duration", "5s", "test duration")
// 	flag.Parse()

// 	if *script == "" {
// 		fmt.Println("❌ provide --script")
// 		return
// 	}

// 	// validate file exists
// 	if _, err := os.Stat(*script); err != nil {
// 		fmt.Println("script not found:", *script)
// 		return
// 	}

// 	store := NewStore()

// 	worker := &Worker{
// 		ID:     fmt.Sprintf("test_%d", time.Now().UnixNano()),
// 		Script: *script,
// 		VUs:    *vus,
// 		Dur:    *dur,
// 		store:  store,
// 	}

// 	fmt.Println("Running:", *script)
// 	worker.Run(context.Background())
// 
//}