package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

// -------- MODEL --------
type Result struct {
	ID, Status, Logs string
}

// -------- STORE --------
type Store struct {
	mu sync.Mutex
	m  map[string]*Result
}

func (s *Store) Set(id string, r *Result) {
	s.mu.Lock()
	s.m[id] = r
	s.mu.Unlock()
}

func (s *Store) Get(id string) *Result {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.m[id]
}

// -------- SEGREGATOR (FAN-OUT) --------
type Segregator struct {
	Out chan string
	Err chan string
}

func NewSeg() *Segregator {
	return &Segregator{
		Out: make(chan string, 100),
		Err: make(chan string, 100),
	}
}

func (s *Segregator) Start(stdout, stderr interface{ Read([]byte) (int, error) }) {

	// stdout → success
	go func() {
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			line := sc.Text()
			fmt.Println("[OUT]", line)
			s.Out <- line
		}
		close(s.Out)
	}()

	// stderr → error
	go func() {
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			line := sc.Text()
			fmt.Println("[ERR]", line)
			s.Err <- line
		}
		close(s.Err)
	}()
}

// -------- AGGREGATOR (FAN-IN) --------
type Aggregator struct {
	ch chan string
}

func NewAgg() *Aggregator {
	return &Aggregator{ch: make(chan string, 200)}
}

func (a *Aggregator) Start(buf *string) {
	// return a done channel so caller can wait until aggregation finishes
	// when a.ch is closed
	go func() {
		for msg := range a.ch {
			*buf += msg + "\n"
		}
	}()
}

// -------- WORKER --------
type Worker struct {
	ID, Script string
	VUs        int
	Dur        string
	store      *Store
	// Temp marks that Script was created from an incoming POST body and
	// should be removed after the run completes.
	Temp bool
	// Tag is an optional user-supplied tag appended to result filenames
	Tag string
}

func (w *Worker) Run(ctx context.Context) {

	res := &Result{ID: w.ID, Status: "running"}
	w.store.Set(w.ID, res)

	cmd := exec.CommandContext(ctx, "k6", "run",
		"--vus", fmt.Sprint(w.VUs),
		"--duration", w.Dur,
		w.Script,
	)

	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()

	seg := NewSeg()
	agg := NewAgg()

	var logs string
	// create a done channel to know when aggregation finished
	done := make(chan struct{})
	go func() {
		for msg := range agg.ch {
			logs += msg + "\n"
		}
		close(done)
	}()

	// start fan-out
	seg.Start(stdout, stderr)

	// fan-in
	go func() {
		for l := range seg.Out {
			agg.ch <- "[OUT] " + l
		}
	}()
	go func() {
		for l := range seg.Err {
			agg.ch <- "[ERR] " + l
		}
	}()

	cmd.Start()
	err := cmd.Wait()
	close(agg.ch)

	// wait until aggregation goroutine finishes writing to logs
	<-done

	res.Logs = logs
	if err != nil || containsError(logs) {
		res.Status = "FAILED"
	} else {
		res.Status = "SUCCESS"
	}

	// remove temporary script if created from POST body
	if w.Temp {
		_ = os.Remove(w.Script)
	}

	// Persist logs immediately: write to run_id-out.txt (SUCCESS) or run_id-err.txt (FAILED)
	resultsDir := os.Getenv("RESULTS_DIR")
	if resultsDir == "" {
		if exe, err := os.Executable(); err == nil {
			resultsDir = filepath.Join(filepath.Dir(exe), "results")
		} else if wd, err := os.Getwd(); err == nil {
			resultsDir = filepath.Join(wd, "results")
		} else {
			resultsDir = "results"
		}
	}

	os.MkdirAll(resultsDir, 0o755)

	var targetPath string
	if strings.ToUpper(res.Status) == "SUCCESS" {
		targetPath = filepath.Join(resultsDir, fmt.Sprintf("%s-out.txt", w.ID))
	} else {
		targetPath = filepath.Join(resultsDir, fmt.Sprintf("%s-err.txt", w.ID))
	}

	os.WriteFile(targetPath, []byte(logs), 0o644)
}

// -------- UTIL --------
func containsError(s string) bool {
	// use the standard library for reliable substring checks
	if s == "" {
		return false
	}
	if strings.Contains(s, "ERR") {
		return true
	}
	// case-insensitive check for "error"
	return strings.Contains(strings.ToLower(s), "error")
}

// -------- GLOBAL --------
var store = &Store{m: make(map[string]*Result)}

// -------- HANDLERS --------
func run(w http.ResponseWriter, r *http.Request) {

	var scriptPath string
	var temp bool

	if r.Method == http.MethodPost {
		// read script from body
		b, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read body", 500)
			return
		}
		if len(b) == 0 {
			http.Error(w, "empty script body", 400)
			return
		}

		// create temp file for k6 inside the container
		f, err := os.CreateTemp("", "k6script-*.js")
		if err != nil {
			http.Error(w, "failed to create temp script", 500)
			return
		}
		if _, err := f.Write(b); err != nil {
			f.Close()
			http.Error(w, "failed to write script", 500)
			return
		}
		f.Close()
		scriptPath = f.Name()
		temp = true
	} else {
		scriptPath = r.URL.Query().Get("script")
		if scriptPath == "" {
			http.Error(w, "script required", 400)
			return
		}
	}

	id := fmt.Sprintf("run_%d", time.Now().UnixNano())

	tag := r.URL.Query().Get("tag")

	vus := 2
	dur := "5s"
	// special case: if the requested script is the error script, use smaller load
	if strings.Contains(scriptPath, "script2.js") || strings.Contains(scriptPath, "script2") {
		vus = 1
		dur = "2s"
	}

	worker := &Worker{
		ID:     id,
		Script: scriptPath,
		VUs:    vus,
		Dur:    dur,
		store:  store,
		Temp:   temp,
		Tag:    tag,
	}

	go worker.Run(context.Background())

	json.NewEncoder(w).Encode(map[string]string{"id": id})
}

func status(w http.ResponseWriter, r *http.Request) {
	res := store.Get(r.URL.Query().Get("id"))
	if res == nil {
		http.Error(w, "not found", 404)
		return
	}
	json.NewEncoder(w).Encode(res)
}

func health(w http.ResponseWriter, _ *http.Request) {
	w.Write([]byte("ok"))
}

// -------- MAIN --------
func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	debug.SetMemoryLimit(128 << 20)

	http.HandleFunc("/run", run)
	http.HandleFunc("/status", status)
	http.HandleFunc("/health", health)

	fmt.Println("server :8080")
	http.ListenAndServe(":8080", nil)
}
