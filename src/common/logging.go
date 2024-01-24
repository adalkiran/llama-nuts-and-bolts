package common

import (
	"fmt"
	"io"
	"log"
	"os"
	"time"
)

var GLogger *Logger

type Logger struct {
	console *log.Logger
	debug   *log.Logger

	debugStartTime time.Time
}

func NewLogger(consoleWriter io.Writer, debugWriter io.Writer) (*Logger, error) {
	if consoleWriter == nil {
		consoleWriter = os.Stdout
	}

	result := &Logger{
		console: log.New(consoleWriter, "[INFO]", log.Flags()),
		debug:   nil,
	}
	if debugWriter != nil {
		result.debug = log.New(debugWriter, "[DEBUG]", log.Lmicroseconds)
	}
	return result, nil
}

func (l *Logger) ConsolePrintf(format string, v ...any) {
	if l.console != nil {
		l.console.Printf(format, v...)
	}
}

func (l *Logger) ConsoleFatal(v ...any) {
	if l.console != nil {
		l.console.Fatal(v...)
	}
}

func (l *Logger) DebugPrintf(format string, v ...any) {
	if l.debug != nil {
		if !l.debugStartTime.IsZero() {
			format += fmt.Sprintf(" (%.4f secs)", time.Since(l.debugStartTime).Seconds())
		}
		l.debug.Printf(format, v...)
		l.debugStartTime = time.Now()
	}
}

func (l *Logger) Close() {
	if l.console != nil {
		f, ok := l.console.Writer().(*os.File)
		if ok {
			f.Close()
		}
	}
	if l.debug != nil {
		f, ok := l.debug.Writer().(*os.File)
		if ok {
			f.Close()
		}
	}
}
