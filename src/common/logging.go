package common

import (
	"io"
	"log"
	"os"
)

var GLogger *Logger

type Logger struct {
	console *log.Logger
	debug   *log.Logger
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
		log.New(debugWriter, "[DEBUG]", log.Flags())
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
		l.debug.Printf(format, v...)
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
