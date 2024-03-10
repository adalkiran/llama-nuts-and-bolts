package common

import (
	"fmt"
	"golang.org/x/sys/windows"
	"os"
)

var (
	kernel32DLL        = windows.NewLazySystemDLL("kernel32.dll")
	getConsoleOutputCP = kernel32DLL.NewProc("GetConsoleOutputCP")
	setConsoleOutputCP = kernel32DLL.NewProc("SetConsoleOutputCP")
)

func init() {
	// See: https://github.com/fatih/color/blob/main/color_windows.go
	// Opt-in for ansi color support for current process.
	// See: https://learn.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences#output-sequences
	var outMode uint32
	out := windows.Handle(os.Stdout.Fd())
	if err := windows.GetConsoleMode(out, &outMode); err != nil {
		return
	}
	outMode |= windows.ENABLE_PROCESSED_OUTPUT | windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING
	_ = windows.SetConsoleMode(out, outMode)

	codePage := uint(65001)
	ret, _, err := setConsoleOutputCP.Call(uintptr(codePage))
	if ret == 0 {
		fmt.Printf("Couldn't set console output codepage: %v", err)
	}
	ret, _, err = getConsoleOutputCP.Call()
	if ret == 0 {
		fmt.Printf("Couldn't get console output codepage: %v", err)
	} else {
		fmt.Printf("Console codepage was set to: %d\n\n", ret)
	}
}
