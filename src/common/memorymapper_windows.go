//go:build windows

package common

import (
	"fmt"
	"os"
	"reflect"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

type MemoryMapper struct {
	FilePath      string
	Size          int64
	MappingHandle windows.Handle

	Data []byte
}

// Only Windows version of this memory mapper flow was borrowed from:
// See: https://github.com/spcau/godiff/blob/master/godiff_windows.go

func NewMemoryMapper(filePath string) (*MemoryMapper, error) {
	result := &MemoryMapper{
		FilePath: filePath,
	}
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("error getting file info: %v", err)
	}

	result.Size = fileInfo.Size()

	// Llama.cpp has given 0 as "size" argument of CreateFileMapping, because it takes uint32, but our file size (14GB)
	// exceeds limits of 32 bit integer. So, here, our code takes the address overflow risk on its own.
	// See: https://github.com/ggerganov/llama.cpp/blob/381ee195721d8e747ee31a60c0751822b3072f02/llama.cpp#L1015
	result.MappingHandle, err = windows.CreateFileMapping(windows.Handle(file.Fd()), nil, windows.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return nil, err
	}

	// perform the file map operation

	// Llama.cpp has given 0 as "size" argument of MapViewOfFile, because it takes uint32, but our file size (14GB)
	// exceeds limits of 32 bit integer. So, here, our code takes the address overflow risk on its own.
	// See: https://github.com/ggerganov/llama.cpp/blob/381ee195721d8e747ee31a60c0751822b3072f02/llama.cpp#L1022
	addr, err := windows.MapViewOfFile(result.MappingHandle, syscall.FILE_MAP_READ, 0, 0, uintptr(0))
	if err != nil {
		return nil, err
	}

	sl := reflect.SliceHeader{Data: addr, Len: int(result.Size), Cap: int(result.Size)}
	result.Data = *(*[]byte)(unsafe.Pointer(&sl))

	return result, nil
}

func (mm *MemoryMapper) Unmap() error {
	// TODO: Check this part again, it gives this error while trying to unmap: "runtime error: invalid memory address or nil pointer dereference"
	/*
		addr := uintptr(unsafe.Pointer(&mm.Data[0]))
		err := syscall.UnmapViewOfFile(addr)

		if err == nil {
			err = windows.CloseHandle(mm.MappingHandle)
		}
	*/
	return nil
}
