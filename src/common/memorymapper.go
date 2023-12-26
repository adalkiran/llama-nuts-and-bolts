package common

import (
	"fmt"
	"os"
	"syscall"
)

type MemoryMapper struct {
	FilePath string
	Size     int64

	Data []byte
}

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
	data, err := syscall.Mmap(int(file.Fd()), 0, int(result.Size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("error mapping file into memory: %v", err)
	}
	result.Data = data

	return result, nil
}

func (mm *MemoryMapper) Unmap() error {
	if mm.Data == nil {
		return nil
	}
	err := syscall.Munmap(mm.Data)
	if err != nil {
		return fmt.Errorf("error unmapping file from memory: %v", err)
	}
	return nil
}
