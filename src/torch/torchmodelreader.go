package torch

import (
	"archive/zip"
	"fmt"
	"regexp"
	"strings"

	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
)

type TorchModelReader struct {
	modelFilePath  string
	inputZipReader *zip.ReadCloser
	dataBasePath   string
}

func NewTorchModelReader(modelFilePath string) *TorchModelReader {
	result := new(TorchModelReader)
	result.modelFilePath = modelFilePath
	return result
}
func (tmr *TorchModelReader) Load() (*pickle.PickleDict, error) {
	var err error
	tmr.inputZipReader, err = zip.OpenReader(tmr.modelFilePath)
	if err != nil {
		return nil, err
	}
	defer tmr.inputZipReader.Close()

	pklRegexp, _ := regexp.Compile(`\.pkl$`)
	pklFileList := tmr.findFilesInZip(pklRegexp)
	if len(pklFileList) != 1 {
		return nil, fmt.Errorf("no .pkl file found in Torch model file \"%s\"", tmr.modelFilePath)
	}
	model, err := tmr.readPickleFile(pklFileList[0])
	if err != nil {
		return nil, err
	}

	return model, nil
}

func (tmr *TorchModelReader) findFilesInZip(fileNameRegexp *regexp.Regexp) []*zip.File {
	result := make([]*zip.File, 0)
	for _, file := range tmr.inputZipReader.File {
		if fileNameRegexp.MatchString(file.Name) {
			result = append(result, file)
		}
	}
	return result
}

func (tmr *TorchModelReader) readPickleFile(inputPickleFile *zip.File) (*pickle.PickleDict, error) {
	fileReader, err := inputPickleFile.Open()
	if err != nil {
		return nil, err
	}
	defer fileReader.Close()
	tmr.dataBasePath = inputPickleFile.FileHeader.Name[:len(inputPickleFile.FileHeader.Name)-4]
	pickleReader := pickle.NewPickleReader(fileReader)
	pickleReader.FindClassFn = findClassTorch
	pickleReader.PersistentLoadFn = tmr.persistentLoad
	model, err := pickleReader.Load()
	if err != nil {
		return nil, err
	}
	return model, nil
}

func findClassTorch(module string, name string) (interface{}, error) {
	if !strings.HasPrefix(module, "torch") {
		return nil, fmt.Errorf("unknown class \"%s.%s\" not found", module, name)
	}
	result, ok := TORCH_CLASSES[module+"."+name]
	if !ok {
		return nil, fmt.Errorf("unknown class \"%s.%s\" not found", module, name)
	}
	return result, nil
}

func (tmr *TorchModelReader) persistentLoad(pid []interface{}) (interface{}, error) {
	if pid[0] != "storage" {
		return nil, fmt.Errorf("pid[0] must have value \"storage\"")
	}
	kind, ok := pid[1].(StorageKind)
	if !ok {
		return nil, fmt.Errorf("pid[1] must be type of StorageKind")
	}
	filenameStem := pid[2].(string)
	filename := fmt.Sprintf("%s/%s", tmr.dataBasePath, filenameStem)

	foundFile, err := tmr.inputZipReader.Open(filename)
	if err != nil {
		return nil, err
	}
	foundFile.Close()
	dataType := kind.dataType
	description := fmt.Sprintf("storage dataType=%v path-in-zip=%s", dataType, filename)
	return StorageDescriptor{filename, pid[1].(StorageKind), description}, nil
}
