package torch

import (
	"archive/zip"
	"fmt"
	"regexp"
	"strings"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
)

type TorchModelReader struct {
	modelFilePath  string
	inputZipReader *zip.ReadCloser
	dataBasePath   string

	memoryMapper *common.MemoryMapper
}

func NewTorchModelReader(modelFilePath string) (*TorchModelReader, error) {
	memoryMapper, err := common.NewMemoryMapper(modelFilePath)
	if err != nil {
		return nil, err
	}

	result := &TorchModelReader{
		modelFilePath: modelFilePath,
		memoryMapper:  memoryMapper,
	}
	return result, nil
}

func (tmr *TorchModelReader) Close() error {
	return tmr.inputZipReader.Close()
}

func (tmr *TorchModelReader) Load() (*pickle.PickleDict[*Tensor], error) {
	var err error
	tmr.inputZipReader, err = zip.OpenReader(tmr.modelFilePath)
	if err != nil {
		return nil, err
	}

	pklRegexp, _ := regexp.Compile(`\.pkl$`)
	pklFileList := tmr.findFilesInZip(pklRegexp)
	if len(pklFileList) != 1 {
		return nil, fmt.Errorf("no .pkl file found in Torch model file \"%s\"", tmr.modelFilePath)
	}
	modelTensorVals, err := tmr.readPickleFile(pklFileList[0])
	if err != nil {
		return nil, err
	}

	modelTensors := pickle.NewPickleDict[*Tensor]()

	for _, key := range modelTensorVals.GetKeys() {
		val, _ := modelTensorVals.Get(key)
		modelTensors.Set(key, val.(*Tensor))
	}
	modelTensorVals = nil

	return modelTensors, nil
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

func (tmr *TorchModelReader) findFileInZip(filename string) *zip.File {
	for _, file := range tmr.inputZipReader.File {
		if filename == file.Name {
			return file
		}
	}
	return nil
}

func (tmr *TorchModelReader) readPickleFile(inputPickleFile *zip.File) (*pickle.PickleDict[interface{}], error) {
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

	elmCount, err := common.InterfaceToInt(pid[4])
	if err != nil {
		return nil, err
	}
	contentFile := tmr.findFileInZip(filename)
	if contentFile == nil {
		return nil, fmt.Errorf("file \"%s\" not found in Torch model file \"%s\"", filename, tmr.modelFilePath)
	}
	storageOffset, err := contentFile.DataOffset()
	if err != nil {
		return nil, err
	}
	dataType := kind.dataType
	description := fmt.Sprintf("storage dataType=%v path-in-zip=%s", dataType, filename)

	storage := TorchStorage{filename, pid[1].(StorageKind), storageOffset, description, nil}
	storage.Load(tmr.memoryMapper, elmCount)
	return storage, nil
}
