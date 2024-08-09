# **2. LOADING TORCH MODEL**

In this chapter, we'll walk through the process of loading model weights stored in the "consolidated.00.pth" file.

## **2.1. Creating TorchModelReader**

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func LoadModelEx(modelDir string, includeTensors bool, includeVocab bool) (*Model, error) {
    model := &Model{}
    if includeTensors {
        modelFilePath := filepath.Join(modelDir, "consolidated.00.pth")
        torchModelReader, err := torch.NewTorchModelReader(modelFilePath)
        if err != nil {
            return nil, err
        }
        defer torchModelReader.Close()
        ...
    }
    ...
}
```

When we step into the function, we observe the creation of a [common.MemoryMapper](../src/common/memorymapper_unix.go) object for the reader. This allows us to set up mappings between virtual memory addresses and file locations on disk and enables us to represent and access them as ```[]byte``` slices in the code.

The file "consolidated.00.pth" contains various weight tensors. We will determine file offsets corresponding to these tensors and create tensor objects by establishing memory mapping with these offset values.

See more: [Memory Mapping](https://en.wikipedia.org/wiki/Memory-mapped_file).

<sup>from [src/torch/torchmodelreader.go](../src/torch/torchmodelreader.go)</sup>

```go
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
```

## **2.2. Calling TorchModelReader.Load()**

[TorchModelReader.Load\(...\)](../src/torch/torchmodelreader.go) is called if ```includeTensors``` is true.

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func LoadModelEx(modelDir string, includeTensors bool, includeVocab bool) (*Model, error) {
    if includeTensors {
        ...
        common.GLogger.ConsolePrintf("Loading model file: \"%s\"...", modelFilePath)
        modelTensors, err := torchModelReader.Load()
        if err != nil {
            return nil, err
        }
        model.Tensors = modelTensors
        common.GLogger.ConsolePrintf("Found %d tensors in the model.", len(model.Tensors.GetKeys()))
        ...
    }
    ...
}
```

### **2.2.1. Reading Model File as a ZIP File**

The file "consolidated.00.pth" in the specified path is an **uncompressed** ZIP file containing 1 Pickle file and 291 files of weight tensors (valid for Llama 3.1 8B/8B-Instruct models). Utilizing **uncompressed ZIP** instead of **compressed ZIP** allows for direct content reading (a must for memory mapping) without the need for an unzipper stream wrapper. Additionally, utilizing a ZIP file allows the storage and transfer of multiple files as a single file.

If you are curious about what this file contains, you can simply change extension of this file to ".zip" and check out it as a ZIP file.

<sup>from [src/torch/torchmodelreader.go](../src/torch/torchmodelreader.go)</sup>

```go
func (tmr *TorchModelReader) Load() (*pickle.PickleDict[*ml.Tensor], error) {
    var err error
    tmr.inputZipReader, err = zip.OpenReader(tmr.modelFilePath)
    if err != nil {
        return nil, err
    }
    ...
}
```

### **2.2.2. Searching for a .pkl File**

If you check out the contents of the "consolidated.00.pth" file, you will find a structure like this:

```sh
consolidated (parent directory)
  |-byteorder
  |-version
  |-data.pkl
  |-data
  |  |-135
  |  |-61
  |  |-95
  |  |-132
  |  |-59
  |  |-92
  |  |-...
```

The name of .pkl file is not important for us (it is data.pkl), we look for **exactlly one** .pkl file in the ZIP file, and call ```tmr.readPickleFile``` function for this file.

<sup>from [src/torch/torchmodelreader.go](../src/torch/torchmodelreader.go)</sup>

```go
    pklRegexp, _ := regexp.Compile(`\.pkl$`)
    pklFileList := tmr.findFilesInZip(pklRegexp)
    if len(pklFileList) != 1 {
        return nil, fmt.Errorf("no .pkl file found in Torch model file \"%s\"", tmr.modelFilePath)
    }
    modelTensorVals, err := tmr.readPickleFile(pklFileList[0])
    if err != nil {
        return nil, err
    }
```

### **2.2.3. Reading Pickle File - Reader Initialization**

Pickle (.pkl) file is a specific file format of Python platform. Our project's Pickle reader was inspired by [pickle.py in Python repository](https://github.com/python/cpython/blob/main/Lib/pickle.py) code. Historically, Pickle project was an open-source project, independent of Python repository. Then it was integrated into Python standard library. It aims to serialize/deserialize Python objects and dictionaries with data and methods. The simplicity of its usage and elasticity made it a superior and most-used library to achieve these types of tasks.

In the ML world, the Pickle library and its format have become a de facto standard. However, it comes with certain side effects, such as the risk of malicious code execution, because it allows calling functions during the serialization/deserialization process. Alternatives like [Safetensors](https://github.com/huggingface/safetensors) provide a more secure approach to accomplishing these tasks.

The original Llama model weight files were saved via Pytorch in Pickle format. You can find these weights in different file formats on the Internet, but we will continue with the original files distributed by Meta.

Torch model file reader functionality was separated into two packages: [pickle](../src/pickle) and [torch](../src/torch).

* The [pickle](../src/pickle) package contains functions and structures for reading a generic Pickle (.pkl) file. It's important to note that our ported version supports only a limited count of Pickle opcodes necessary for loading specific Llama model.
* The [torch](../src/torch) package contains functions and structures specialized for Pytorch model files. The Pickle format comes with an elastic structure that allows us to extend it with customized functionalities. ```pickle``` package functions will call/use our customized functions/structures defined in ```torch``` if it reads an ```opcode``` that requires to do, in the file.

Our steps:

* Found .pkl file in the ZIP stream is open as ```io.ReadCloser```
* A [pickle.PickleReader](../src/pickle/picklereader.go) is created
* [findClassTorch](../src/torch/torchmodelreader.go) function is pointed as ```pickleReader.FindClassFn```. This function will be used to instantiate supported classes and to point supported functions, referred in the Pickle file, via the defined [TORCH_CLASSES](../src/torch/types.go) map. Further, we will see how a Pickle file refers and lets us instantiate/point Go objects/functions.
* [persistentLoad](../src/torch/torchmodelreader.go) function is pointed as ```pickleReader.PersistentLoadFn```. This function will be used to point out a tensor file in the ZIP file as [torch.TorchStorage](../src/torch/types.go) and to do memory mapping to the specified file location/offset. Returned objects will be used while creating weight tensors.

<sup>from [src/torch/torchmodelreader.go](../src/torch/torchmodelreader.go)</sup>

```go
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
```

### **2.2.4. Reading Pickle File - Reading Opcodes**

Now, we are ready to read the bytes of the Pickle file.

#### **2.2.4.1. Opcodes**

Pickle uses a simple stack-based virtual machine that records the instructions used to reconstruct the object. The file is just a list of serialized opcodes, the first one being expected to be the protocol version and the last one a stop opcode. When the stop opcode is met, the current object on the stack is popped.
>See: [Wikipedia](https://en.wikipedia.org/wiki/Serialization#Python) | [Diving into the Python Pickle format](https://spootnik.org/entries/2014/04/05/diving-into-the-python-pickle-formatt/)

Pickle format consists of pairs of opcode and data bytes. Each pair starts with one byte that stands for an "opcode" and then continues with data bytes varying by each type of opcode.

Each supported Pickle "opcode" should be defined as a byte constant and have a corresponding dispatcher (reader) function in our ```dispatcher``` map.

In our project, our goal is to exclusively read Llama 3.1 8B/8B-Instruct model files. As a result, we have defined and implemented only the necessary opcodes and corresponding functions required for successfully reading this specific model file.

After defining supported opcode byte constants and reader functions, we need to register them into ```dispatcher``` map via ```init()``` function.

Also, we have ```stack []interface{}``` and ```metastack []interface{}``` arrays, also ```memo map[int]interface{}``` map in our ```PickleReader``` to store, collect, and structure already read data chunks.

<sup>from [src/pickle/pickledispatch.go](../src/pickle/pickledispatch.go)</sup>

```go
const (
    ...
    PROTO byte = 0x80 // identify pickle protocol

    EMPTY_DICT      byte = '}'    // push empty dict
    BINPUT          byte = 'q'    //   "     "    "   "   " ;   "    " 1-byte arg
    MARK            byte = '('    // push special markobject on stack
    ...
    BINPERSID       byte = 'Q'    //  "       "         "  ;  "  "   "     "  stack
    ...
    REDUCE          byte = 'R'    // apply callable to argtuple, both on stack
    ...
)

type dispatchFunc = func(*PickleReader) error

var dispatcher = make(map[byte]dispatchFunc)

func init() {
    dispatcher[PROTO] = load_proto
    dispatcher[EMPTY_DICT] = load_empty_dictionary
    dispatcher[BINPUT] = load_binput
    dispatcher[MARK] = load_mark
    ...
    dispatcher[BINPERSID] = load_binpersid
    ...
    dispatcher[REDUCE] = load_reduce
    ...
}
```

#### **2.2.4.2. Reader Functions Corresponding to Opcodes**

To help you understand well, I'll provide some example reader functions/opcodes:

* **load_empty_dictionary (EMPTY_DICT byte = '}')**
    >Creates an empty dictionary (the term "dictionary" exists in the Python world, and we have defined ```PickleDict[T any]```, a custom type of map, to fulfill this custom requirement).
* **load_binput (BINPUT byte = 'q')**
    >Reads a byte ```i```, takes the last element of ```stack``` and assigns it as ```i```-th item of ```memo```.
* **load_mark (MARK byte = '(')**
    >* Appends the ```stack``` array into ```metastack``` array (array of arrays),
    >* Creates new ```stack``` array.
* **load_binpersid (BINPERSID byte = 'Q')**
    >* Pops the last item in the ```stack```, which will be our ```pid``` argument,
    >* Passes this pid (persistent id) argument to the [PickleReader.persistentLoad(...)](../src/pickle/picklereader.go) method,
    >* Calls the custom ```PersistentLoadFn```,
    >* Appends the result into the ```stack```.
    >
    ><br>

    >In our case, ```PersistentLoadFn``` is [TorchModelReader.persistentLoad(...)](../src/torch/torchmodelreader.go) method, pid argument is the name of the file containing tensor weights in the ZIP file.
    >
    >This method:
    >* Finds the file header for specified file in the ZIP file,
    >* Takes the offset (starting point location) of the file,
    >* Creates a [TorchStorage](../src/torch/types.go) object, with memory-mapped ```rawData``` byte array,
    >* And, returns it.
* **load_reduce (REDUCE byte = 'R')**
    >* Pops the last item in the ```stack```, it will be our ```rawArgsArr``` argument array,
    >* Takes the last item in the ```stack``` (this time we don't remove with pop),
    >* Converts ```rawArgsArr``` items to expected data types of reflected function object ```fn```,
    >* Calls the ```fn``` with passing converted arguments, then replaces the last item in the ```stack``` with the function's result.
    >
    ><br>

    >In our case, ```fn``` can be:
    >* [NewPickleDict\[interface{}\]\(...\)](../src/pickle/types.go) corresponds to "collections.OrderedDict" defined in [BASE_CLASSES](../src/pickle/types.go), creates a new PickleDict (keys are ordered)
    >* [rebuild_tensor_v2(...)](../src/torch/types.go) corresponds to "torch._utils._rebuild_tensor_v2" defined in [TORCH_CLASSES](../src/torch/types.go), creates and returns a [ml.Tensor](../src/ml/tensor.go) with specified arguments.

<sup>from [src/pickle/pickledispatch.go](../src/pickle/pickledispatch.go)</sup>

```go
func load_empty_dictionary(pr *PickleReader) error {
    pr.Append(NewPickleDict[interface{}]())
    return nil
}

func load_binput(pr *PickleReader) error {
    i, err := pr.ReadByte()
    if err != nil {
        return err
    }
    pr.memo[int(i)] = pr.stack[len(pr.stack)-1]
    return nil
}

func load_mark(pr *PickleReader) error {
    pr.metastack = append(pr.metastack, pr.stack)
    pr.stack = nil
    pr.stack = make([]interface{}, 0)
    return nil
}

func load_binpersid(pr *PickleReader) error {
    var pid interface{}
    pr.stack, pid = pop(pr.stack)
    result, err := pr.persistentLoad(pid.([]interface{}))
    if err != nil {
        return err
    }
    pr.Append(result)
    return nil
}
```

#### **2.2.4.4. Reading the Whole File**

In view of this succinct information, we can understand following simple flow. To read the Pickle file:

* We initiate a loop,
* Read a key (opcode byte),
* Dispatch this opcode function,
* Execute it,
* Check for errors, and continue if no [StopSignal](../src/torch/types.go) is encountered.
* If we get a [StopSignal](../src/torch/types.go), its value is our result containing content map of the whole file.
* We return this result as ```pickle.PickleDict[interface{}]```.

```pickle.PickleDict[interface{}]``` consists of ```keys``` array and ```items``` map, imitating an [OrderedDict](https://www.geeksforgeeks.org/ordereddict-in-python/) in Python. Keys are stored in sorted array, items (values) are stored in a key-value structure.

>**Note: If you're curious about the details of how the Pickle file structure can be read, please refer to:** [3. LOADING TORCH MODEL \(DETAILS\)](../docs/03-LOADING-TORCH-MODEL-DETAILS.md)

<sup>from [src/pickle/picklereader.go](../src/pickle/picklereader.go)</sup>

```go
func (pr *PickleReader) Load() (*PickleDict[interface{}], error) {
    for {
        key, err := pr.ReadByte()
        if err != nil {
            return nil, err
        }
        err = dispatch(pr, key)
        if err != nil {
            if stopSignal, ok := err.(*StopSignal); ok {
                return stopSignal.Value, nil
            }
            return nil, err
        }
    }
}
```

Result ```pickle.PickleDict[interface{}]``` will be consisted of tensor names and [ml.Tensor](../src/ml/tensor.go) objects, will be seem like:

```go
model.keys:
    [0]: "tok_embeddings.weight"
    [1]: "norm.weight"
    [2]: "output.weight"
    [3]: "layers.0.attention.wq.weight"
    [4]: "layers.0.attention.wk.weight"
    ...

model.items: 
"layers.0.attention.wo.weight": interface {}(*github.com/adalkiran/llama-nuts-and-bolts/src/ml.Tensor {Size: []int len: 2, cap: 2, [4096,4096],...

"layers.7.feed_forward.w2.weight": interface {}(*github.com/adalkiran/llama-nuts-and-bolts/src/ml.Tensor) *{Size: []int len: 2, cap: 2, [4096,14336],...

"layers.31.attention.wq.weight": interface {}(*github.com/adalkiran/llama-nuts-and-bolts/src/ml.Tensor) *{Size: []int len: 2, cap: 2, [4096,4096],...

"layers.1.feed_forward.w1.weight": interface {}(*github.com/adalkiran/llama-nuts-and-bolts/src/ml.Tensor) *{Size: []int len: 2, cap: 2, [14336,4096],...
...
```

### **2.2.5. Reading Pickle File - Returning Model Tensors**

After running [tmr.readPickleFile](../src/torch/torchmodelreader.go), the result ```*pickle.PickleDict[interface{}]``` is converted to ```*pickle.PickleDict[*ml.Tensor]``` object.

<sup>from [src/torch/torchmodelreader.go](../src/torch/torchmodelreader.go)</sup>

```go
func (tmr *TorchModelReader) Load() (*pickle.PickleDict[*ml.Tensor], error) {
    ...
    modelTensorVals, err := tmr.readPickleFile(pklFileList[0])
    if err != nil {
        return nil, err
    }

    modelTensors := pickle.NewPickleDict[*ml.Tensor]()

    for _, key := range modelTensorVals.GetKeys() {
        val, _ := modelTensorVals.Get(key)
        tensorVal := val.(*ml.Tensor)
        tensorVal.Name = key
        modelTensors.Set(key, tensorVal)
    }
    modelTensorVals = nil

    return modelTensors, nil
}
```

 Then, [LoadModelEx](../src/model/loader.go) function takes this result and assigns it to result object's Tensors property: ```model.Tensors```.

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func LoadModelEx(modelDir string, includeTensors bool, includeVocab bool) (*Model, error) {
    model := &Model{}
    if includeTensors {
        ...
        modelTensors, err := torchModelReader.Load()
        if err != nil {
            return nil, err
        }
        model.Tensors = modelTensors
        common.GLogger.ConsolePrintf("Found %d tensors in the model.", len(model.Tensors.GetKeys()))
    }
    ...
    return model, nil
}
```

Now,

* The "data.pkl" file in the ZIP file "consolidated.00.pth" has been read,
* Names and file locations of each tensor have been determined,
* Memory-mapped []byte arrays have been defined for each tensor,
* [Tensor](../src/ml/tensor.go) objects have been created and added into ```model.Tensors```.

And we can see output lines in the console as follows:

```sh
[INFO] ... Loading model file: "/workspace/models-original/Meta-Llama-3.1-8B-Instruct/consolidated.00.pth"...
[INFO] ... Found 291 tensors in the model.
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: INITIALIZATION](./01-INITIALIZATION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: LOADING TORCH MODEL (DETAILS)&nbsp;&nbsp;&gt;](./03-LOADING-TORCH-MODEL-DETAILS.md)

</div>
