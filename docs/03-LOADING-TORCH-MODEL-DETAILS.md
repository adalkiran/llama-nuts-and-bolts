# **3. LOADING TORCH MODEL (DETAILS)**

In this chapter, we'll walk through the process of reading a tensor stored in the "consolidated.00.pth" file, step by step.

>**<u>A Quick Reminder:</u>**<br>
>Pickle uses a simple stack-based virtual machine that records the instructions used to reconstruct the object. The file is just a list of serialized opcodes, the first one being expected to be the protocol version and the last one a stop opcode. When the stop opcode is met, the current object on the stack is popped.<br>
>See: [Wikipedia](https://en.wikipedia.org/wiki/Serialization#Python) | [Diving into the Python Pickle format](https://spootnik.org/entries/2014/04/05/diving-into-the-python-pickle-formatt/)
>
>Pickle format consists of pairs of opcode and data bytes. Each pair starts with one byte that stands for an "opcode" and then continues with data bytes varying by each type of opcode.

## **3.1. Loading PROTO**

>```PROTO``` stands for Pickle Protocol version number. We start to reading with it.

* Read key byte: 0x80, corresponding opcode: PROTO, function: load_proto
    * Read one byte: 0x02, identifies Pickle Protocol v2.

```yaml
pr.stack: {}
pr.metastack: {}
memo: {}
```

## **3.2. Preparation to read a Tensor**

>The file continues with data of the first weights tensor. Because of the Pickle format is generic, the file contains some instructions to construct some data structures.

* Read key byte: 0x7D, char: '}', corresponding opcode: EMPTY_DICT, function: load_empty_dictionary
    * Call ```NewPickleDict[interface{}]()``` to create a new empty pickle.PickleDict[interface {}] object
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    PickleDict{}
}
pr.metastack: {}
memo: {}
```

* Read key byte: 0x71, char: 'q', corresponding opcode: BINPUT, function: load_binput
    * Read one byte: 0x00 as memo map key
    * Take last element of ```pr.stack``` and assigns it as 0th item of ```pr.memo```

```yaml
pr.stack: {
    PickleDict{}
}
pr.metastack: {}
memo: {
    0: PickleDict{}
}
```

* Read key byte: 0x28, char: '(', corresponding opcode: MARK, function: load_mark
    * Append the ```pr.stack``` array into ```metastack``` array (array of arrays),
    * Create new ```pr.stack``` array.

```yaml
pr.stack: {}
pr.metastack: {
    PickleDict{}
}
memo: {
    0: PickleDict{}
}
```

## **3.3. Reading Name of the Tensor**

>The file continues with the name of the first tensor.

* Read key byte: 0x58, char: 'X', corresponding opcode: BINUNICODE, function: load_binunicode
    * Read 4 bytes: [0x15, 0x00, 0x00, 0x00], convert it to int32 as little-endian: 21 (decimal). Identifies length of unicode string.
    * Read 21 bytes, convert it to string identifies the name of upcoming tensor: "tok_embeddings.weight".
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    "tok_embeddings.weight"
}
pr.metastack: {
    {
        PickleDict{}
    }
}
memo: {
    0: PickleDict{}
}
```

* Read key byte: 0x71, char: 'q', corresponding opcode: BINPUT, function: load_binput
    * Read one byte: 0x01 as memo map key
    * Take last element of ```pr.stack``` and assigns it as 1st item of ```pr.memo```

```yaml
pr.stack: {
    "tok_embeddings.weight"
}
pr.metastack: {
    {
        PickleDict{}
    }
}
memo: {
    0: PickleDict{}
    1: "tok_embeddings.weight"
}
```

## **3.4. Obtaining a TorchStorage Object**

>The file continues with the ```torch.rebuild_tensor_v2(...)``` function. Then, we will read some instructions that construct ```torch.TorchStorage``` object. This object will be one of arguments of for ```torch.rebuild_tensor_v2(...)``` function.

* Read key byte: 0x63, char: 'c', corresponding opcode: GLOBAL, function: load_global
    * Read one line string (until '\n' byte): "torch._utils", identifies Python module name
    * Read one line string (until '\n' byte): "_rebuild_tensor_v2", identifies Python class/function name
    * Call pr.findClass to get corresponding Go object to the module and name: ```torch.rebuild_tensor_v2 function```.
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    "tok_embeddings.weight",
    torch.rebuild_tensor_v2(...)
}
pr.metastack: {
    {
        PickleDict{}
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight"
}
```

* Read key byte: 0x71, char: 'q', corresponding opcode: BINPUT, function: load_binput
    * Read one byte: 0x02 as memo map key
    * Take last element of ```pr.stack``` and assigns it as 2nd item of ```pr.memo```

```yaml
pr.stack: {
    "tok_embeddings.weight",
    torch.rebuild_tensor_v2(...)
}
pr.metastack: {
    {
        PickleDict{}
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...)
}
```

* Read key byte: 0x28, char: '(', corresponding opcode: MARK, function: load_mark
    * Append the ```pr.stack``` array into ```metastack``` array (array of arrays),
    * Create new ```pr.stack``` array.

```yaml
pr.stack: {}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...)
}
```

* Read key byte: 0x28, char: '(', corresponding opcode: MARK, function: load_mark
    * Append the ```pr.stack``` array into ```metastack``` array (array of arrays),
    * Create new ```pr.stack``` array.

```yaml
pr.stack: {}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    },
    {}
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...)
}
```

* Read key byte: 0x58, char: 'X', corresponding opcode: BINUNICODE, function: load_binunicode
    * Read 4 bytes: [0x07, 0x00, 0x00, 0x00], convert it to int32 as little-endian: 7 (decimal). Identifies length of unicode string.
    * Read 7 bytes, convert it to string identifies the name of upcoming tensor: "storage".
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    "storage"
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    },
    {}
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...)
}
```

..... Some steps were taken

* Current state:

```yaml
pr.stack: {
    "storage",
    StorageKind{ml.DT_BF16},
    "0",
    "cpu",
    131072000
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    },
    {}
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu"
}
```

* Read key byte: 0x74, char: 't', corresponding opcode: TUPLE, function: load_tuple
    * Call pop_mark function to build tuple from topmost stack items
        * Backup current ```pr.stack```
        * Pop last item from ```pr.metastack```, result is an array
        * Assign the result to ```pr.stack```
        * Return backed up stack array
    * Push the returned stack array into the ```pr.stack``` as an array

```yaml

pr.stack: {
    {
        "storage",
        StorageKind{ml.DT_BF16},
        "0",
        "cpu",
        131072000
    }
}
pr.metastack: {
    {
        PickleDict{}
    }
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu"
}
```

* Read key byte: 0x71, char: 'q', corresponding opcode: BINPUT, function: load_binput
    * Read one byte: 0x07 as memo map key
    * Take last element of ```pr.stack``` and assigns it as 7th item of ```pr.memo```

```yaml
pr.stack: {
    {
        "storage",
        StorageKind{ml.DT_BF16},
        "0",
        "cpu",
        131072000
    }
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        }
}
```

* Read key byte: 0x51, char: 'Q', corresponding opcode: BINPERSID, function: load_binpersid
    * Pop last item from ```pr.stack```, result is an array: ```{"storage", StorageKind{ml.DT_BF16}, "0", "cpu", 131072000}```. Identifies "pid" argument for ```pr.persistentLoad(...)``` function
    * ```pr.persistentLoad(...)``` function calls ```pr.PersistentLoadFn(...)``` custom function with "pid" array argument
    * ```TorchModelReader.persistentLoad(...)``` function is called with "pid" array argument
        * This function parses the pid array ```{"storage", StorageKind{ml.DT_BF16}, "0", "cpu", 131072000}```
            * pid[0] = "storage", it must be
            * pid[1] = StorageKind{ml.DT_BF16}, data type kind of defined storage
            * pid[2] = "0", filenameStem, filename is defined as: "consolidated/data/0"
            * pid[3] = "cpu", identifies the tensor device, we don't use this data
            * pid[4] = 131072000, identifies element count of the tensor contained by "consolidated/data/0" file
        * Find "consolidated/data/0" file entry in the ZIP file, get its storage offset, 34304 (starting location of the tensor bytes)
        * Create a TorchStorage object with given data type and storage offset
        * Calculate byte locations (starting location, end location) with given storage offfset and given element count
        * Do memory-mapping between ```TorchStorage.rawData``` and bytes in calculated locations. Now we have a memory-mapped to the file ```[]byte``` array for bytes of the current tensor
        * Return the TorchStorage object
    * Push the TorchStorage object into the ```pr.stack```

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    }
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        }
}
```

## **3.5. Continuing File Reading to Gather torch.rebuild_tensor_v2(...) Function Arguments**

>Now, we have ```torch.TorchStorage``` object. The file continues with other arguments of the ```torch.rebuild_tensor_v2(...)``` function. We will read some instructions that construct other arguments. We will call this function with these gathered arguments further.

* Read key byte: 0x4b, char: 'K', corresponding opcode: BININT1, function: load_binint1
    * Push 1-byte unsigned int
    * Read one byte: 0x00
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    }
    0
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        }
}
```

* Read key byte: 0x4d, char: 'M', corresponding opcode: BININT2, function: load_binint2
    * Push 2-byte unsigned int
    * Read 2 bytes: [0x00, 0x7D], convert it to uint16 as little-endian: 32000 (decimal).
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    },
    0,
    32000
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        }
}
```

* Read key byte: 0x4d, char: 'M', corresponding opcode: BININT2, function: load_binint2
    * Push 2-byte unsigned int
    * Read 2 bytes: [0x00, 0x10], convert it to uint16 as little-endian: 4096 (decimal).
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    },
    0,
    32000,
    4096,
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        }
}
```

..... Some steps were taken

* Current state:

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    },
    0,
    {32000, 4096},
    {4096, 1},
    false,
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        }
}
```

* Read key byte: 0x63, char: 'c', corresponding opcode: GLOBAL, function: load_global
    * Read one line string (until '\n' byte): "collections", identifies Python module name
    * Read one line string (until '\n' byte): "OrderedDict", identifies Python class/function name
    * Call pr.findClass to get corresponding Go object to the module and name: ```pickle.NewPickleDict[interface {}]() function```.
    * Push it into the ```pr.stack```

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    },
    0,
    {32000, 4096},
    {4096, 1},
    false,
    pickle.NewPickleDict[interface {}]()
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        }
}
```

..... Some steps were taken

* Current state:

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    },
    0,
    {32000, 4096},
    {4096, 1},
    false,
    pickle.NewPickleDict[interface {}](),
    {}
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        },
    8: {32000, 4096},
    9: {4096, 1},
    10: pickle.NewPickleDict[interface {}]()
}
```

* Read key byte: 0x52, char: 'R', corresponding opcode: REDUCE, function: load_reduce
    * Apply callable to argtuple, both on stack
    * Pop last item from ```pr.stack```, result is an empty array, identifies rawArgsArr array will be passed to upcoming function
    * Take the last item in the ```pr.stack``` (this time we don't remove with pop)
    * Taken item is our function: ```pickle.NewPickleDict[interface {}]()```
    * Convert ```rawArgsArr``` items to expected data types of reflected function object (an empty array)
    * Call the ```pickle.NewPickleDict[interface {}]()``` with passing converted arguments, then replace the last item in the ```pr.stack``` with the function's result: an empty ```pickle.PickleDict[interface {}]{}```

```yaml
pr.stack: {
    torch.TorchStorage {
        filename: "consolidated/data/0",
        kind: torch.StorageKind{dataType: ml.DT_BF16},
        storageOffset: 34304
    },
    0,
    {32000, 4096},
    {4096, 1},
    false,
    PickleDict{}
}
pr.metastack: {
    {
        PickleDict{}
    },
    {
        "tok_embeddings.weight",
        torch.rebuild_tensor_v2(...)
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        },
    8: {32000, 4096},
    9: {4096, 1},
    10: pickle.NewPickleDict[interface {}]()
}
```

..... Some steps were taken

* Current state:

```yaml
pr.stack: {
    "tok_embeddings.weight",
    torch.rebuild_tensor_v2(...),
    {
        torch.TorchStorage {
            filename: "consolidated/data/0",
            kind: torch.StorageKind{dataType: ml.DT_BF16},
            storageOffset: 34304
        },
        0,
        {32000, 4096},
        {4096, 1},
        false,
        PickleDict{}
    }
}
pr.metastack: {
    {
        PickleDict{}
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        },
    8: {32000, 4096},
    9: {4096, 1},
    10: pickle.NewPickleDict[interface {}]()
}
```

## **3.6. Calling torch.rebuild_tensor_v2(...) Function and Obtaining First Tensor**

> Now, we have all required arguments to call ```torch.rebuild_tensor_v2(...)``` function in our ```pr.stack```.

* Read key byte: 0x52, char: 'R', corresponding opcode: REDUCE, function: load_reduce
    * Apply callable to argtuple, both on stack
    * Pop last item from ```pr.stack```, result is an array array, identifies rawArgsArr array will be passed to upcoming function, result array: ```{torch.TorchStorage {...}, 0, {32000, 4096}, {4096, 1}, false, PickleDict{}}```
    * Take the last item in the ```pr.stack``` (this time we don't remove with pop)
    * Taken item is our function: ```torch.rebuild_tensor_v2(...)```
    * Convert ```rawArgsArr``` items to expected data types of reflected function object
    * Call the ```torch.rebuild_tensor_v2(...)``` with passing converted arguments, then replace the last item in the ```pr.stack``` with the function's result: a [ml.Tensor](../src/ml/tensor.go) object with Size=```[32000,4096]```, Stride=```[4096,1]```, DataType=```ml.DT_BF16```, RawData=```(memory mapped []byte)```

```yaml
pr.stack: {
    "tok_embeddings.weight"
    ml.Tensor{Size=[32000,4096], ...}
}
pr.metastack: {
    {
        PickleDict{}
    }
}
memo: {
    0: PickleDict{},
    1: "tok_embeddings.weight",
    2: torch.rebuild_tensor_v2(...),
    3: "storage",
    4: StorageKind{ml.DT_BF16},
    5: "0",
    6: "cpu",
    7:  {
            "storage",
            StorageKind{ml.DT_BF16},
            "0",
            "cpu",
            131072000
        },
    8: {32000, 4096},
    9: {4096, 1},
    10: pickle.NewPickleDict[interface {}](),
    11: PickleDict{}
}
```

..... Now,<br>

* The first item of our ```pr.stack``` is ```"tok_embeddings.weight"``` (the name of the tensor)
* The second item of our  ```pr.stack``` is ```ml.Tensor{Size=[32000,4096], ...}``` (the tensor itself)
* In the next steps, this flow will be recurred for other tensors.

## **3.7. Completion of Reading All Tensors**

>Now, we have all of 292 tensors with names in our ```pr.stack```. A healthy Pickle file ends with a [StopSignal](../src/pickle/types.go) instruction.

..... Some steps were taken

* Current state:

```yaml
pr.stack: {
    PickleDict {
        a PickleDict (map) of 292 tensor names and corresponding ml.Tensor objects
    }
}
pr.metastack: {}
memo: {an array of 2342 items}
```

* Read key byte: 0x2e, char: '.', corresponding opcode: STOP, function: load_stop
    * Every pickle ends with STOP
    * Pop last item from ```pr.stack```, result is a ```PickleDict``` (map) of tensor names and corresponding [ml.Tensor](../src/ml/tensor.go) objects
    * Create a [StopSignal](../src/pickle/types.go) signal object with ```Value``` of the result ```PickleDict``` (map) and return it

* Finished.

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: LOADING TORCH MODEL](./02-LOADING-TORCH-MODEL.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: LOADING MODEL ARGS&nbsp;&nbsp;&gt;](./04-LOADING-MODEL-ARGS.md)

</div>
