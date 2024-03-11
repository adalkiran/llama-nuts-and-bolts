# **4. LOADING MODEL ARGS**

[loadModelArgs](../src/model/loader.go) is called if ```includeTensors``` is true.

In this phase, the model configuration file "params.json" is parsed as [ModelArgs](../src/model/modelargs.go) object via JSON parser.

 Then, [loadModelArgs](../src/model/loader.go) function takes its result and assigns it to result object's ModelArgs property: ```model.ModelArgs```.

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func LoadModelEx(modelDir string, includeTensors bool, includeVocab bool) (*Model, error) {
    if includeTensors {
        ...
        err = loadModelArgs(modelDir, model)
        if err != nil {
            return nil, err
        }
    }
    ...
}
```

Now, we have ```model.ModelArgs``` object with similar values:

```yaml
Dim: 4096
N_Layers: 32
N_Heads: 32
N_KVHeads: -1 // if -1, it will be equal to N_Heads
VocabSize: -1 // if -1, it will be size of tokenizer.model
MultipleOf: 256
FFNDimMultiplier: -1 // will be calculated further
NormEpsilon: 0.000001
MaxSequenceLength: 4096
N_Rep: 0 // will be calculated further
HeadDim: 0 // will be calculated further
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: LOADING TORCH MODEL \(DETAILS\)](./03-LOADING-TORCH-MODEL-DETAILS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: LOADING TOKENIZER MODEL&nbsp;&nbsp;&gt;](./05-LOADING-TOKENIZER-MODEL.md)

</div>
