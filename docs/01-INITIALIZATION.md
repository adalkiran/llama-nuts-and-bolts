# **1. INITIALIZATION**

The entrypoint of the application is the "main" function in [cmd/main.go](../cmd/main.go).

## **1.1. Defining Application State**

At first sight, the console application doesn't need to be complicated functionality, just calls the inference engine's functions and prints the results on the screen. But in our case, we want to update the screen with various information, not only the output text.

Also, sometimes (you'll see reading further) one output might not be generated at one time, the hard one, emojis are come as UTF-8 byte-by-byte, 1 byte or a few bytes per token, then the result will be combined as an emoji. So, we need to have a enough design that supports combine and orchestrate these operations.

We start with defining our application state in one struct. We store such state variables in this variable.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
func main() {
    ...
    appState = NewAppState()
    ...
}
```

## **1.2. Determining Machine Endianness**

As you know, computers store data, including numbers in binary bits. It's valid with all types of data, but in our case, numbers are stored in 1-byte (8-bit), 2-bytes (16-bit), 4-bytes (32-bit), 8-bytes (64-bit), etc... When your number size is two or more bytes, machines have two ways for grouping them: **big-endian** or **little-endian**, and we call this preference as **endianness** or **byte order**.

Endianness is an important factor for performance. Because of this, this preference may change along microprocessor or device architectures. Recent Intel x86, x86-64, also Apple Silicon (M1/2/3 ARM64) architecture CPUs are **little-endian**, if you didn't use **byte-swapping**. For e.g. networking protocols mostly use **big-endian**.

In normal cases, the ideal is developing a software that is independent from and doesn't care machine's native endianness. 

But in our case, our project uses [unsafe package](https://pkg.go.dev/unsafe), which allows us to make memory operations without type-safety, with the risk of trying to access not allowed addresses (finishes with segmentation fault errors).

Because of we will do too many to be counted mathematical operations and reads/writes to memory, we should intentionally avoid from some overheads of safe environment of Go platform when needed.

It's not important that which endianness is used in machine for our variables in RAM. But we will use the Pytorch model file as memory mapped, and it is **little-endian**. At first, I had done it with [binary.LittleEndian](https://pkg.go.dev/encoding/binary) of Go. But Go compiler didn't use it as [inline function](https://www.geeksforgeeks.org/inline-functions-cpp/) (Go compiler itself decides inlining conditions, there isn't any directive to say compiler to use it as inline, you can check if a function is called as inline or not, via go pprof tool), which hurts performance for millions of calls.

I've tried to swap between native endian and other endian, but it hurts performance significantly.

And... Long story short, I gave up, and just added a check to support only **little-endian** systems :smiley:

See more:

* [Endianness (Wikipedia)](https://en.wikipedia.org/wiki/Endianness)
* [What is Endianness?](https://www.freecodecamp.org/news/what-is-endianness-big-endian-vs-little-endian/)
* [Unsafe Pointers in Go, Should I Ever I Bothered About It](https://blog.devgenius.io/unsafe-pointers-in-go-should-i-ever-i-bothered-about-it-9d1d9db1a97c)
* [Why Compiler Function Inlining Matters](https://www.polarsignals.com/blog/posts/2021/12/15/why-compiler-function-inlining-matters)
* [Debugging Go Code: Using pprof and trace to Diagnose and Fix Performance Issues](https://www.infoq.com/articles/debugging-go-programs-pprof-trace/)

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
    machineEndian := common.DetermineMachineEndian()
    common.GLogger.ConsolePrintf("Determined machine endianness: %s", machineEndian)
    if machineEndian != "LITTLE_ENDIAN" {
        common.FriendlyPanic(fmt.Errorf("error: Endianness of your machine is not supported. Expected LITTLE_ENDIAN but got %s", machineEndian))
    }
```


<br>

## **1.3. Searching for Model File Path**

Actually, in a real world console application should contain an usable command-line interface (CLI) that supports configuration files and command-line arguments/flags. Model file path should be one of these arguments. But, even so, our project aims to focus on "demonstrating how a model works" instead of "being a good usable application", these type of features intentionally weren't added.

Besides, we need to support searching for a model path near the executable path. Because we have a CLI application and unit tests. Unit tests may be executed with different "working directories" like "src/", debugging may start with different working directory like "cmd/", or compiled executable may exist with the model directory in same parent directory. This part looks around the working directory for model files.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
    modelDir, err := searchForModelPath(modelsDirName, "Meta-Llama-3.1-8B-Instruct")
    if err != nil {
        common.FriendlyPanic(err)
    }
```

## **1.4. Creating Go Context**

[Context](https://pkg.go.dev/context) variables are used to manage deadlines, cancellation signals, etc... especially in parallel running goroutines.

Here, we create a context with a cancel function, so we will be able to send a "cancellation signal" to parallelly long-running (including go channel listener infinite loops) independent goroutines, and we will have a chance for "finishing gracefully".

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
```

## **1.5. Loading the Model**

Finally, we can start to load our Llama 3.1 8B-Instruct model and tokenizer model files. For further details, continue reading [02. LOADING TORCH MODEL](./02-LOADING-TORCH-MODEL.md).

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
    llamaModel, err := model.LoadModel(modelDir)
    if err != nil {
        common.GLogger.ConsoleFatal(err)
    }
    defer llamaModel.Free()

    fmt.Printf("Model \"%s\" was loaded.\n", modelDir)
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: THE JOURNEY](./00-THE-JOURNEY.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: LOADING TORCH MODEL&nbsp;&nbsp;&gt;](./02-LOADING-TORCH-MODEL.md)

</div>
