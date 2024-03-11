# **0. THE JOURNEY**

I had been considering implementing a project that runs inference with an open-source large language model (LLM) using a language/platform other than Python and without using external dependencies or libraries. This approach would enable me to dig into the internals of a transformer model and encounter details that I had not been aware of before. These details might include things I already knew theoretically, new things to learn, or new insights to gain.

After having a chance to work with some proprietary or open-source LLMs on professional projects, this thought became stronger. Then, I have had enough free time to fulfill one of the items on my wishlist: **LLaMA Nuts and Bolts!**

## **0.1. The Motivation Behind**

As a curious, passionate computer engineer and software developer, I love to learn new things or dig into the internals of some technology, even if it means *reinventing the wheel*.

I developed another project with the same passionate motivation and experimental approach about WebRTC technologies 1.5 years ago, [WebRTC Nuts and Bolts](https://github.com/adalkiran/webrtc-nuts-and-bolts), you can check out it if you haven't seen before.

Some of you may think: *"It was not necessary to have"*, *"It's just reinventing the wheel"*, or *"We have lots of existing tools in Python and C ecosystem"*.

Most of these are correct for production and commercial usage. If your goal is only to use (or finetune) an LLM for production, you must go on with existing, most used Python libraries or for instance, [Georgi Gerganov](https://github.com/ggerganov)'s [llama.cpp](https://github.com/ggerganov/llama.cpp), because they are robust, community-supported and battle-tested projects.

If you aim to understand what exists behind the scenes while using them and working with LLMs, you are in the right place,
let's take a step back in time with me as while I'm revisiting my journey to document the experiences!

## **0.2. The Expectation**

Everything starts with a deep curiosity. There are numerous useful articles and videos explaining LLMs, transformers, and NLP concepts. However, as expected, they primarily concentrate on one specific part of the comprehensive system.

On the other hand, my style of learning leans on the deductive method. In contrast to the conventional approach, I don't start by learning atomic pieces and concepts first. Instead, I prefer going linearly from beginning to the end, and learning an atomic piece at the time when learning this piece is required.

The objective of this project was to give me hands-on experience with the fundamentals of machine learning, transformer models, attention mechanism, rotary positional embeddings (RoPE), tradeoffs of alternatives, and mathematics behind them by stepping outside of comfort zone of Python ecosystem.

## **0.3. The Objective**

The main objective is to develop a console application from scratch, without using any existing ML or mathematics libraries, which generates meaningful text outputs using pretrained LLaMa 2 7B-Chat model weights.

To achieve this, we need:

* To download the LLaMa 2 7B-Chat model from [Meta LLaMa website](https://llama.meta.com/llama-downloads/). We will need files *consolidated.00.pth*, *params.json* and *tokenizer.model* files.

* To debug [original LLaMa 2 Python repository of Meta](https://github.com/facebookresearch/llama/blob/main/llama/model.py) and [Georgi Gerganov](https://github.com/ggerganov)'s [llama.cpp](https://github.com/ggerganov/llama.cpp) projects running in local machine.
    >I had to add configuration to run Pytorch with [memory mapping](https://en.wikipedia.org/wiki/Memory-mapped_file) support. [llama.cpp](https://github.com/ggerganov/llama.cpp) has builtin memory mapping support.

* To implement [Pickle (.pkl)](https://github.com/python/cpython/blob/main/Lib/pickle.py) file reader (unpickler). We will use it to read Pytorch model checkpoint and weight tensors from *consolidated.00.pth*. This file is an uncompressed ZIP file containing 292 weight tensors. [Pickle](https://github.com/python/cpython/blob/main/Lib/pickle.py) format is a Python specific file format to store/load Python objects.
    >**Note that**, in this project, we only need to read Pickle file, not write, and we have implemented only required opcodes that used in the model file. Also only Pytorch model tensor storage stream is supported.

    >Implementation can be found in [src/pickle](../src/pickle) directory.

* To implement [Memory Mapping](https://en.wikipedia.org/wiki/Memory-mapped_file) that allows us to map a large file content to memory address space. Thanks to this, we don't have to load all ~13GB of data into the physical RAM, at the same time we can access a specified part of the file in the disk as we do with byte arrays in the RAM. This operation is completely managed by the operating system.
    >Implementation can be found in [src/common/memorymapper_unix.go](../src/common/memorymapper_unix.go) and [src/common/memorymapper_windows.go](../src/common/memorymapper_windows.go).

* To implement custom storage functions for Pytorch, which will be called by Pickle reader.
    >Implementation can be found in [src/torch](../src/torch) directory.

* To implement [Protocol Buffers (Protobuf)](https://protobuf.dev/) reader inspired by [protobuf-go](https://github.com/protocolbuffers/protobuf-go/blob/e8baad6b6c9e2bb1c48e4963866248d4f35d4fd7/encoding/protowire/wire.go). Because, LLamA 2's tokenizer file *tokenizer.model* stores a [SentencePiece (SPM)](https://github.com/google/sentencepiece) tokenizer model in Protobuf message format.
    >Implementation can be found in [src/protobuf](../src/protobuf) directory.

* To implement custom types and functions for [SentencePiece (SPM)](https://github.com/google/sentencepiece), which is represented as [this Protobuf structure](https://github.com/google/sentencepiece/blob/022f8c3fed4d2feb4e4c670949cf01cef477dcc4/src/sentencepiece_model.proto).
    >Implementation can be found in [src/sentencepiece](../src/sentencepiece) directory.

* To implement [BFloat16 (Brain Floating Point)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) data type. The Go language doesn't have neither BFloat16 nor float16 (2-byte floating point) primitive type. Because of this, we need to implement it ourselves. Even it had support for 16-bit floating point (float16), bfloat16 has some differences from standard float16.<br>
LLaMa 2 Pytorch model file contains tensors stored as 2-byte BFloat16 data type.<br>
At first draft of this project, an 3rd party BFloat16 Go library was used, but because of performance issues, to decrease cycle count, own implementation was done.
    >Implementation can be found in [src/dtype](../src/dtype) directory.

* To implement LLaMa Transformer Model blocks and model loader. Inspired by [original LLaMa 2 Python repository of Meta](https://github.com/facebookresearch/llama/blob/main/llama/model.py).
    >Implementation can be found in [src/model](../src/model) directory.

* To implement [Tensor](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) type, tensor aritmetic and machine learning mathematical operation functions.
    >Implementation can be found in [src/ml](../src/ml) directory.

* To implement a console application which gets together these components, and provides a CLI (command line interface).
    >Implementation can be found in [cmd](../cmd) directory.

* To make the inference engine and application supporting "streaming", in Go, [channels](https://go101.org/article/channel.html) was used. To support this, both inference engine and application must be redesigned properly considering normal and edge cases. Also console application must have ability to update its output instead of printing new lines, and it must do it with multi-platform support.

* To add support for Unicode emojis which come separately byte-by-byte, to print names of generated emojis too, because Windows terminals have limited support for them.

* To write and organize a comprehensive documentation explaining the whole of stages, which you can find [this journey in the documentation directory](./).

* To draw some diagrams to support written explanations about the architecture and flow with visuals. In my opinion, the hardest part is this. :blush:

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Documentation Index](./README.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: INITIALIZATION&nbsp;&nbsp;&gt;](./01-INITIALIZATION.md)

</div>
