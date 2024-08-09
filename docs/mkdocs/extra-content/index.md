---
title: HOME
type: docs
menus:
  - main
weight: 0
---
# <img src="assets/icon.svg" style="width: 0.8em"></img> **Llama Nuts and Bolts**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&style=flat-square)](https://www.linkedin.com/in/alper-dalkiran/)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=X&logoColor=white&style=flat-square)](https://twitter.com/aalperdalkiran)
![HitCount](https://hits.dwyl.com/adalkiran/llama-nuts-and-bolts.svg?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

!!! info "Welcome!"

    This documentation website is a customized version of original documentation of the [:fontawesome-brands-github: Llama Nuts and Bolts repository](https://github.com/adalkiran/llama-nuts-and-bolts). You can find the running Go implementation of the project codes in this repository.

A holistic way of understanding how Llama and its components run in practice, with code and detailed documentation. "The nuts and bolts" (practical side instead of theoretical facts, pure implementation details) of required components, infrastructure, and mathematical operations without using external dependencies or libraries.

This project intentionally **<u>doesn't have</u>** support for [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) (such as [nVidia CUDA](https://tr.wikipedia.org/wiki/CUDA), [OpenCL](https://tr.wikipedia.org/wiki/OpenCL)) as well as [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) because it doesn't aim to be a production application, for now. Instead, the project relies on CPU cores to perform all mathematical operations, including linear algebraic computations. To increase performance, the code has been optimized as much as necessary, utilizing parallelization via [goroutines](https://gobyexample.com/goroutines).

![Llama Nuts and Bolts Screen Recording GIF](images/llama-nuts-and-bolts-screen-record.gif)<br>
<sup>*Llama Nuts and Bolts Screen Recording GIF, captured while the application was running on the Apple MacBook Pro M1 Chip. Predefined prompts within the application were executed. The GIF is 20x faster.*</sup>

## :thought_balloon: **WHY THIS PROJECT?**

This project was developed for only educational purposes, and has not been tested for production or commercial usage. The goal is to make an experimental project that can perform inference on the Llama 3.1 8B-Instruct model completely outside of the Python ecosystem. Throughout this journey, the aim is to acquire knowledge and shed light on the abstracted internal layers of this technology.

This journey is an intentional journey of literally *reinventing the wheel*. While reading this journey here, you will navigate toward the target with a deductive flow. You will encounter the same stops and obstacles I encountered during this journey.

If you are curious like me about how the LLMs (Large Language Models) and transformers work and have delved into conceptual explanations and schematic drawings in the sources but hunger for deeper understanding, then this project is perfect for you too!

## :triangular_ruler: **MODEL DIAGRAM**

The whole flow of Llama 3.1 8B-Instruct model without abstraction:<br><br>

![Complete Model Diagram](images/DIAG01-complete-model.drawio.svg)

## :dart: **COVERAGE**

Due to any of the existing libraries (except the built-in packages and a few helpers) wasn't used, all of the required functions were implemented by this project in the style of Go. However, the main goal of this project is to do inference only on the Llama 3.1 8B-Instruct model, the functionality fulfills only the requirements of this specific model. Not much, not less, because the goal of our project is not to be a production-level tensor framework.

The project provides a CLI (command line interface) application allowing users to choose from predefined prompts or write custom prompts. It then performs inference on the model and displays the generated text on the console. The application supports "streaming," enabling immediate display of generated tokens on the screen without waiting for the entire process to complete.

As you can see in the chapters here, covered things are:

* All diagrams about the model and the flow are listed in [Chapter 20](20-DIAGRAMS.md),
* Parallelization and concurrency in Go, see [Chapter 13](13-GENERATING-NEXT-TOKENS.md),
* Implementing [Memory Mapping](https://en.wikipedia.org/wiki/Memory-mapped_file) that allows us to map a large file content to memory address space in Go for both Linux/MacOS and Windows platforms, see [Chapter 2](02-LOADING-TORCH-MODEL.md),
* Implementing [BFloat16 (Brain Floating Point)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) data type which isn't supported by the Go language, from scratch, see [Chapter 7](07-BFLOAT16-DATA-TYPE.md),
* Implementing support for "streaming" output via [Go channels](https://go101.org/article/channel.html), see [Chapter 13](13-GENERATING-NEXT-TOKENS.md),
* Loading a [PyTorch](https://pytorch.org/) model weights file ("consolidated.00.pth") which was saved as [Pickle (.pkl)](https://github.com/python/cpython/blob/main/Lib/pickle.py) format, from scratch, see [Chapter 2](02-LOADING-TORCH-MODEL.md) and [Chapter 3](03-LOADING-TORCH-MODEL-DETAILS.md),
* Loading the model arguments JSON file ("params.json"), see [Chapter 4](04-LOADING-MODEL-ARGS.md),
* Loading a [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/en/chapter6/5) tokenizer model which was saved as [Tiktoken tokenizer format](https://github.com/openai/tiktoken) file ("tokenizer.model"), from scratch, see [Chapter 5](05-LOADING-TOKENIZER-MODEL/),
* Implementing a [Tensor](https://en.wikipedia.org/wiki/Tensor_%28machine_learning%29) type, tensor aritmetic and machine learning mathematical operation functions, see [Chapter 8](08-TENSOR.md),
* Working with  [C contiguous](https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays) arrays in multi-dimensional form, see [Chapter 8](08-TENSOR.md),
* Building the blocks of Llama 3.1 model architecture, see [Chapter 9](09-IMPLEMENTING-LLAMA-MODEL-ARCHITECTURE.md),
* Implementing [RoPE \(Rotary Positional Embeddings\)](https://arxiv.org/abs/2104.09864v5) and precomputing frequency tensor, see [Chapter 10](10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md) and [Chapter 10.BONUS](10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb),
* Understanding tokens, vocabulary, and tokenization, see [Chapter 12](12-TOKENIZATION.md),
* Generating the next token, internals of transformer block, being an auto-regressive model, multi-head self-attention, and much more, see [Chapter 13](13-GENERATING-NEXT-TOKENS.md), [Chapter 14](14-MAKING-PREDICTION-WITH-LLAMA-MODEL-1.md), [Chapter 15](15-MAKING-PREDICTION-WITH-LLAMA-MODEL-2.md), [Chapter 16](16-MAKING-PREDICTION-WITH-LLAMA-MODEL-3.md),
* Understanding internals of the [Unicode Standard](https://en.wikipedia.org/wiki/Unicode), [UTF-8](https://en.wikipedia.org/wiki/UTF-8) encoding, and how emojis are represented and rendered, see [Chapter 17](17-UNICODE-UTF-8-EMOJIS.md),
* And, so much more!

## :package: **INSTALLATION and BUILDING**

Installation and building instructions are described at [:fontawesome-brands-github: GitHub README](https://github.com/adalkiran/llama-nuts-and-bolts#package-installation-and-building).

## :computer: **RUNNING**

Run the project with executing ```go run ...``` command or executing the compiled executable. It's more suggested that to run this project's executable after building it, and without virtualization for higher performance.

When you run the project, you will see the following screen. It prints the summary of the loading process of model files and a summary of model details.

![First start of the application](images/SS01-first-start.png)

### Printing Model Metadata

If you select the first item in the menu by pressing 0 key and ENTER, the application prints the metadata of Llama 3.1 8B-Instruct model on the console:

![Printing metadata 1](images/SS02-print-metadata-1.png)
![Printing metadata 2](images/SS02-print-metadata-2.png)

### Executing a Prompt

Alongside you can select one of predefined prompts in the menu, you can select one of latest two items (```Other, manual input```) to input your custom prompts.

With the ```[Text completion]``` choices, the model is used only to perform text completion task. New tokens will be generated according to the input prompt text.

With the ```[Chat mode]``` choices, the application starts the prompt with ```<|begin_of_text|>``` string to specify "this is an instruction prompt". Also it surrounds the system prompt part with ```<|start_header_id|>system<|end_header_id|>\n``` and ```<|eot_id|>``` strings to specify this part is a *system prompt*, surrounds the user prompt part with ```<|start_header_id|>user<|end_header_id|>\n``` and ```<|eot_id|>``` strings to specify this part is a *user prompt*.

At the end, a chat mode prompt string will be look like following:

```sh
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Always answer with emojis<|eot_id|><|start_header_id|>user<|end_header_id|>

How to go from Beijing to NY?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
```

And the output of this prompt is like the following (consists of emojis with their names and unicode escape sequences):

![Example emoji output](images/SS03-example-emoji-output.png)

## :bricks: **ASSUMPTIONS**

The full-compliant, generic, production-ready, and battle-tested tensor frameworks should have support for a wide range of platforms, acceleration devices/processors/platforms, use cases, and lots of convertibility between data types, etc.

In **Llama Nuts and Bolts** scenario, some assumptions have been made to focus only on required set of details.

| Full-compliant applications/frameworks | Llama Nuts and Bolts |
|---|---|
| Use existing robust libraries to read/write file formats, perform calculations, etc. | This project aims to *reinvent the wheel*, so it doesn't use any existing library. It implements everything it requires, precisely as much as necessary. |
| Should support a wide range of different data types and perform calculations between different typed tensors in an optimized and performant way. | Has a limited elasticity for only required operations. |
| Should support a wide range of different file formats. | Has a limited support for only required file formats with only required instructions. |
| Should support *top-k*, *top-p*, and, *temperature* concepts of the LLMs (Large Language Models) to randomize the outputs, explained [here](https://ivibudh.medium.com/a-guide-to-controlling-llm-model-output-exploring-top-k-top-p-and-temperature-parameters-ed6a31313910). | This project doesn't have support for randomized outputs intentionally, just gives the outputs that have the highest probability. |
| Should support different acceleration technologies such as [nVidia CUDA](https://tr.wikipedia.org/wiki/CUDA), [OpenCL](https://tr.wikipedia.org/wiki/OpenCL), [Metal Framework](https://developer.apple.com/documentation/metal), [AVX2 instructions](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions), and [ARM Neon instructions](https://developer.arm.com/Architectures/Neon), that enable us [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) or [SIMD (Single instruction, multiple data)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) usage. | This project doesn't have support for [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) and [SIMD (Single instruction, multiple data)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) intentionally because it doesn't aim to be a production application, for now. However, for a few days, I had tried an experiment with [ARM Neon instructions](https://developer.arm.com/Architectures/Neon) on my MacBook Pro M1, it  worked successfully with float32 data type, but with the CPU cycles required to convert BFloat16 to float32 negated the saved time that came with ARM Neon.<br><br>Also, I've realized that the Go compiler doesn't have support for 2-byte floats, even though I've tried using CGO. So, I gave up on this issue. If you're curious about it, you can check out the single commit on the experiment branch [arm_neon_experiment](https://github.com/adalkiran/llama-nuts-and-bolts/commit/7fd3ad4e1268a79fe6e404780b3ebce39cb0710e). |

## :star: **CONTRIBUTING and SUPPORTING the PROJECT**

You are welcome to [create issues](https://github.com/adalkiran/llama-nuts-and-bolts/issues/new) to report any bugs or problems you encounter. At present, I'm not sure whether this project should be expanded to cover more concepts or not. Only time will tell :blush:.

If you liked and found my project helpful and valuable, I would greatly appreciate it if you could give the repo a star :star: on [:fontawesome-brands-github: GitHub](https://github.com/adalkiran/llama-nuts-and-bolts). Your support and feedback not only help the project improve and grow but also contribute to reaching a wider audience within the community. Additionally, it motivates me to create even more innovative projects in the future.

## :book: **REFERENCES**

I want to thank to contributors of the awesome sources which were referred during development of this project and writing this documentation. You can find these sources below, also in between the lines in code and documentation.

You can find a complete and categorized list of refereces in [19. REFERENCES](19-REFERENCES.md) chapter of this documentation.

The following resources are  most crucial ones, but it's suggested that to check out the [19. REFERENCES](19-REFERENCES.md) chapter:

* [Meta Llama website](https://llama.meta.com/)
* [Original Llama 3.1 Python package repository of Meta](https://github.com/meta-llama/llama-models/)
* [Original Llama Toolchain Python repository of Meta](https://github.com/meta-llama/llama-toolchain)
* [Georgi Gerganov](https://github.com/ggerganov)'s [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [Wikipedia](https://en.wikipedia.org)
* [PyTorch Documentation](https://pytorch.org/)
* [Youtube - Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
* [Llama: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Youtube - Umar Jamil - Llama explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU](https://www.youtube.com/watch?v=Mn_9W1nCFLo)
* [Youtube - DeepLearning Hero - RoPE (Rotary positional embeddings) explained: The positional workhorse of modern LLMs](https://www.youtube.com/watch?v=GQPOtyITy54)
* [Youtube - Intermation - Computer Organization and Design Fundamentals - Ep 020: Unicode Code Points and UTF-8 Encoding](https://www.youtube.com/watch?v=tbdym9ZtepQ&list=PLxfrSxK7P38X7XfG4X8Y9cdOURvC7ObMF)
* Several documents, articles, and code samples: In the code and documentation of this project, you can find several code or document links that were cited.

## :scroll: **LICENSE**

Llama Nuts and Bolts is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/adalkiran/llama-nuts-and-bolts/blob/main/LICENSE) for the full license text.
