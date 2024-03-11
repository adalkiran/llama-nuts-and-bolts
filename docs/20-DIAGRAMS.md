# **20. DIAGRAMS**

## Complete Model Diagram

The whole flow of the LLaMa 7B-Chat model, this diagram tries to reveal all of the details as much as possible:<br><br>

![Complete Model Diagram](./images/DIAG01-complete-model.drawio.svg)

## STAGE 1: Tokenization

![STAGE 1: Tokenization Diagram](./images/DIAG01-STAGE01-tokenization.drawio.svg)

## STAGE 2: Creating tokens tensor

![STAGE 2: Creating tokens tensor Diagram](./images/DIAG01-STAGE02-creating-tokens-tensor.drawio.svg)

## STAGE 3: Looping through sequence length Diagram

![STAGE 3: Looping through sequence length Diagram](./images/DIAG01-STAGE03-looping-through-sequence-length.drawio.svg)

## STAGE 4: Creating inputTensor

![STAGE 4: Creating inputTensor Diagram](./images/DIAG01-STAGE04-creating-input-tensor.drawio.svg)

## STAGE 5: Forward Pass Through Each Transformer Block Diagram

![STAGE 5: Forward Pass Through Each Transformer Block Diagram](./images/DIAG01-STAGE05-forward-pass-through-each-transformer-block.drawio.svg)

## STAGE 6: Forward Pass Through Attention Pre-normalization

![STAGE 6: Forward Pass Through Attention Pre-normalization Diagram](./images/DIAG01-STAGE06-forward-pass-through-attention-prenormalization.drawio.svg)

## STAGE 7: Forward Pass Through Attention Module

![STAGE 7: Forward Pass Through Attention Module Diagram](./images/DIAG01-STAGE07-forward-pass-through-attention-module.drawio.svg)

## STAGE 8: Forward Pass Through Attention Module - Calculating xq, xk, and xv

![STAGE 8: Forward Pass Through Attention Module - Calculating xq, xk, and xv Diagram](./images/DIAG01-STAGE08-attention-fwd-calculating-xq-xk-xv.drawio.svg)

## STAGE 9: Forward Pass Through Attention Module - Do reshapings

![STAGE 9: Forward Pass Through Attention Module - Do reshapings Diagram](./images/DIAG01-STAGE09-attention-do-reshapings.drawio.svg)

## STAGE 10: Forward Pass Through Attention Module - Apply Rotary Embeddings

![STAGE 10: Forward Pass Through Attention Module - Apply Rotary Embeddings Diagram](./images/DIAG01-STAGE10-attention-apply-rotary-embeddings.drawio.svg)

## STAGE 11: Forward Pass Through Attention Module - Update KV cache

![STAGE 11: Forward Pass Through Attention Module - Update KV cache Diagram](./images/DIAG01-STAGE11-attention-update-kv-cache.drawio.svg)

## STAGE 12: Forward Pass Through Attention Module - Do transposes

![STAGE 12: Forward Pass Through Attention Module - Do transposes Diagram](./images/DIAG01-STAGE12-attention-do-transposes.drawio.svg)

## STAGE 13: Forward Pass Through Attention Module - Calculate scores

![STAGE 13: Forward Pass Through Attention Module - Calculate scores Diagram](./images/DIAG01-STAGE13-attention-calculate-scores.drawio.svg)

## STAGE 14: Forward Pass Through Attention Module - Calculate output

![STAGE 14: Forward Pass Through Attention Module - Calculate output Diagram](./images/DIAG01-STAGE14-attention-calculate-output.drawio.svg)

## STAGE 15: Add attention module output and current tensor

![STAGE 15: Add attention module output and current tensor Diagram](./images/DIAG01-STAGE15-add-attention-output-and-current-tensor.drawio.svg)

## STAGE 16: Stage: Forward Pass Through Feed-Forward Pre-normalization

![STAGE 16: Stage: Forward Pass Through Feed-Forward Pre-normalization Diagram](./images/DIAG01-STAGE16-forward-pass-through-ffn-prenormalization.drawio.svg)

## STAGE 17: Forward Pass Through Feed-Forward Module

![STAGE 17: Forward Pass Through Feed-Forward Module Diagram](./images/DIAG01-STAGE17-forward-pass-through-ffn-module.drawio.svg)

## STAGE 18: Add Feed-Forward module output and current tensor

![STAGE 18: Add Feed-Forward module output and current tensor Diagram](./images/DIAG01-STAGE18-add-ffn-output-and-current-tensor.drawio.svg)

## STAGE 19: Forward Pass Through Output of The Transformer

![STAGE 19: Forward Pass Through Output of The Transformer Diagram](./images/DIAG01-STAGE19-forward-pass-through-output.drawio.svg)
<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: REFERENCES](./19-REFERENCES.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Home&nbsp;&nbsp;&gt;&gt;](../README.md)

</div>
