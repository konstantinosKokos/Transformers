#### Transformers
A working repository containing from-scratch PyTorch implementations of a few Transformer variants.

Currently includes:

* The Vanilla transformer ([Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf))
* The Universal transformer ([Universal Transformers](https://arxiv.org/pdf/1807.03819.pdf))
* A variable length R-transformer (loosely based on [R-Transformers: Recurrent Neural Network Enhanced Transformers](https://arxiv.org/abs/1907.05572))

A few utility functions are taken from [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) 
tutorial. Important additions, aside the implementation of the Universal Transformer, include PEP-484 style type annotations, more efficient vectorization, and a batch-vectorized implementation of beam search decoding. 

*Note that as of version 1.2 PyTorch provides implementations of core transformer modules.*

The code has been tested in various generalized translation tasks where it performs consistently and yields high scores with minimal training times. 

An example usecase of the standard transformer for type-logical grammar induction, including training and evaluation scripts,
can be found [here](https://github.com/konstantinosKokos/Lassy-TLG-Supertagging).

If you encounter any issues or require any assistance, do not hesitate to open an issue or get in touch.

![alt text](https://slack-imgs.com/?c=1&url=https%3A%2F%2Fwww.thespruce.com%2Fthmb%2FCG6T8EjAeGCjHKES-5xcRbMy05M%3D%2F1500x1500%2Ffilters%3Ano_upscale()%3Amax_bytes(150000)%3Astrip_icc()%2FRescue_Bots_Optimus_Prime_Racing_trailer-585d93633df78ce2c3237f7e.jpg)
