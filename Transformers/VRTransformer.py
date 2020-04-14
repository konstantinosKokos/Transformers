# Variable length R-Transformer.

# A variant of the standard R-Transformer, where a bounded window recurrence of variable length is applied over the
# decoder output. Useful in cases of translating to a locally structured space.

# During training, the window lengths need to be given a priori, enabling batching over the localist RNN.
# During inference, the recurrence is iterated over the generated output, and is reset whenever a pre-specified output
#     symbol is generated.
#
# This content is provided under GNU General Public License.
# If you find this architecture useful to your experiments, please cite:
#
#     @misc{VRTransformer,
#         author = {Kogkalidis, Konstantinos},
#         title = {Variable length R-Transformers},
#         year = {2019},
#         publisher = {GitHub},
#         journal = {GitHub repository},
#         howpublished = {\url{https://github.com/konstantinosKokos/Transformers}},
#     }


from Transformers.utils import *
from Transformers.Transformer import EncoderLayer, DecoderLayer, Encoder, Decoder


class VRTransformer(nn.Module):
    def __init__(self, num_classes: int, encoder_heads: int = 8, decoder_heads: int = 8, encoder_layers: int = 6,
                 decoder_layers: int = 6, d_model: int = 300, d_intermediate: int = 128, dropout: float = 0.1,
                 device: str = 'cpu', activation: Callable[[Tensor], Tensor] = sigsoftmax,
                 reuse_embedding: bool = True, predictor: Optional[nn.Module] = None,
                 recurrence: Optional[nn.Module] = None) -> None:

        self.device = device
        super(VRTransformer, self).__init__()
        self.encoder = Encoder(num_layers=encoder_layers, num_heads=encoder_heads, d_model=d_model,
                               d_k=d_model // encoder_heads, d_v=d_model // encoder_heads,
                               d_intermediate=d_intermediate, dropout=dropout).to(self.device)
        self.decoder = Decoder(num_layers=decoder_layers, num_heads=decoder_heads, d_model=d_model,
                               d_k=d_model // decoder_heads, d_v=d_model // decoder_heads,
                               d_intermediate=d_intermediate, dropout=dropout).to(self.device)
        self.embedding_matrix = torch.nn.Parameter(torch.rand(num_classes, d_model, device=device) * 0.02)
        self.output_embedder = lambda x: F.embedding(x, self.embedding_matrix, padding_idx=0, scale_grad_by_freq=True)

        if reuse_embedding:
            self.predictor = lambda x: x @ (self.embedding_matrix.transpose(1, 0) + 1e-10)
        elif predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = nn.Linear(in_features=d_model, out_features=num_classes).to(self.device)

        if recurrence is not None:
            self.recurrence = recurrence
        else:
            print("Defaulting to LSTM recurrence")
            self.recurrence = nn.LSTM(input_size=d_model, hidden_size=d_model, bidirectional=False,
                                      num_layers=1, batch_first=True).to(self.device)

        self.activation = activation

    def forward(self, encoder_input: Tensor, decoder_input: Tensor, encoder_mask: LongTensor,
                decoder_mask: LongTensor, windows: Windows) -> Tensor:
        self.train()

        b, n, dk = encoder_input.shape
        n_out = decoder_input.shape[1]
        pe = make_positional_encodings(b, n, dk, dk, device=self.device)
        pe_dec = make_positional_encodings(b, n_out, dk, dk, device=self.device)
        encoder_output = self.encoder(EncoderInput(encoder_input + pe, encoder_mask[:, :n, :]))

        # local contextualization
        local_batched, local_indices = batchify_local(decoder_input, windows)
        local_batched, _ = self.recurrence(local_batched)
        decoder_input = recover_batch(decoder_input, local_batched, local_indices)

        decoder_output = self.decoder(DecoderInput(encoder_output=encoder_output.encoder_input,
                                                   encoder_mask=encoder_mask, decoder_input=decoder_input + pe_dec,
                                                   decoder_mask=decoder_mask))
        prediction = self.predictor(decoder_output.decoder_input)
        return torch.log(self.activation(prediction))

    def infer(self, encoder_input: Tensor, encoder_mask: LongTensor, sos_symbol: int, reset_symbol: int) \
            -> Tensor:
        self.eval()

        with torch.no_grad():
            b, n, dk = encoder_input.shape
            max_steps = encoder_mask.shape[1]
            pe = make_positional_encodings(b, max_steps, dk, dk, device=self.device)
            encoder_output = self.encoder(EncoderInput(encoder_input + pe[:, :n], encoder_mask[:, :n, :])).encoder_input
            sos_symbols = (torch.ones(b) * sos_symbol).long().to(self.device)

            decoder_output = self.output_embedder(sos_symbols).unsqueeze(1) + pe[:, 0:1, :]
            decoder_output, (decoder_hidden, decoder_context) = self.recurrence(decoder_output)

            output_probs = torch.Tensor().to(self.device)
            inferer = infer_wrapper(self, encoder_output, encoder_mask, b)
            decoder_mask = make_mask((b, encoder_mask.shape[1], encoder_mask.shape[1])).to(self.device)

            for t in range(max_steps):
                prob_t = inferer(decoder_output=decoder_output, t=t, decoder_mask=decoder_mask)

                class_t = prob_t.argmax(dim=-1)
                reset = torch.where(class_t == reset_symbol, torch.ones(1, dtype=torch.uint8).to(self.device),
                                    torch.zeros(1, dtype=torch.uint8).to(self.device)).unsqueeze(0).unsqueeze(-1)
                decoder_hidden = decoder_hidden.masked_fill(reset.repeat(1, 1, dk), 0)
                decoder_context = decoder_context.masked_fill(reset.repeat(1, 1, dk), 0)

                emb_t = self.output_embedder(class_t).unsqueeze(1) + pe[:, t + 1:t + 2, :]
                if emb_t.shape[1]:
                    emb_t, (decoder_hidden, decoder_context) = self.recurrence(emb_t, (decoder_hidden, decoder_context))
                decoder_output = torch.cat([decoder_output, emb_t], dim=1)

                output_probs = torch.cat([output_probs, prob_t.unsqueeze(1)], dim=1)

        return output_probs

    def infer_one(self, encoder_output: Tensor, encoder_mask: LongTensor, decoder_output: Tensor,
                  t: int, b: int, decoder_mask: Optional[LongTensor] = None) -> Tensor:
        if decoder_mask is None:
            decoder_mask = make_mask((b, t + 1, t + 1)).to(self.device)
        decoder_step = self.decoder(DecoderInput(encoder_output=encoder_output, encoder_mask=encoder_mask,
                                                 decoder_input=decoder_output,
                                                 decoder_mask=decoder_mask[:, :t+1, :t+1])).decoder_input
        prob_t = self.predictor(decoder_step[:, -1])
        return self.activation(prob_t)  # b, num_classes


def test(device: str):
    def make_random_windows(batch_size, seq_len) -> Sequence[Sequence[range]]:
        R = range(seq_len)
        ret = []
        for b in range(batch_size):
            local = []
            curr = 0
            while curr < seq_len - 1:
                next = np.random.randint(curr + 1, seq_len)
                local.append(range(curr, next))
                curr = next
            ret.append(local)
        return ret

    b = 128
    sl = 25
    nc = 1000
    windows = make_random_windows(b, sl)

    t = VRTransformer(12, device=device)
    encoder_input = torch.rand(b, sl, 300).to(device)
    encoder_mask = torch.ones(b, sl*2, sl).to(device)
    decoder_input = torch.rand(b, sl * 2, 300).to(device)
    decoder_mask = make_mask((b, sl * 2, sl * 2)).to(device)
    # p, s = t.vectorized_beam_search(encoder_input[0:20], encoder_mask[0:20], 0, 3)
    f_v = t.forward(encoder_input, decoder_input, encoder_mask, decoder_mask, windows)
    i_v = t.infer(encoder_input[0:20], encoder_mask[0:20, :50], 0, 0)
    import pdb
    pdb.set_trace()
