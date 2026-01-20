"""
Sequence-to-Sequence (Seq2Seq) with Neural Networks (2014)

Sequence to Sequence Learning with Neural Networks
Authors: Ilya Sutskever, Oriol Vinyals, Quoc V. Le (Google)

Seq2Seq introduced the encoder-decoder architecture for mapping variable-length
input sequences to variable-length output sequences using recurrent neural
networks (LSTMs). This became the foundation for neural machine translation
and many other sequence transduction tasks.

Key Insight:
    A sequence can be compressed into a fixed-length "thought vector" (encoding)
    by an encoder RNN, then decoded into a new sequence by a decoder RNN.
    This enables end-to-end learning without manual feature engineering.

Architecture:
    - Encoder: LSTM that reads source sequence and produces hidden state
    - Decoder: LSTM that generates target sequence conditioned on encoding
    - The final hidden state of encoder becomes the initial state of decoder

Key Contributions:
    - End-to-end trainable sequence transduction
    - Reversing source sequence improves performance
    - Deep LSTMs (4 layers) with 1000 units each
    - Beam search decoding
"""


from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage
from typing import Dict, List


SEQ2SEQ = MLMethod(
    method_id="seq2seq_2014",
    name="Sequence-to-Sequence",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE],
    authors=["Ilya Sutskever", "Oriol Vinyals", "Quoc V. Le"],
    paper_title="Sequence to Sequence Learning with Neural Networks",
    paper_url="https://arxiv.org/abs/1409.3215",
    key_innovation="Encoder-decoder architecture using LSTMs to map variable-length "
                   "input sequences to variable-length output sequences via a fixed-size "
                   "'thought vector', enabling end-to-end neural machine translation",
    mathematical_formulation="""
    Encoder (processes source sequence x_1, ..., x_T):
        For t = 1 to T:
            h_t = LSTM_enc(x_t, h_{t-1})
        context = h_T  (final hidden state)

    Decoder (generates target sequence y_1, ..., y_T'):
        s_0 = context  (initialize with encoder final state)
        For t = 1 to T':
            s_t = LSTM_dec(y_{t-1}, s_{t-1})
            p(y_t | y_{<t}, x) = softmax(W_s * s_t)

    Training Objective:
        L = -sum_{t=1}^{T'} log p(y_t | y_{<t}, x)
        (maximize log-likelihood of target sequence)

    Source Reversal (key trick):
        Instead of x_1, x_2, ..., x_T
        Use x_T, x_{T-1}, ..., x_1 as input
        Reduces distance between corresponding words

    Beam Search Decoding:
        Maintain top-k partial hypotheses at each step
        Score = sum of log probabilities
    """,
    predecessors=["lstm_1997", "neural_language_model_2003"],
    successors=["attention_mechanism_2014", "transformer_2017"],
    tags=["machine_translation", "encoder_decoder", "lstm", "sequence_transduction"],
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Sequence-to-Sequence."""
    return SEQ2SEQ


def pseudocode() -> str:
    """Return pseudocode describing Seq2Seq architecture."""
    return """
    ENCODER:

    function encode(source_sequence, embedding, lstm_enc):
        # source_sequence: [x_1, x_2, ..., x_T] (word indices)
        # Optionally reverse the source sequence
        source_sequence = reverse(source_sequence)

        # Initialize hidden state
        h = zeros(hidden_size)
        c = zeros(hidden_size)  # LSTM cell state

        # Process each token
        for t in range(len(source_sequence)):
            x_embed = embedding[source_sequence[t]]
            h, c = lstm_enc(x_embed, h, c)

        # Return final hidden state as context
        return h, c


    DECODER:

    function decode_train(target_sequence, context_h, context_c, embedding, lstm_dec, output_proj):
        # target_sequence: [y_1, y_2, ..., y_T'] (word indices)
        # Training uses teacher forcing (feed ground truth at each step)

        h, c = context_h, context_c  # Initialize with encoder output
        loss = 0

        for t in range(len(target_sequence) - 1):
            # Input: current target word
            y_embed = embedding[target_sequence[t]]

            # LSTM step
            h, c = lstm_dec(y_embed, h, c)

            # Project to vocabulary and compute loss
            logits = output_proj(h)  # Shape: (vocab_size,)
            loss += cross_entropy(logits, target_sequence[t + 1])

        return loss / len(target_sequence)


    function decode_inference(context_h, context_c, embedding, lstm_dec, output_proj,
                              max_length, start_token, end_token):
        # Greedy decoding (or beam search)
        h, c = context_h, context_c
        output_sequence = [start_token]

        for _ in range(max_length):
            # Input: previously generated word
            y_embed = embedding[output_sequence[-1]]

            # LSTM step
            h, c = lstm_dec(y_embed, h, c)

            # Get next word
            logits = output_proj(h)
            next_word = argmax(logits)
            output_sequence.append(next_word)

            if next_word == end_token:
                break

        return output_sequence


    BEAM SEARCH DECODING:

    function beam_search_decode(context_h, context_c, embedding, lstm_dec, output_proj,
                                max_length, start_token, end_token, beam_width=5):
        # Initialize beam with start token
        beams = [Beam(tokens=[start_token], score=0.0, h=context_h, c=context_c)]

        for _ in range(max_length):
            all_candidates = []

            for beam in beams:
                if beam.tokens[-1] == end_token:
                    all_candidates.append(beam)
                    continue

                # Get next word probabilities
                y_embed = embedding[beam.tokens[-1]]
                h, c = lstm_dec(y_embed, beam.h, beam.c)
                log_probs = log_softmax(output_proj(h))

                # Expand beam with top-k words
                top_k_words = top_k(log_probs, beam_width)
                for word, log_prob in top_k_words:
                    new_beam = Beam(
                        tokens=beam.tokens + [word],
                        score=beam.score + log_prob,
                        h=h, c=c
                    )
                    all_candidates.append(new_beam)

            # Keep top beam_width candidates
            beams = sorted(all_candidates, key=lambda b: b.score, reverse=True)[:beam_width]

            # Stop if all beams have ended
            if all(b.tokens[-1] == end_token for b in beams):
                break

        return beams[0].tokens  # Return best beam


    FULL SEQ2SEQ MODEL:

    function seq2seq_forward(source, target, training=True):
        # Encode source sequence
        context_h, context_c = encode(source, src_embedding, encoder_lstm)

        if training:
            # Teacher forcing
            loss = decode_train(target, context_h, context_c,
                              tgt_embedding, decoder_lstm, output_projection)
            return loss
        else:
            # Inference
            output = beam_search_decode(context_h, context_c,
                                       tgt_embedding, decoder_lstm, output_projection,
                                       max_length=50, start_token=SOS, end_token=EOS)
            return output


    DEEP SEQ2SEQ (from paper: 4 layers):

    function deep_encode(source_sequence, embeddings, lstms):
        # lstms: list of 4 LSTM layers
        x = source_sequence

        for layer_idx, lstm in enumerate(lstms):
            h, c = zeros(hidden_size), zeros(hidden_size)
            outputs = []

            for t in range(len(x)):
                if layer_idx == 0:
                    input = embeddings[x[t]]
                else:
                    input = x[t]
                h, c = lstm(input, h, c)
                outputs.append(h)

            x = outputs  # Output becomes input to next layer

        return h, c  # Final layer's final state
    """


def key_equations() -> Dict[str, str]:
    """Return key equations for Seq2Seq in LaTeX-style notation."""
    return {
        "lstm_encoder":
            "h_t, c_t = \\text{LSTM}_{enc}(x_t, h_{t-1}, c_{t-1})",

        "context_vector":
            "\\mathbf{c} = h_T \\quad (\\text{final encoder hidden state})",

        "lstm_decoder":
            "s_t, c'_t = \\text{LSTM}_{dec}(y_{t-1}, s_{t-1}, c'_{t-1})",

        "output_distribution":
            "p(y_t | y_{<t}, \\mathbf{x}) = \\text{softmax}(W_s \\cdot s_t + b_s)",

        "training_objective":
            "L = -\\frac{1}{T'} \\sum_{t=1}^{T'} \\log p(y_t | y_{<t}, \\mathbf{x})",

        "source_reversal":
            "\\mathbf{x}_{reversed} = [x_T, x_{T-1}, \\ldots, x_1]",

        "beam_search_score":
            "\\text{score}(\\mathbf{y}) = \\sum_{t=1}^{|\\mathbf{y}|} \\log p(y_t | y_{<t}, \\mathbf{x})",

        "length_normalization":
            "\\text{score}_{norm}(\\mathbf{y}) = \\frac{1}{|\\mathbf{y}|^\\alpha} \\sum_{t} \\log p(y_t)",
    }


def architecture_details() -> Dict:
    """Return Seq2Seq architecture specifications."""
    return {
        "original_paper": {
            "encoder": "4-layer LSTM, 1000 units each",
            "decoder": "4-layer LSTM, 1000 units each",
            "embedding_dim": 1000,
            "vocabulary": "160K source, 80K target (most frequent)",
            "parameters": "~380M",
            "training": {
                "optimizer": "SGD with gradient clipping (norm 5)",
                "initial_lr": 0.7,
                "lr_schedule": "Halve LR every half epoch after 5 epochs",
                "batch_size": 128,
                "epochs": 7.5,
                "parallelism": "8 GPUs (data parallel)",
            },
        },
        "key_tricks": {
            "source_reversal": "Reverse input sequence (improves BLEU by 1-2)",
            "deep_lstm": "4 layers significantly better than 1 layer",
            "beam_search": "Beam width 2-12, diminishing returns after 12",
            "ensemble": "8-model ensemble adds ~2 BLEU",
        },
        "results_wmt14_en_fr": {
            "single_model": "BLEU 34.8",
            "ensemble": "BLEU 36.5",
            "state_of_art_2014": "BLEU 37.0 (phrase-based with large LM)",
        }
    }


def get_historical_context() -> str:
    """Return historical context and significance of Seq2Seq."""
    return """
    Sequence-to-Sequence (2014) demonstrated that neural networks could perform
    end-to-end sequence transduction, revolutionizing machine translation.

    Before Seq2Seq:
    - Machine translation used statistical phrase-based methods (Moses, etc.)
    - Required extensive feature engineering and linguistic resources
    - Translation pipeline: alignment, phrase extraction, language model, decoding
    - Neural approaches were limited to rescoring or components

    The Breakthrough:
    - Single neural model learns entire translation end-to-end
    - No feature engineering, alignment tables, or linguistic rules
    - Matched phrase-based systems with only neural components
    - Showed that RNNs can capture long-range dependencies

    Published Simultaneously:
    - Sutskever et al. (this paper): Basic encoder-decoder
    - Cho et al.: GRU-based encoder-decoder
    - Bahdanau et al.: Attention mechanism (addressed bottleneck)

    The Fixed-Length Bottleneck:
    - Encoder compresses entire source into single vector
    - Works poorly for long sentences
    - Attention mechanism (Bahdanau 2014) solved this
    - Led directly to the Transformer (2017)

    Impact:
    - Launched the "neural machine translation" revolution
    - Google, Microsoft, Facebook adopted NMT by 2016-2017
    - Architecture became standard for seq-to-seq tasks
    - Foundation for modern dialogue systems, summarization, etc.
    """


def get_limitations() -> List[str]:
    """Return known limitations of Seq2Seq."""
    return [
        "Fixed-length bottleneck limits performance on long sequences",
        "Information compression loses details from early tokens",
        "No explicit alignment between source and target",
        "Struggles with rare words and out-of-vocabulary tokens",
        "Sequential nature prevents parallelization",
        "Long training times due to recurrent computation",
        "Exposure bias: trained with teacher forcing, tested autoregressively",
    ]


def get_applications() -> List[str]:
    """Return applications of Seq2Seq."""
    return [
        "Machine translation (original application)",
        "Text summarization",
        "Dialogue systems and chatbots",
        "Speech recognition (encoder: audio, decoder: text)",
        "Image captioning (encoder: CNN, decoder: LSTM)",
        "Question answering",
        "Code generation",
        "Grammar correction",
        "Text-to-SQL",
    ]


def seq2seq_evolution() -> Dict[str, str]:
    """Return evolution of seq2seq architectures."""
    return {
        "Basic Seq2Seq (2014)": "Encoder-decoder with fixed-length context",
        "Attention (2014)": "Dynamic context at each decoder step (Bahdanau)",
        "Dot-Product Attention (2015)": "Simplified attention computation (Luong)",
        "Copy Mechanism (2015)": "Ability to copy words from source",
        "Coverage (2016)": "Prevents repeated attention to same positions",
        "Transformer (2017)": "Self-attention replaces recurrence entirely",
        "Copy + Coverage (2017)": "Pointer-generator networks",
        "Pre-trained Seq2Seq (2019+)": "BART, T5, mBART - pre-trained encoders and decoders",
    }
