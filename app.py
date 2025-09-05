# app.py
import re
import numpy as np
import tensorflow as tf
import streamlit as st
from datasets import load_dataset

# ---- Helpers to load model + build vocab (cached) ----
@st.cache(allow_output_mutation=True)
def load_model_and_vocab(model_path="./saved_model/model.keras"):
    # load model (fixing small typo from earlier)
    model = tf.keras.models.load_model(model_path)

    # Build vocab same as your preprocessing
    ds = load_dataset("Trelis/tiny-shakespeare")
    newVocab = []
    def create_new_vocab(sentence, newVocab):
        sentence_array = re.findall(r"\b\w+\b", sentence.lower())
        new_words = [x for x in sentence_array if x not in newVocab]
        newVocab += list(set(new_words))

    ds['train'].map(lambda x: create_new_vocab(x["Text"], newVocab))
    ds['test'].map(lambda x: create_new_vocab(x["Text"], newVocab))

    # insert PAD at 0, UNK at 1 to match your pipeline
    newVocab.insert(0, "<UNK>")
    newVocab.insert(0, "<PAD>")

    index_to_word = {i: k for i, k in enumerate(newVocab)}
    word_to_index = {k: i for i, k in enumerate(newVocab)}

    # determine maxInDs the same way you did during preprocessing
    maxInDs = 0
    for x in ds['train']:
        sentence_array = re.findall(r"\b\w+(?:'\w+)?\b|[.!?,-]", x['Text'].lower())
        maxInDs = max(len(sentence_array), maxInDs)

    return model, index_to_word, word_to_index, maxInDs

model, index_to_word, word_to_index, maxInDs = load_model_and_vocab()

# pull out trained layers (keep your indexing)
embedding_layer = model.layers[1]
lstm_layer_1 = model.layers[2]
lstm_layer_2 = model.layers[4]
dense_layer_1 = model.layers[6]

# get units to init states (safe)
units_1 = getattr(lstm_layer_1, "units", 156)
units_2 = getattr(lstm_layer_2, "units", 256)


# ---- Generation logic ----
def tokenize_and_pad(text, word_to_index, max_len):
    # same tokenization you used
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[.!?,-]", text.lower())
    idxs = [word_to_index.get(w, word_to_index.get("<UNK>", 1)) for w in tokens]
    # truncate to rightmost tokens if longer
    if len(idxs) > max_len:
        idxs = idxs[-max_len:]
    # pad to max_len on the right with PAD=0
    pad_len = max_len - len(idxs)
    return idxs + [0] * pad_len

def stable_sample_from_logits(logits, temperature=1.0, top_k=None):
    """
    logits: numpy array shape (vocab,) (for a single timestep)
    returns: sampled token index
    """
    # numerical stability
    scaled = logits / max(temperature, 1e-8)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    probs = exp / np.sum(exp)

    if top_k is not None and top_k > 0 and top_k < probs.size:
        top_idx = np.argsort(probs)[-top_k:]
        top_probs = probs[top_idx]
        top_probs = top_probs / top_probs.sum()
        return np.random.choice(top_idx, p=top_probs)
    else:
        return np.random.choice(len(probs), p=probs)

def generate_from_seed(seed_text, num_words=30, temperature=1.0, top_k=10):
    # prepare initial sequence (pad to model input length)
    seq = tokenize_and_pad(seed_text, word_to_index, maxInDs)
    # init states with zeros (batch_size=1)
    h0 = np.zeros((1, units_1), dtype=np.float32)
    c0 = np.zeros((1, units_1), dtype=np.float32)
    h1 = np.zeros((1, units_2), dtype=np.float32)
    c1 = np.zeros((1, units_2), dtype=np.float32)

    # feed the whole sequence once to get the states after the seed
    inputs = tf.constant([seq], dtype=tf.int32)  # shape (1, maxInDs)
    X = embedding_layer(inputs)  # (1, maxInDs, emb_dim)

    # try to get states from LSTM layers (works if the layer was created with return_state=True)
    try:
        out1, h0, c0 = lstm_layer_1(X)  # process full seed
        out2, h1, c1 = lstm_layer_2(out1)
        last_token = seq[len([i for i in seq if i != 0]) - 1] if any(seq) else 0
    except Exception:
        # fallback: layer doesn't return states; use output only and take final timestep output
        out1 = lstm_layer_1(X)
        out2 = lstm_layer_2(out1)
        # can't update hidden states; derive logits from last timestep and continue via argmax sampling fallback
        logits = dense_layer_1(out2).numpy()
        last_token = int(np.argmax(logits[0, -1, :]))

    generated = []
    # if seed contains actual tokens, include their words in the returned text optionally
    seed_words = []
    # decode seed into words (skip PAD)
    nonpad_idxs = [i for i in seq if i != 0]
    for idx in nonpad_idxs:
        seed_words.append(index_to_word.get(idx, "<UNK>"))

    # generate step-by-step using states and last_token
    for _ in range(num_words):
        # embed last token
        inp = tf.constant([[int(last_token)]], dtype=tf.int32)
        emb = embedding_layer(inp)  # (1,1,emb_dim)

        # call lstm layers with states; handle both possible returns
        try:
            out1_step, h0, c0 = lstm_layer_1(emb, initial_state=[h0, c0])
        except Exception:
            out1_step = lstm_layer_1(emb, initial_state=[h0, c0])

        try:
            out2_step, h1, c1 = lstm_layer_2(out1_step, initial_state=[h1, c1])
        except Exception:
            out2_step = lstm_layer_2(out1_step, initial_state=[h1, c1])

        logits = dense_layer_1(out2_step).numpy()  # (1,1,vocab)
        logits = logits[0, -1, :]  # (vocab,)

        # sample next token
        next_tok = stable_sample_from_logits(logits, temperature=temperature, top_k=top_k)
        generated.append(int(next_tok))
        last_token = int(next_tok)

    # decode generated tokens to words
    gen_words = [index_to_word.get(i, "<UNK>") for i in generated]
    # join seed and generated for final output
    full_words = seed_words + gen_words
    return " ".join(full_words)

# ---- Streamlit UI ----
st.title("Shakespeare-style text generator")

seed = st.text_input("Enter 2–3 initial words (seed):", value="to be")
num = st.slider("Number of words to generate", min_value=5, max_value=200, value=40)
temp = st.slider("Temperature (randomness)", min_value=0.1, max_value=2.0, value=0.7, step=0.05)
topk = st.slider("Top-k (0 = disabled)", min_value=0, max_value=200, value=10)

if st.button("Generate"):
    if not seed.strip():
        st.warning("Please provide a seed (2-3 words recommended).")
    else:
        with st.spinner("Generating..."):
            result = generate_from_seed(seed, num_words=num, temperature=float(temp), top_k=(None if topk == 0 else int(topk)))
        st.text_area("Generated text", value=result, height=300)
