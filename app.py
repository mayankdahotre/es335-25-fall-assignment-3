import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
import numpy as np
import os
from PIL import Image # Import Image to display plots

# --- 1. Model Definition (Must match the one in the notebook) ---

class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, num_layers, activation_fn):
        super(NextWordMLP, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.layer_1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.activation1 = activation_fn
        self.dropout1 = nn.Dropout(0.5)

        self.num_layers = num_layers
        if num_layers == 2:
            self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
            self.activation2 = activation_fn
            self.dropout2 = nn.Dropout(0.5)

        # Output layer
        self.layer_3 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(inputs.size(0), -1)

        out = self.activation1(self.layer_1(embeds))
        out = self.dropout1(out)

        if self.num_layers == 2:
            out = self.activation2(self.layer_2(out))
            out = self.dropout2(out)

        out = self.layer_3(out)
        return out

# --- 2. Helper Functions ---

def preprocess_text(text, category_name='Category I'):
    """
    Lowercase and tokenize text. Use slightly different rules for Category I vs Category II.
    Category I (Sherlock Holmes): allow letters, digits and periods (treated as sentence token).
    Category II (Sklearn Docs): allow common punctuation used in docs so tokenization is more permissive.
    """
    text = text.lower()

    # Category II (Sklearn Docs) — keep commas, colons, semicolons, parentheses and hyphens
    if 'category ii' in category_name.lower() or 'sklearn' in category_name.lower() or 'cat2' in category_name.lower():
        # remove characters that are not alnum or common punctuation used in docs
        text = re.sub(r"[^a-z0-9 \.,;:\-()\[\]{}'\"]", ' ', text)
        # ensure punctuation separated from words for tokenization
        text = re.sub(r'([\.,;:\-()\[\]{}\"\'])', r' \1 ', text)
        words = [w for w in text.split() if w]
        return words

    # Default / Category I: stricter cleanup (matches original notebook preprocessing)
    text = re.sub(r'[^a-z0-9 \.]', '', text)
    text = text.replace('.', ' . ')
    words = text.split()
    return words

@st.cache_resource
def load_resources(category_name, activation_name, embed_dim, num_layers):
    """
    Loads the correct model and vocabulary files based on ALL user selections.
    """

    # --- 1. Determine File Paths ---
    if "Category I" in category_name or "Natural" in category_name:
        ds_name = 'cat1'
    else: # Category II
        ds_name = 'cat2'

    act_name = activation_name.lower()
    
    # Use the fixed hidden dim from the assignment/notebook
    HIDDEN_DIM = 1024 
    # Use the fixed context size from the notebook
    CONTEXT_SIZE = 8  

    # Construct the unique model ID from the notebook's training loop
    model_id = f"{ds_name}_embed{embed_dim}_layers{num_layers}_{act_name}"

    MODEL_PATH = f'model_{model_id}.pth'
    VOCAB_PATH = f'vocab_{ds_name}.json'
    
    # Paths for plots
    LOSS_PLOT_PATH = f'loss_{model_id}.png'
    TSNE_PLOT_PATH = f'tsne_{model_id}.png'

    # --- 2. Check if files exist ---
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        st.error(f"Error: Model file `{MODEL_PATH}` or vocab file `{VOCAB_PATH}` not found.")
        st.error("Please ensure all 16 .pth files and the 2 vocab.json files from the notebook are in the same directory as this app.")
        return None, None, None, None, None, None

    # --- 3. Load Vocabulary ---
    try:
        with open(VOCAB_PATH, 'r') as f:
            vocab = json.load(f)
        word_to_ix = vocab['word_to_ix']
        ix_to_word = {int(k): v for k, v in vocab['ix_to_word'].items()}
        vocab_size = len(word_to_ix)
    except Exception as e:
        st.error(f"Error loading vocab file `{VOCAB_PATH}`: {e}")
        return None, None, None, None, None, None

    # --- 4. Define Model Architecture ---
    activation_fn = nn.ReLU() if act_name == "relu" else nn.Tanh()

    model = NextWordMLP(vocab_size, embed_dim, CONTEXT_SIZE, HIDDEN_DIM, num_layers, activation_fn)

    # --- 5. Load Trained Weights ---
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model state dict from `{MODEL_PATH}`: {e}")
        st.info("Ensure the model definition in app.py matches the one used for training in the notebook.")
        return None, None, None, None, None, None

    # --- 6. Load Plots ---
    loss_plot_img = Image.open(LOSS_PLOT_PATH) if os.path.exists(LOSS_PLOT_PATH) else None
    tsne_plot_img = Image.open(TSNE_PLOT_PATH) if os.path.exists(TSNE_PLOT_PATH) else None


    # Store params for display
    params = {
        "Model ID": model_id,
        "Category": category_name,
        "Embedding Dim": embed_dim,
        "Context Size": CONTEXT_SIZE,
        "Hidden Dim 1": HIDDEN_DIM,
        "Hidden Dim 2": HIDDEN_DIM if num_layers == 2 else "N/A",
        "Num Layers": num_layers,
        "Activation": act_name,
        "Vocabulary Size": vocab_size
    }

    return model, word_to_ix, ix_to_word, params, loss_plot_img, tsne_plot_img

def generate_text(model, word_to_ix, ix_to_word, context_size, input_text, n_words, temperature, category_name='Category I'):
    generated_words = []
    context_words = preprocess_text(input_text, category_name)
    
    # Handle OOV words from user by mapping to <UNK>
    def get_ix(word):
        return word_to_ix.get(word, word_to_ix['<UNK>'])

    for _ in range(n_words):
        # Prepare context tensor
        if len(context_words) < context_size:
            pad_indices = [word_to_ix['<UNK>']] * (context_size - len(context_words))
            input_indices = pad_indices + [get_ix(w) for w in context_words]
        else:
            input_indices = [get_ix(w) for w in context_words[-context_size:]]

        context_tensor = torch.tensor([input_indices], dtype=torch.long)

        with torch.no_grad():
            log_probs = model(context_tensor)
            # Apply temperature
            probs = F.softmax(log_probs / max(temperature, 1e-6), dim=1)
            predicted_index = torch.multinomial(probs, 1).item()

        predicted_word = ix_to_word.get(predicted_index, '<UNK>')
        generated_words.append(predicted_word)
        context_words.append(predicted_word) # Add predicted word to context for next step

    output_text = ' '.join(generated_words)
    # Clean up output (e.g., remove space before a period)
    output_text = output_text.replace(' .', '.').strip()
    return output_text

# --- 3. Streamlit App UI ---

st.set_page_config(layout="wide")

# --- Simple styling to improve look and feel ---
st.markdown(
    """
    <style>
    .app-header { font-size:28px; font-weight:700; margin-bottom:6px }
    .app-sub { color: #666; margin-bottom:18px }
    .stSidebar .stButton>button { width:100%; }
    .stSidebar .stSelectbox, .stSidebar .stSlider, .stSidebar .stNumberInput { width:100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-header">Next-Word Deep Learning Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Interactive generator, loss curves and embedding visualizations</div>', unsafe_allow_html=True)


# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Model Controls")
    st.markdown("Select model parameters. The app will load the corresponding pre-trained model.")

    # Match assignment categories
    category_choice = st.selectbox(
        "Category",
        ("Category I (Sherlock Holmes)", "Category II (Sklearn Docs)"),
        index=0,
        help="Choose which dataset model to use."
    )

    # Match assignment options
    embed_size = st.selectbox("Embedding size", (32, 64), index=1, help="[32, 64]")
    
    num_hidden_layers = st.selectbox("No. of hidden layers", (1, 2), index=1, help="[1, 2]")

    activation_choice = st.selectbox(
        "Activation function",
        ("relu", "tanh"),
        index=0,
        help="[relu, tanh]"
    )
    
    # Note: Hidden Dims are fixed at 1024 as per the notebook/assignment
    st.info("Hidden dimension is fixed at 1024 (as per assignment spec).")

    st.markdown("---")
    st.header("Generation Settings")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="Controls randomness. Lower values are more predictable; higher values are more creative."
    )

    seed = st.number_input("Random Seed", value=42)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    st.markdown("---")
    # Load / reload controls
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
        st.session_state.model = None
        st.session_state.word_to_ix = None
        st.session_state.ix_to_word = None
        st.session_state.params = None
        st.session_state.loss_plot = None
        st.session_state.tsne_plot = None

    load_clicked = st.button("Load Model")
    unload_clicked = st.button("Unload Model")

    if unload_clicked:
        # Clear loaded model from session state
        st.session_state.loaded = False
        st.session_state.model = None
        st.session_state.word_to_ix = None
        st.session_state.ix_to_word = None
        st.session_state.params = None
        st.session_state.loss_plot = None
        st.session_state.tsne_plot = None
        st.success("Model unloaded from session.")

    if load_clicked:
        with st.spinner("Loading model..."):
            model, word_to_ix, ix_to_word, params, loss_plot, tsne_plot = load_resources(
                category_choice,
                activation_choice,
                embed_size,
                num_hidden_layers
            )

            if model:
                st.session_state.model = model
                st.session_state.word_to_ix = word_to_ix
                st.session_state.ix_to_word = ix_to_word
                st.session_state.params = params
                st.session_state.loss_plot = loss_plot
                st.session_state.tsne_plot = tsne_plot
                st.session_state.loaded = True
                st.success(f"Loaded model: {params['Model ID']}")
            else:
                st.session_state.loaded = False

# --- Main App Interface (with Tabs) ---
tab_generator, tab_loss, tab_tsne = st.tabs(["Generator", "Loss Curves", "TSNE Embeddings"])

with tab_generator:
    st.subheader("Text Generator")

    if not st.session_state.loaded:
        st.info("No model loaded. Select model options in the sidebar and click **Load Model**.")
    else:
        params = st.session_state.params
        model = st.session_state.model
        word_to_ix = st.session_state.word_to_ix
        ix_to_word = st.session_state.ix_to_word

        default_text = "sherlock holmes said it" if "cat1" in params["Model ID"] else "the model should be fit"
        input_text = st.text_area("Enter input context:", default_text)
        k_words = st.slider("How many next words?", 1, 100, 20)

        if st.button("Generate Next k Words"):
            if not input_text.strip():
                st.error("Please enter some starting text.")
            else:
                with st.spinner("Generating..."):
                    generated_output = generate_text(
                        model, word_to_ix, ix_to_word,
                        params["Context Size"], input_text,
                        k_words, temperature, params.get('Category', 'Category I')
                    )
                st.subheader("Generated sequence:")
                st.markdown(f"> **{input_text}** *{generated_output}*")

with tab_loss:
    st.subheader("Loss Curves")
    if not st.session_state.loaded:
        st.info("Load a model to view its loss curve.")
    else:
        if st.session_state.loss_plot:
            st.image(st.session_state.loss_plot, caption=f"Loss Curve for {st.session_state.params['Model ID']}")
        else:
            st.warning(f"Could not find loss plot: `loss_{st.session_state.params['Model ID']}.png`")

with tab_tsne:
    st.subheader("TSNE Embeddings")
    if not st.session_state.loaded:
        st.info("Load a model to view its t-SNE embeddings.")
    else:
        if st.session_state.tsne_plot:
            st.image(st.session_state.tsne_plot, caption=f"t-SNE Plot for {st.session_state.params['Model ID']}")
        else:
            st.warning(f"Could not find t-SNE plot: `tsne_{st.session_state.params['Model ID']}.png`")