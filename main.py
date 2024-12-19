import streamlit as st
import joblib
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Function to load the model and tokenizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_numpy.pkl")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Load the model and tokenizer
model, tokenizer = load_model()

# Configure Streamlit page
st.set_page_config(
    page_title="Translate Aceh to Indonesia",
    page_icon="translator-icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Supported languages
Languages = {"aceh": "ace", "indonesia": "ind"}

# Translator app
st.title("Language Translator :balloon:")

# User input for text
aceh_text = st.text_area("Enter text:", height=None, max_chars=None, key=None, help="Enter your text here")

# Language selection
option1 = st.selectbox("Input language", ("aceh",))
option2 = st.selectbox("Output language", ("indonesia",))

# Fetch language codes
value1 = Languages[option1]
value2 = Languages[option2]

if st.button("Translate Sentence"):
    if aceh_text.strip() == "":
        st.warning("Please **enter text** for translation.")
    else:
        # Prepare input for translation
        input_text = f"translate Aceh to Indonesian: {aceh_text}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate translation
        outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display result
        st.write("### Translation:")
        st.info(translated_text)
        st.success("Translation is **successfully** completed!")
        st.balloons()

else:
    st.info("Click the **Translate Sentence** button to get the translation.")
