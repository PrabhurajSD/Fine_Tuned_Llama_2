import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftConfig

# Load PEFT model
model_name = "Prabhuraj/Fine_Tuned_Llama_2"  # Replace with your actual model name
peft_config = PeftConfig.from_pretrained(model_name)
model = AutoPeftModelForCausalLM.from_pretrained(model_name, config=peft_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_text(prompt):
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    output_tokens = model.generate(input_tokens, max_length=50, num_return_sequences=1)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

# Streamlit UI
st.title("PEFT Model Streamlit App")

user_prompt = st.text_input("Enter a prompt:")
if user_prompt:
    generated_text = generate_text(user_prompt)
    st.text("Generated Text:")
    st.write(generated_text)