import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import AutoPeftModelForCausalLM, PeftModel

model_name = "Prabhuraj/Fine_Tuned_Llama_2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoPeftModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    live=True,
    title="17th Century French LLM",
    description="Enter a prompt in French, and the model will generate text in a 17th-century style.",
)

iface.launch()
