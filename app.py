import streamlit as st
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import json

@st.cache_resource
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="ahmeterdempmk/Gemma2-2b-E-Commerce-Tuned",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

st.title(" E-Commerce Text Generation")

text = st.text_area("Enter product information:", placeholder="Example: Rosehip Marmalade, keep it cold")

if st.button("Apply"):
    if text:
        with st.spinner("Generating response..."):
            prompt = f"""
            You are extracting product title and description from given text and rewriting the description and enhancing it when necessary.
            Always give response in the user's input language.
            Always answer in the given json format. Do not use any other keywords. Do not make up anything.
            Explanations should contain at least five sentences each.

            Json Format:
            {{
            "title": "<title of the product>",
            "description": "<description of the product>"
            }}

            Examples:

            Product Information: Rosehip Marmalade, keep it cold
            Answer: {{"title": "Rosehip Marmalade", "description": "You should store this delicisious roseship marmelade in cold conditions. You can use it in your breakfasts and meals."}}

            Product Information: Blackberry jam spoils in the heat
            Answer: {{"title": "Blackberry Jam", "description": "Please store it in cold conditions. Recommended to be consumed at breakfast. Very sweet."}}

            Now answer this:
            Product Information: {text}
            """
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

            output = model.generate(**inputs, max_new_tokens=128)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            answer_start = response.find("Now answer this:") + len("Now answer this:")
            answer = response[answer_start:].strip()

            json_start = answer.find("{")
            json_end = answer.find("}") + 1 
            json_response = answer[json_start:json_end].strip()

            st.subheader("JSON Format Answer:")
            st.text(f"{json_response}")

            try:
                json_data = json.loads(json_response)
                title = json_data["title"]
                description = json_data["description"]

                st.subheader("Product Title:")
                st.text(title)

                st.subheader("Product Description:")
                st.text(description)

            except json.JSONDecodeError:
                st.error("An error has occurred! Please try again.")
    else:
        st.warning("Please enter product information.")