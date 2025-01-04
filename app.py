import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import re
from transformers import pipeline

# Initialize EasyOCR reader (specifying English as the language)
reader = easyocr.Reader(['en'])

# Streamlit app title
st.title("Patient Name, Date of Birth, and Disease Extractor using Transformers NER")

# File uploader for images (TIFF, JPG, PNG)
uploaded_file = st.file_uploader("Choose an image (TIFF, JPG, PNG)", type=['tiff', 'jpg', 'png'])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)

    # Convert the image to a NumPy array
    img_np = np.array(img)

    # Perform OCR using EasyOCR
    extracted_text = reader.readtext(img_np, detail=0)  # detail=0 returns only text
    extracted_text = " ".join(extracted_text)  # Combine list of text into a single string

    # Display extracted text in Streamlit
    st.subheader("Extracted Text")
    st.text(extracted_text)

    # Initialize the NER model pipeline using Hugging Face's transformers
    ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

    # Apply general NER on the extracted text (to get Patient Name and Date of Birth)
    ner_results = ner_model(extracted_text)

    # Initialize the BioBERT model pipeline for medical term recognition
    bio_ner_model = pipeline("ner", model="dmis-lab/biobert-base-cased-v1.1", grouped_entities=True)

    # Apply BioBERT NER on the extracted text (to get disease/medical terms)
    bio_ner_results = bio_ner_model(extracted_text)

    # Define a function to extract patient information from the NER results
    def extract_patient_info_bert(ner_results):
        patient_name = None
        dob = None

        # Iterate through the NER results to find PERSON and DATE entities
        for entity in ner_results:
            if entity['entity_group'] == 'PER':
                patient_name = entity['word']
            elif entity['entity_group'] == 'DATE':
                dob = entity['word']

        # If DOB is not found by NER, attempt regex extraction
        if not dob:
            dob_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b', extracted_text)
            if dob_match:
                dob = dob_match.group(0)

        # Return results
        return patient_name, dob

    # Define a function to extract disease information from BioBERT NER results
    def extract_disease_info_biobert(bio_ner_results):
        diseases = []

        # Iterate through the BioBERT NER results to find medical condition entities
        for entity in bio_ner_results:
            if entity['entity_group'] == 'DISEASE':
                diseases.append(entity['word'])

        # Return results
        return diseases

    # Extract patient info using the general NER model
    patient_name, dob = extract_patient_info_bert(ner_results)

    # Extract disease info using BioBERT NER
    diseases = extract_disease_info_biobert(bio_ner_results)

    # Display the results in Streamlit
    st.subheader("Extracted Information")

    if patient_name:
        st.write(f"**Patient Name:** {patient_name}")
    else:
        st.write("**Patient Name:** Not found")

    if dob:
        st.write(f"**Date of Birth:** {dob}")
    else:
        st.write("**Date of Birth:** Not found")

    if diseases:
        st.write(f"**Diseases/Conditions:** {', '.join(diseases)}")
    else:
        st.write("**Diseases/Conditions:** Not found")
