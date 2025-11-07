import streamlit as st

st.set_page_config(page_title="Bib Processor")

st.title("Bib File Processor")
st.write("Upload a .bib file to process it.")

uploaded = st.file_uploader("Choose a .bib file", type=["bib"])

if uploaded is not None:
    # Read the content as text
    content = uploaded.read().decode("utf-8")
    
    # Do something with it (replace this with your logic)
    st.write("File loaded successfully!")
    st.text_area("Preview", content[:1000], height=300)
    
    if st.button("Process"):
        # Example: call your processing function
        # result = process_bib(content)
        result = f"Found {content.count('@')} entries in the .bib file."
        st.success(result)
