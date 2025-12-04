import streamlit as st
from rag import rag_query

# Better Urdu font styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    .stMarkdown, .stTextInput > div > div > input, p, div {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Mehr Nastaliq Web', serif !important;
        font-size: 18px !important;
        line-height: 2 !important;
    }
    
    h1, h2, h3 {
        font-family: 'Noto Nastaliq Urdu', serif !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Bang-e-Dara RAG — Local Llama")

query = st.text_input("Ask something about Iqbal or any poem explanation:")

if st.button("Run"):
    answer = rag_query(query)
    st.write(answer)
