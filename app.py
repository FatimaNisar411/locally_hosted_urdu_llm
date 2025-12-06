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

# -------------------------------------------------
# SESSION STORAGE for conversation / RAG debugging
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------
# User Input
# ---------------------------
query = st.text_input("Ask something about Iqbal or any poem explanation:")

if st.button("Run") and query:
    
    # 1) Run full RAG pipeline
    answer, retrieved_context = rag_query(query)  
    # NOTE: rag_query must return BOTH answer and context

    # 2) Store the result in session memory
    st.session_state.history.append({
        "query": query,
        "context": retrieved_context,
        "answer": answer
    })

    # 3) Display answer
    st.markdown("### 📌 جواب")
    st.write(answer)


# -------------------------------------------------
# Display previous conversation + RAG context
# -------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🧠 پچھلی گفتگو اور Context (Debug View)")

    for i, item in enumerate(st.session_state.history):
        with st.expander(f"Step {i+1}: {item['query']}"):
            st.write("**Retrieved Context:**")
            st.write(item["context"])
            st.write("**Generated Answer:**")
            st.write(item["answer"])