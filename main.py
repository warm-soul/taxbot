import streamlit as st
from pinecone 
from openai import OpenAI
from typing import List
from deep_translator import GoogleTranslator
import time
import math

# Debug mode
DEBUG = st.sidebar.checkbox("Debug Mode", False)

# System prompt definition
system_prompt = """You are an authoritative expert on the Gujrat Property Tax Law.
Your responses should be:
1. Comprehensive and detailed
2. Include step-by-step procedures when applicable
3. Quote relevant sections directly from the Tax Act
4. Provide specific references (section numbers, chapters, and page numbers)
5. Break down complex processes into numbered steps
6. Include any relevant timelines or deadlines
7. Mention any prerequisites or requirements
8. Highlight important caveats or exceptions

For every fact or statement, include a reference to the source document and page number in this format:
[Source: Document_Name, Page X]
the page number is stored in the vector index
Always structure your responses in a clear, organized manner using:
- Bullet points for lists
- Numbered steps for procedures
- Bold text for important points
- Separate sections with clear headings"""

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("gujtaxlaw")

def chunk_text(text: str, max_tokens: int = 3000) -> List[str]:
    """Split text into chunks of approximately max_tokens."""
    chars_per_chunk = max_tokens * 4
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length > chars_per_chunk and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding for the input text using OpenAI's embedding model."""
    if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in text):
        try:
            text = translate_text(text, 'en')
        except Exception as e:
            if DEBUG:
                st.error(f"Translation error in get_embedding: {str(e)}")
            pass

    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def search_pinecone(query: str, k: int = 3):
    """Search Pinecone index with embedded query."""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    return results

def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using deep-translator with error handling."""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                max_chunk_size = 4000
                if len(text) > max_chunk_size:
                    chunks = [text[i:i+max_chunk_size]
                            for i in range(0, len(text), max_chunk_size)]
                    translated_chunks = []
                    for chunk in chunks:
                        translated_chunk = translator.translate(chunk)
                        translated_chunks.append(translated_chunk)
                    return ' '.join(translated_chunks)
                else:
                    return translator.translate(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)

    except Exception as e:
        raise Exception(f"Translation failed: {str(e)}")

def generate_response(query: str, context: str, system_prompt: str):
    """Generate response using OpenAI with context and system prompt, handling long contexts."""
    context_chunks = chunk_text(context)
    response_parts = []

    for i, chunk in enumerate(context_chunks):
        if i == 0:
            chunk_prompt = f"Part 1/{len(context_chunks)} of the context. Answer based on this and upcoming parts: {query}"
        else:
            chunk_prompt = f"Part {i+1}/{len(context_chunks)} of the context. Continue building the answer with this additional information."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {chunk}\n\nQuestion: {chunk_prompt}"}
        ]

        if i > 0:
            messages.append({"role": "assistant", "content": "Previous parts of the response: " + " ".join(response_parts)})

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            response_parts.append(response.choices[0].message.content)
            time.sleep(1)

        except Exception as e:
            if DEBUG:
                st.error(f"Error generating response for chunk {i+1}: {str(e)}")
            continue

    final_response = "\n\n".join(response_parts)

    if len(response_parts) > 1:
        try:
            summary_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original question: {query}\n\nPlease provide a concise, coherent summary of the following detailed response: {final_response}"}
            ]

            summary_response = client.chat.completions.create(
                model="gpt-4",
                messages=summary_messages,
                temperature=0.7,
                max_tokens=1000
            )
            final_response = summary_response.choices[0].message.content
        except Exception as e:
            if DEBUG:
                st.error(f"Error generating summary: {str(e)}")

    # Translate the final response to Gujarati
    try:
        final_response = translate_text(final_response, 'gu')
    except Exception as e:
        if DEBUG:
            st.error(f"Translation error: {str(e)}")

    return final_response

# Streamlit UI
st.title("ગુજરાત કર કાયદો સહાયક")
st.write("કર કાયદા વિશે કોઈપણ પ્રશ્ન પૂછો")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("તમે શું જાણવા માંગો છો?"):
    # Display original query
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Translate to English if in Gujarati
    if any(ord(c) >= 0x0A80 and ord(c) <= 0x0AFF for c in prompt):
        with st.spinner('પ્રશ્ન પર પ્રક્રિયા કરી રહ્યા છીએ...'):
            prompt = translate_text(prompt, 'en')

    # Search and generate response
    with st.spinner('જવાબ તૈયાર કરી રહ્યા છીએ...'):
        search_results = search_pinecone(prompt)
        context = "\n".join([result.metadata.get('text', '') for result in search_results.matches])
        response = generate_response(prompt, context, system_prompt)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("વિશે")
    st.write("""
    આ ચેટબોટ ગુજરાત કર કાયદા અને અમદાવાદ મ્યુનિસિપલ કોર્પોરેશન વિશે માહિતી પ્રદાન કરે છે.
    """)
    st.write("""
    ભાષા સુવિધાઓ:
    - તમે ગુજરાતી અથવા અંગ્રેજીમાં પ્રશ્નો પૂછી શકો છો
    - તમને જવાબ ગુજરાતીમાં મળશે
    """)
