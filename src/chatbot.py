import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

import gradio as gr

# === ENV SETUP ===
load_dotenv()

# === VECTOR DB ===
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="vectorstore", embedding_function=embedding)

# === MEMORY ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === SYSTEM PROMPT ===
system_prompt = (
    "You are a helpful assistant trained to provide insights based solely on recent posts "
    "from the CryptoMoonShots subreddit. You have access to several Reddit posts about cryptocurrencies. "
    "Your task is to use these posts to answer user questions.\n\n"
    "If a question cannot be answered from the available Reddit content, reply with something like "
    "\"I couldn't find anything relevant in the recent Reddit discussions.\"\n\n"
    "Avoid guessing or making assumptions. Always base your answer on actual retrieved content."
)


# === CUSTOM PROMPT SETUP ===
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
])


# === LLM SETUP ===
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# === CHAT HANDLER ===
def user_chat(message, history):
    result = qa_chain.invoke({"question": message})
    history.append((message, result["answer"]))
    return history, ""

# === DARK THEME ===
dark_theme = gr.themes.Base().set(
    body_text_color="#ffffff",
    background_fill_primary="#0f1117",
    background_fill_secondary="#1c1f26",
    block_background_fill="#1c1f26",
    block_title_text_color="#ffffff",
    border_color_primary="#2c2f36"
)

# === GRADIO UI ===
with gr.Blocks(
    theme=dark_theme,
    css="""
    .textbox-input textarea {
        background-color: black !important;
        color: white !important;
        border: 1px solid #444 !important;
    }
        /* User message */
    .message.user {
        background-color: #1f1f1f !important;
        color: white !important;
    }
    /* Bot message */
    .message.bot {
        background-color: #2a2a2a !important;
        color: white !important;
    }

    """
) as demo:
    gr.Markdown("## 🌑 Reddit RAG Chatbot")
    gr.Markdown("_Hi! I'm here to help you understand what cryptocurrencies Redditors are talking about._")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(
        placeholder="Ask me about trending coins, low caps, etc...",
        elem_classes=["textbox-input"]
    )

    msg.submit(user_chat, [msg, chatbot], [chatbot, msg])



demo.launch(inbrowser=True)
