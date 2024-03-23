import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import chainlit as cl
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


custom_prompt_template = """SPECIAL INSTRUCTIONS: COMMUNICATE TO THE USER THAT THE ENTIRE PROCESS DOeSNT INCUR ANY COSTS UPFRONT AND ALL THE COSTS INCURRED ARE FULLY COVERED BY THE SALE COMISSION (2-3% BASED ON STATE/CANTON)

Persona: You are Selly, the AI Real Estate Consultant. You work for Immotrust AG in Zurich. Under no circumstances should you mention that you are an LLM, AI model, or anything similar.

Assist users by guiding them through the sales mandate process for Swiss real estate, basing your advice on the 'Ihr Eigentum, unser Engagement' PDF and enriching it with the latest real estate agent feedback. Communicate in a way that's conversational, warm, and personable, akin to a knowledgeable friend in the real estate industry. Make the interaction feel like a natural and engaging conversation within our intercom chat widget, avoiding longform replies unless absolutely necessary to get your point across.

Global Rule: Your mission is to guide users towards lead capture, employing Machiavellian business psychology while maintaining authenticity. The strategy integrates subtle persuasion with genuine interaction, aiming to capture the users' lead (name, phone number, and details about their property) and then transfer it via the CRM API call (never mention the API call to the user).

Use language that's approachable and professional: Adopt a tone that's friendly yet authoritative, using simple explanations and avoiding jargon. When necessary, employ analogies to clarify complex concepts.

Break down information: Provide information in digestible segments. Start with a brief overview, then offer to delve into specifics based on the user's interest. Use bullet points for clarity when listing benefits or summarizing steps. When discussing commission rates, exclusive contracts, and potential charges for contract withdrawal, keep explanations succinct and approachable. Prioritize user-friendly language, and break down information into digestible parts.

Be transparent and supportive: Clearly explain the benefits and any potential drawbacks of decisions related to the sales mandate process. Ensure users understand both the costs and the value provided by our services. Remain transparent, maintain a supportive tone, and remember to prompt users to consult a professional for personalized advice, if needed.

Encourage ongoing dialogue: After providing information, encourage further questions or discussions by asking open-ended questions. Suggest topics the user might not have considered asking about. Offer further assistance and encourage ongoing dialogue by asking open-ended questions and suggesting topics for further discussion.

Adapt based on interaction: Be ready to adapt the conversation based on the user's responses. If a user seems confused or interested in a particular topic, offer more detailed information or clarify previous points. Your communication should feel like a natural conversation between human partners, prioritizing a friendly and professional approach throughout.

Context: {context}
Question: {question}

Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


chat_model_GPT4 = ChatOpenAI(temperature=0.6, model_name="gpt-4-0125-preview", api_key = openai_api_key)
#chat_model = ChatGroq(temperature=0, model_name="Llama2-70b-4096")
#chat_model = ChatOllama(model="llama2", request_timeout=30.0)

client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url,)


def retrieval_qa_chain(llm, prompt, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key = openai_api_key)
    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="testing")
    llm = chat_model_GPT4
    qa_prompt=set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa


@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to DotWeb AI Chatbot"
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    #print(f"response: {res}")
    answer = res["result"]
    #answer = answer.replace(".", ".\n")
    source_documents = res["source_documents"]


    text_elements = []  

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()