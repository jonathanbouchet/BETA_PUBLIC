import os
import re
import openai
import streamlit as st
from st_pages import Page, show_pages, hide_pages
from utils import ai_bot, simple_questionnaire, small_questionnaire, full_questionnaire, \
    insurance_advisor, get_tokens, download_transcript, download_user_data
from firebase_admin import firestore
from datetime import datetime
import logging
from models import Tags0, extract_data
from langchain.chat_models import ChatOpenAI
from app import VERSION


def _get_logger(name):
    loglevel = logging.INFO
    l = logging.getLogger(name)
    if not getattr(l, 'handler_set', None):
        # print("setting new logger")
        l.setLevel(loglevel)
        h = logging.FileHandler(filename=name + ".log")
        f = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        h.setFormatter(f)
        l.addHandler(h)
        l.setLevel(loglevel)
        l.handler_set = True
        l.propagate = False
    return l


log_name = "chat"
# chat_logger = _get_logger(log_name)


def simple_chat():
    # name = "chat.log"
    # if not os.path.exists(name):
    #     global chat_logger
    #     chat_logger = _get_logger(log_name)
    # else:
    #     print(f"chat.log: {chat_logger}")

    # check a logout has not been issued already
    if not st.session_state["logout"] or st.session_state["authentication_status"] is True:

        user_details = Tags0()

        # initialization
        if 'messages_chat' not in st.session_state:
            st.session_state["messages_chat"] = []
        if 'messages_QA' not in st.session_state:
            st.session_state["messages_QA"] = []
        if 'total_tokens' not in st.session_state:
            st.session_state.total_tokens = 0
        if 'pydantic_life_insurance_model' not in st.session_state:
            st.session_state.pydantic_life_insurance_model = user_details

        st.sidebar.markdown(f"""version {VERSION}""")

        if st.sidebar.button("Logout", help="quit session"):
            # print("logout starts")
            db = firestore.client()  # log in table
            # print(f"db: {db}")
            obj = {"name": st.session_state["name"],
                   "username": st.session_state["username"],
                   "login_connection_time": st.session_state["login_connection_time"],
                   "messages_chat": st.session_state["messages_chat"],
                   "messages_QA": st.session_state["messages_QA"],
                   "pydantic_life_insurance_model": st.session_state.pydantic_life_insurance_model.dict(),
                   "created_at": datetime.now()}
            # print(f"writing to database:{obj}")
            doc_ref = db.collection(u'users_app').document()  # create a new document.ID
            doc_ref.set(obj)  # add obj to collection
            db.close()

            st.empty()  # clear page
            # print(f"in logout from chatbot:{st.session_state}")
            # for key in st.session_state.keys():
            #     print(st.session_state[key])

            # st.session_state.end_conversation = time.time()
            # total_time = int(st.session_state.end_conversation) - int(st.session_state.start_conversation)
            # print(f"total tokens spent :{st.session_state.total_tokens}, total duration of chat:{total_time}")
            # print(f"total tokens spent :{st.session_state.total_tokens}")
            st.session_state["logout"] = True
            st.session_state["name"] = None
            st.session_state["username"] = None
            st.session_state["authentication_status"] = None
            st.session_state["login_connection_time"] = None
            st.session_state['messages_chat'] = []
            st.session_state['messages_QA'] = []
            st.session_state.total_tokens = 0
            # when loggin out, re-initialize the pydantic model to default
            # reason is after a logout, a login can restart but then the pydantic model is None
            st.session_state.pydantic_life_insurance_model = Tags0()
            # print(st.session_state)
            return st.session_state["logout"]

        st.title("Reflexive AI")
        st.header("Virtual Insurance Agent Accelerator")

        if st.sidebar.button("Download transcripts", help="download the chat history with the agent"):
            download_transcript(log_name)

        model = st.sidebar.selectbox(
            label=":blue[MODEL]",
            options=["gpt-3.5-turbo",
                     "gpt-4"],
            help="openAI model(GPT-4 recommended)")

        systemprompt = st.sidebar.selectbox(
            label=":blue[AI Persona]",
            options=["Simple AI Assistant",
                     "mini questionnaire",
                     "full questionnaire"],
            help="AI agent type")

        show_tokens = st.sidebar.radio(label=":blue[Display tokens]",
                                       options=('Yes', 'No'),
                                       help="show the number of tokens used by the LLM")

        # Set API key if not yet
        openai_api_key = st.sidebar.text_input(
            ":blue[API-KEY]",
            placeholder="Paste your OpenAI API key here",
            type="password",
            help="format is st-***")

        st.sidebar.markdown("How to:")
        st.sidebar.markdown("1. Choose model")
        st.sidebar.markdown("2. Choose the type of agent")
        st.sidebar.markdown("3. Enter your openAI api key")

        if openai_api_key:

            openai.api_key = openai_api_key

            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = model

            # initialization of the session_state
            if len(st.session_state["messages_chat"]) == 0:
                # print(f"system prompt from drop down: {systemprompt}")
                if systemprompt == "full questionnaire":
                    template = full_questionnaire()
                elif systemprompt == "mini questionnaire":
                    template = small_questionnaire()
                    # template = simple_questionnaire()
                elif systemprompt == "Insurance Advisor":
                    template = insurance_advisor()
                else:
                    template = ai_bot()
                # print(f"template chosen : {template}")
                st.session_state['messages_chat'] = [
                    {"role": "system", "content": template}
                ]
                # greetings message
                greetings = {
                    "role": "assistant",
                    "content": "What can I do for you ?"
                }
                st.session_state["messages_chat"].append(greetings)

            # display chat messages from history on app rerun
            for message in st.session_state.messages_chat:
                # don't print the system content
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # react to user input
            if prompt := st.chat_input("Hello"):
                # display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                    if show_tokens == "Yes":
                        prompt_tokens = get_tokens(prompt, model)
                        st.session_state.total_tokens += prompt_tokens
                        tokens_count = st.empty()
                        tokens_count.caption(f"""query used {prompt_tokens} tokens """)
                # print(f"extracted data: {st.session_state.pydantic_life_insurance_model}")
                # add user message to chat history
                st.session_state["messages_chat"].append({"role": "user", "content": prompt})
                # logging.info(f"[user]:{prompt}, # tokens:{prompt}")
                # print(f"your query :{prompt}")
                chat_logger.info(f"[user]:{prompt}")

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for response in openai.ChatCompletion.create(
                            model=st.session_state["openai_model"],
                            temperature=0,
                            stream=True,
                            messages=[
                                {"role": m["role"], "content": m["content"]} for m in st.session_state["messages_chat"]]
                    ):
                        full_response += response.choices[0].delta.get("content", "")
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    if show_tokens == "Yes":
                        assistant_tokens = get_tokens(full_response, model)
                        st.session_state.total_tokens += assistant_tokens
                        tokens_count = st.empty()
                        tokens_count.caption(f"""assistant used {assistant_tokens} tokens """)
                # add assistant response to chat history
                st.session_state["messages_chat"].append({"role": "assistant", "content": full_response})
                # logging.info(f"[assistant]:{prompt}, # tokens:{full_response}")
                chat_logger.info(f"[assistant]:{full_response}")
                # check for user response right at the beginning by looking at any message non-empty
                if len(re.findall(r"summa[a-zA-Z]", full_response, re.I)) > 0:
                    current = extract_data(full_response, openai_api_key)
                    st.session_state.pydantic_life_insurance_model = current
                    download_user_data(st.session_state)


# Run the main page
if __name__ == "__main__":
    print(f"in simple_chat.py, starting to run simple_chat")
    # name = "chat.log"
    # if not os.path.exists(name):
    #     chat_logger = _get_logger(log_name)

    chat_logger = _get_logger(log_name)
    simple_chat()
