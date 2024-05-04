import streamlit as st
from nppes_chatbot_w_handling import full_chain

st.title("💬 Q&A with National Provider Registry(NPPES)")
st.caption("I am slow coz i am running on Abi's laptop")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me something like : Help me find pediatricians near charlotte"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response=full_chain.invoke({"input": prompt},config={"configurable": {"session_id": "<foo>"}})
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response['output']
    print(response)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)