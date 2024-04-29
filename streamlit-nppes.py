from openai import OpenAI
import streamlit as st
import os
import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
import langchain
langchain.debug = True

load_dotenv()
db = SQLDatabase.from_uri("postgresql://mdx:des!avengers@venom.des.mdx.med:5432/bi_smrf",schema='test_abi')
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

template = '''You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. Always include NPI, provider name and address in the columns. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:
{table_info}

Question: {input}'''
postgresprompt = PromptTemplate.from_template(template)
write_query = create_sql_query_chain(llm, db,prompt=postgresprompt)

execute_query = QuerySQLDataBaseTool(db=db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
     If the answer contains details about a provider, include NPI, provider name, specialty and address of provider.
     Include atleast 5 results where possible.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

# response=chain.invoke({"question": "Find NPIs with names that specialize in Pediatrics in New Jersey"})


st.title("💬 Q&A with National Provider Registry(NPPES)")
st.caption("I am slow coz i am running on Abi's laptop")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me something like : Help me find pediatricians near charlotte"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response=chain.invoke({"question": prompt})
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response
    print(response)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)