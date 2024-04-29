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
from query_embedding import get_embedding_matches
langchain.debug = False
# call_function()

load_dotenv()
# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db = SQLDatabase.from_uri("postgresql://mdx:des!avengers@venom.des.mdx.med:5432/bi_smrf",schema='test_abi')
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM provider_data_nppes LIMIT 10;"))

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# chain = create_sql_query_chain(llm, db)
# response = chain.invoke({"question": "How many employees are there"})
# print(response)

template = '''You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
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

chain = write_query | execute_query


# answer_prompt = PromptTemplate.from_template(
#     """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

# Question: {question}
# SQL Query: {query}
# SQL Result: {result}
# Answer: """
# )

# answer = answer_prompt | llm | StrOutputParser()
# chain = (
#     RunnablePassthrough.assign(query=write_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | answer
# )

question="Find heart disease doctors for adults "
response=chain.invoke({"question": question})
print(response)

embedding_matches=get_embedding_matches(question)
print('Embedding Matches:\n',embedding_matches)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, and SQL results, answer the user question. If available list the NPI, name and specialty.

Question: {question}
SQL Result1: {response}
SQL Result2: {embedding_matches}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
print(answer.invoke({'question':question,'response':response,'embedding_matches':embedding_matches}))