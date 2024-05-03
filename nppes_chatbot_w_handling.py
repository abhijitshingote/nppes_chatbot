import langchain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType, AgentExecutor, initialize_agent, create_tool_calling_agent
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from pprint import pprint
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from query_embedding import get_specialty_matches_using_embeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os
langchain.debug = False

db = SQLDatabase.from_uri(f"postgresql://mdx:{os.environ['mdxpass1']}@venom.des.mdx.med:5432/bi_smrf", schema='nppes_chatbot')
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def get_extract_specialty_chain():
    extract_specialty_template = """If the given question contains a specialist doctor, provide just the specialty of the doctor from the given question, end \
    the queries with ’**’. Question: \
    {input} Answer:"""
    extract_specialty_prompt = ChatPromptTemplate.from_template(extract_specialty_template)

    def _parse(text):
        return text.strip('"').strip("**")

    extract_specialty_runnable = extract_specialty_prompt | llm | StrOutputParser() | _parse
    extract_specialty_chain = RunnablePassthrough.assign(original_specialty=extract_specialty_runnable)
    return extract_specialty_chain


def get_new_specialty_chain():
    new_specialty_runnable = itemgetter('original_specialty') | RunnableLambda(get_specialty_matches_using_embeddings)
    new_specialty_chain = RunnablePassthrough.assign(new_specialty=new_specialty_runnable)
    return new_specialty_chain


def get_rephrase_question_chain():
    rephrase_template = """
    Rephrase the question below to make it suitable for querying. In the rephrased question use the below specialties exactly as is.
    Question: {input}
    Specialties: {new_specialty}
    """
    rephrase_prompt = PromptTemplate.from_template(rephrase_template)
    rephrase_question_runnable = rephrase_prompt | llm | StrOutputParser()
    rephrase_question_chain = RunnablePassthrough.assign(final_question=rephrase_question_runnable)
    return rephrase_question_chain


def get_sql_query_chain():
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
    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    chain = write_query | execute_query
    chain = RunnablePassthrough.assign(write_query=chain)
    return chain


from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(
    llm, db=db,
    agent_type="openai-tools",
    verbose=True,
    top_k=5,
    prefix="Ensure that you use specialties from the question as is in the SQL query. If specialties are involved use LIKE operator as much as you can. always only query for 5 records.")


def get_overall_chain():
    # return get_extract_specialty_chain() | get_new_specialty_chain() | get_rephrase_question_chain() | RunnablePassthrough.assign(question=itemgetter('final_question')) | get_sql_query_chain()
    return get_extract_specialty_chain() | get_new_specialty_chain() | get_rephrase_question_chain() | agent_executor


def get_determine_route_chain():
    template = """Examine the question below and answer if the question is related to doctors.
    Answer in either yes or no.

    Question: {input}
    """

    prompt = PromptTemplate.from_template(template)
    determine_route = prompt | llm | StrOutputParser()
    determine_route_chain = RunnablePassthrough.assign(is_provider=determine_route)
    return determine_route_chain


def get_non_provider_chain():
    non_provider_runnable = PromptTemplate.from_template(
        """Answer the question below briefly.\nQuestion:{input}""") | llm | StrOutputParser()
    non_provider_chain = RunnablePassthrough.assign(output=non_provider_runnable)
    return non_provider_chain


def route(info):
    info['is_provider'] = info['is_provider'].lower()
    if info['is_provider'] == 'no':
        return get_non_provider_chain()
    else:
        return get_overall_chain()


full_chain = get_determine_route_chain() | RunnableLambda(route)

if __name__ == '__main__':
    result = full_chain.invoke(
        {'input': 'Are you friendlier than the terminator?'})
    print(result)
