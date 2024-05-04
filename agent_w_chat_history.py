from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
# from langchain.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.utilities import SQLDatabase
import os

# prompt = hub.pull("hwchase17/ openai-functions-agent")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
memory = ChatMessageHistory(session_id="test-session")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
db = SQLDatabase.from_uri(f"postgresql://mdx:{os.environ['mdxpass1']}@venom.des.mdx.med:5432/bi_smrf", schema='nppes_chatbot')

# wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
# search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

#
# tools = [search, ]
# agent = create_openai_functions_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools)

from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(
    llm, db=db,
    agent_type="openai-tools",
    verbose=True,
    top_k=5,
    prefix="Ensure that you use specialties from the question as is in the SQL query. If specialties are involved use LIKE operator as much as you can. always only query for 5 records.")


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
        print('\n\n')
        res=agent_with_chat_history.invoke({"input": input("Question:")},
        config={"configurable": {"session_id": "<foo>"}},
        )
        print(res)