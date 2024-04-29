from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType,AgentExecutor,initialize_agent, create_tool_calling_agent
load_dotenv()
template="""
================================ System Message ================================

You are a helpful assistant

============================= Messages Placeholder =============================


================================ Human Message =================================


============================= Messages Placeholder =============================


"""
prompt = PromptTemplate.from_template(template)
wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

tools=[search.run]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Construct the tool calling agent
# agent = create_tool_calling_agent(llm, tools, prompt)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
agent_chain.run("Obama")
# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
res=agent_executor.invoke(
    {
        "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
    }
)

print(res)
# results=search.run("Obama")

# print(results)

# Initialize the LLM from OpenAI (you will need an API key)
# llm = OpenAI(api_key="your_openai_api_key")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# # Initialize the search engine
# search_engine = GoogleSearch(api_key="your_google_custom_search_api_key")

# # Create an LLM with search capabilities
# llm_with_search = LLMWithSearch(llm=llm, search=search_engine)

# # Function to ask a question and get an answer
# def ask_question(question):
#     chain_input = ChainInput(prompt=question)
#     answer = llm_with_search.run(chain_input)
#     return answer

# # Example usage
# question = "What is the tallest mountain in the world?"
# answer = ask_question(question)
# print("Answer:", answer)
