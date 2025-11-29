import argparse
import asyncio
from typing import List, Dict, Any
from generators.math_problem_generator import get_problems
from tools.search import google_search_tool
import weave
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory.buffer import ConversationBufferMemory
from config import (
    AGENT_TEMPERATURE,
    MAX_TOKENS,
    OLLAMA_MODEL,
    REPEAT_PENALTY,
    TOP_K,
    TOP_P,
)
from tools.calculator import CalculatorTool
from helpers.evaluation import (
    get_calculation_evaluation,
)
from helpers.logging_helper import trace_logger
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
import uuid

# Configure Ollama LLM
llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=AGENT_TEMPERATURE,
    num_predict=MAX_TOKENS,
    top_k=TOP_K,
    top_p=TOP_P,
    repeat_penalty=REPEAT_PENALTY,
    stop=["\nObservation:"],
)

# Define tools
tools = [CalculatorTool()]

# Create prompt template
prompt = PromptTemplate.from_template(
    """
You are a helpful AI assistant. Use the following tools to answer the question.
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
"""
)


# Create agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)


# Multi-agent system using langgraph
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_agent: str


# Google Search Agent
class GoogleSearchAgent:
    def __init__(self, model_name: str = "google_search_agent"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=AGENT_TEMPERATURE,
            num_predict=MAX_TOKENS,
            top_k=TOP_K,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
        )

    def get_search_response(self, question: str) -> str:
        """Get search results for the question"""
        try:
            # First, let's try to determine if this is a search question
            search_prompt = f"""
            You are a search assistant. Determine if the following question requires searching the web for information.
            
            Question: {question}
            
            Answer with only "YES" if it requires web search, or "NO" if it doesn't.
            """

            response = self.llm.invoke([HumanMessage(content=search_prompt)])
            needs_search = response.content.strip().upper()

            if needs_search == "YES":
                # Perform the search
                search_results = google_search_tool.search(question, num_results=3)
                return search_results
            else:
                return "This question doesn't require web search."

        except Exception as e:
            return f"Error in search agent: {str(e)}"


class MathFormatterAgent:
    def __init__(self, model_name: str = "math_formatter_agent"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=AGENT_TEMPERATURE,
            num_predict=MAX_TOKENS,
            top_k=TOP_K,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
        )

    def get_math_response(self, question: str) -> str:
        """Convert question into mathematical expression"""
        try:
            # Create a more direct prompt for mathematical reasoning
            direct_prompt = f"""
            You are a mathematical formatting expert. Your task is to convert the given question into a precise mathematical expression.

            Question: {question}

            Instructions:
            1. Identify all mathematical operations and variables mentioned in the question
            2. Translate the question into a mathematical expression using appropriate mathematical notation
            3. Do NOT include the = symbol or any equality signs
            4. Use standard mathematical symbols and notation
            5. Ensure the expression is mathematically valid and complete

            Examples:
            - "What is 5 plus 3?" → 5 + 3
            - "Calculate the area of a circle with radius 4" → π × 4²
            - "Find the sum of 2x and 3y" → 2x + 3y

            Mathematical Expression:
            """

            response = self.llm.invoke([HumanMessage(content=direct_prompt)])
            if isinstance(response.content, str):
                return response.content.replace("\n", "")
        except Exception as e:
            return f"Error solving problem: {str(e)}"

        return "No response"


class MultiAgentSystem:
    def __init__(self):
        self.math_formatter_agent = MathFormatterAgent()
        self.calculator_agent = agent_executor
        self.google_search_agent = GoogleSearchAgent()
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=AGENT_TEMPERATURE,
            num_predict=MAX_TOKENS,
            top_k=TOP_K,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
        )

    @weave.op()
    def route_to_agent(self, state: AgentState) -> str:
        """Route to appropriate agent based on question content using LLM decision making"""
        messages = state["messages"]
        if not messages:
            return "math_formatter_agent"

        last_message = messages[-1]["content"]

        # Use LLM to make routing decision
        routing_prompt = f"""
        You are an agent router. Based on the following question, determine which agent should handle it:
        
        Question: {last_message}
        
        Available agents:
        1. math_formatter_agent - for mathematical calculations and arithmetic operations
        2. google_search_agent - for factual questions requiring web search
        
        Answer with only the agent name (math_formatter_agent or google_search_agent) that should handle this question.
        
        If the question involves:
        - Mathematical operations (addition, subtraction, multiplication, division, etc.)
        - Calculations with numbers
        - Algebraic expressions
        - Mathematical problem solving
        
        Then respond with: math_formatter_agent
        
        If the question involves:
        - Factual information that requires current web data
        - Historical events
        - Current statistics
        - Definitions or explanations requiring up-to-date information
        - Questions about people, places, or things that need current knowledge
        
        Then respond with: google_search_agent
        
        Question: {last_message}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=routing_prompt)])
            agent_decision = response.content.strip().lower()

            # Validate the decision
            if agent_decision in ["math_formatter_agent", "google_search_agent"]:
                return agent_decision
            else:
                # Default to calculator agent for mathematical questions
                if any(
                    keyword in last_message.lower()
                    for keyword in [
                        "calculate",
                        "solve",
                        "math",
                        "add",
                        "subtract",
                        "multiply",
                        "divide",
                        "sum",
                        "product",
                        "quotient",
                    ]
                ):
                    return "math_formatter_agent"
                   
                else:
                    return "google_search_agent"


        except Exception as e:
            # Fallback to original logic if LLM fails
            if "calculate" in last_message.lower() or "solve" in last_message.lower():
                return "math_formatter_agent"
            else:
                return "google_search_agent"

    @weave.op()
    def math_formatter_agent_node(self, state: AgentState) -> AgentState:
        """Process with mathematical reasoning agent"""
        messages = state["messages"]
        question = messages[-1]["content"] if messages else ""

        response = self.math_formatter_agent.get_math_response(question)

        # Add the response to the conversation
        updated_messages = messages + [{"role": "assistant", "content": response}]

        return {"messages": updated_messages, "current_agent": "math_formatter_agent"}

    @weave.op()
    def calculator_agent_node(self, state: AgentState) -> AgentState:
        """Process with calculator agent"""
        messages = state["messages"]
        question = messages[-1]["content"] if messages else ""

        try:
            # Use the existing calculator agent
            result = self.calculator_agent.invoke({"input": question})
            response = result.get("output", "No output")
        except Exception as e:
            response = f"Error: {str(e)}"

        # Add the response to the conversation
        updated_messages = messages + [{"role": "assistant", "content": response}]

        return {"messages": updated_messages, "current_agent": "calculator_agent"}

    @weave.op()
    def google_search_agent_node(self, state: AgentState) -> AgentState:
        """Process with calculator agent"""
        messages = state["messages"]
        question = messages[-1]["content"] if messages else ""

        try:
            # Use the existing calculator agent
            response = self.google_search_agent.get_search_response(question)
        except Exception as e:
            response = f"Error: {str(e)}"

        # Add the response to the conversation
        updated_messages = messages + [{"role": "assistant", "content": response}]

        return {"messages": updated_messages, "current_agent": "google_search_agent"}


# Initialize the multi-agent system
multi_agent_system = MultiAgentSystem()


@weave.op()
def run_agent(input_text):
    """Run the agent with input text"""
    try:
        # Single agent approach (existing functionality)
        result = agent_executor.invoke({"input": input_text})

        # Log execution trace
        trace_logger.log_agent_execution(input_text, result)

        return result
    except Exception as e:
        print(f"Error in agent execution: {e}")
        raise


@weave.op()
def get_result(question: str):
    result = {"output": "No output"}

    try:
        result = run_agent_multi(question)
    except Exception as e:
        print(f"Multi-agent failed: {e}")

    return result


@weave.op()
def get_response(question: str):
    result = get_result(question)
    return {"generated_text": result.get("output", "")}


@weave.op()
def run_multiple_calculations(questions):
    """Run agent on multiple questions and return results"""
    results = []
    for question in questions:
        try:
            result = get_result(question)
            results.append(
                {
                    "question": question,
                    "answer": result.get("output", ""),
                    "input": question,
                }
            )
        except Exception as e:
            results.append(
                {"question": question, "answer": f"Error: {str(e)}", "input": question}
            )
    return results

@weave.op()
def run_agent_multi(input_text):
    """Run the multi-agent system with input text"""
    try:
        # Initialize state
        initial_state = {
            "messages": [{"role": "user", "content": input_text}],
            "current_agent": "math_formatter_agent",
        }

        # Create a unique thread ID for this execution
        thread_id = str(uuid.uuid4())

        # Create the graph
        graph = StateGraph(AgentState)
        graph.add_node("math_formatter_agent", multi_agent_system.math_formatter_agent_node)
        graph.add_node("calculator_agent", multi_agent_system.calculator_agent_node)
        graph.add_node(
            "google_search_agent", multi_agent_system.google_search_agent_node
        )

        # Add edges - Fixed the routing logic
        # graph.add_edge(START, "router")
        graph.add_conditional_edges(
        START,
        multi_agent_system.route_to_agent,
        {
            "math_formatter_agent": "math_formatter_agent",
            "google_search_agent": "google_search_agent",
        }
    )
        graph.add_edge("math_formatter_agent", "calculator_agent")
        graph.add_edge("calculator_agent", END)
        graph.add_edge("google_search_agent", END)

        # Compile the graph with checkpointer
        compiled_graph = graph.compile(checkpointer=MemorySaver())

        # Run with thread_id
        result = compiled_graph.invoke(
            initial_state, config={"configurable": {"thread_id": thread_id}}
        )

        return {"output": result["messages"][-1]["content"]}
    except Exception as e:
        print(f"Error in run_agent_multi: {e}")
        return {"output": f"Error: {str(e)}"}


def interactive_mode():
    """Run agent in interactive mode"""
    print("AI Agent Interactive Mode (type 'exit' to quit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if user_input:
                result = get_result(user_input)
                print(f"Answer: {result.get('output', 'No output')}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


@weave.op()
def run_evaluation_suite(num_problems: int):
    """Run a complete evaluation suite with multiple questions"""
    # Define test questions and expected answers
    questions, expected_answers = get_problems(num_problems=num_problems)
    examples = [
        {"question": problem, "expected": answer}
        for problem, answer in zip(questions, expected_answers)
    ]
    evaluation = weave.Evaluation(
        dataset=examples, scorers=[get_calculation_evaluation]
    )
    asyncio.run(evaluation.evaluate(get_response))

    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Langchain Agent")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--evaluation", action="store_true", help="Run in evaluation mode"
    )
    parser.add_argument(
        "--num_problems", type=int, default=5, help="The number of problems"
    )
    parser.add_argument("--question", type=str, help="Single question to process")
    parser.add_argument(
        "--file", type=str, help="File containing questions (one per line)"
    )
    parser.add_argument(
        "--multi-agent", action="store_true", help="Use multi-agent system"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.evaluation:
        run_evaluation_suite(args.num_problems)
    elif args.question:
        if args.multi_agent:
            result = run_agent_multi(args.question)
            print(f"Question: {args.question}")
            print(f"Answer: {result.get('output', 'No output')}")
        else:
            result = run_agent(args.question)
            print(f"Question: {args.question}")
            print(f"Answer: {result.get('output', 'No output')}")
    elif args.file:
        try:
            with open(args.file, "r") as f:
                questions = [line.strip() for line in f if line.strip()]

            for question in questions:
                print(f"\nQuestion: {question}")
                if args.multi_agent:
                    result = run_agent_multi(question)
                    print(f"Answer: {result.get('output', 'No output')}")
                else:
                    result = run_agent(question)
                    print(f"Answer: {result.get('output', 'No output')}")
        except FileNotFoundError:
            print(f"File {args.file} not found")
    else:
        # Default: interactive mode
        interactive_mode()
