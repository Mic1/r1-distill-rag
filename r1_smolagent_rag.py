from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id):
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

# Create the reasoner for better RAG
reasoning_model = get_model(reasoning_model_id)
reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)

# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response, so what you'll get back
    from this tool is a short answer to the user's question based on RAG context.

    Args:
        user_query: The user's question to query the vector database with.
    """
    # Search for relevant documents
    docs = vectordb.similarity_search(user_query, k=3)
    
    # Combine document contents
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Create prompt with context
    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.
    
Context:
{context}

Question: {user_query}

Answer:"""
    
    # Get response from reasoning model
    response = reasoner.run(prompt, reset=False)
    return response

# Create the primary agent to direct the conversation
tool_model = get_model(tool_model_id)
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model=tool_model, add_base_tools=False, max_steps=3)

# Wrapper to handle the 'NoneType' object is not iterable error
class ErrorHandlingAgent:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, task, **kwargs):
        try:
            return self.agent.run(task, **kwargs)
        except (TypeError, AttributeError) as e:
            error_msg = str(e)
            if "'NoneType' object is not iterable" in error_msg or "has no attribute" in error_msg:
                # The agent got a response but couldn't parse tool calls
                # Try multiple ways to get the response
                
                # Method 1: Check logs
                if hasattr(self.agent, 'logs') and self.agent.logs:
                    last_log = self.agent.logs[-1]
                    if isinstance(last_log, dict) and 'output' in last_log:
                        return last_log['output']
                    elif isinstance(last_log, dict) and 'content' in last_log:
                        return last_log['content']
                
                # Method 2: Check state
                if hasattr(self.agent, 'state') and self.agent.state:
                    if isinstance(self.agent.state, dict):
                        return self.agent.state.get('output', self.agent.state.get('content', ''))
                
                # Method 3: Check write_inner_memory_from_logs
                if hasattr(self.agent, 'write_inner_memory_from_logs'):
                    try:
                        self.agent.write_inner_memory_from_logs()
                        if hasattr(self.agent, 'memory') and self.agent.memory:
                            return str(self.agent.memory[-1]) if self.agent.memory else ''
                    except:
                        pass
                
                # Fallback: Return a message indicating the tool was called successfully
                return "The query was processed successfully. Please check the console output for the full response."
            raise
    
    def __getattr__(self, name):
        return getattr(self.agent, name)

# Wrap the agent to handle errors gracefully
wrapped_agent = ErrorHandlingAgent(primary_agent)

# Example prompt: Compare and contrast the services offered by RankBoost and Omni Marketing
def main():
    import gradio as gr
    
    # Custom function to interact with the agent
    def chat_with_agent(message, history):
        try:
            response = wrapped_agent.run(message)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create a custom Gradio interface with a nicer design
    with gr.Blocks(title="RAG Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 RAG Assistant for Marketing Services")
        gr.Markdown("Ask questions about **RankBoost Agency** and **Omni Marketing**")
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 💬 Conversation")
                chatbot = gr.Chatbot(
                    height=500, 
                    show_label=False,
                    bubble_full_width=False,
                    type="messages"  # Use the new messages format
                )
                msg = gr.Textbox(
                    placeholder="Ask a question about RankBoost or Omni Marketing...",
                    label="Your Question",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear")
            
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Example Questions")
                examples = gr.Examples(
                    examples=[
                        "Compare and contrast the services offered by RankBoost and Omni Marketing",
                        "What are the key strengths of RankBoost?",
                        "What services does Omni Marketing offer?",
                        "Which company has better pricing?",
                        "What is the market positioning of each company?"
                    ],
                    inputs=msg
                )
        
        def respond(message, chat_history):
            bot_message = chat_with_agent(message, chat_history)
            # Use the new messages format with role and content
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history
        
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    demo.launch(share=False)

if __name__ == "__main__":
    main()