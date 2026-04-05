from core import LocalAgent, Tools
from langgraph.graph import StateGraph, START, MessagesState, END
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, HumanMessage
from typing import Literal
import base64
import os
import re

class UseCases():
    def __init__(self):
        pass
    
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def gen_image(self, prompt):
        # 1. Instantiate and start tools for THIS use case
        toolkit = Tools()
        toolkit.start_sd()

        system_msg = """
        You are an autonomous agent responsible for generating high-quality images.
        When you receive an image description from the 'Image Verifier', evaluate if it matches your original intent.
        - If the image meets your standards and intent, simply provide a brief summary and the task will end.
        - If the image is NOT good enough or has major flaws, use the 'generate_image' tool again with improved prompts to fix the issues.
        """
        human_msg = f"{prompt} NOT ask me for details, select the image description autonomously."
        
        # 2. Setup tools and LLMs
        tools = [toolkit.generate_image]
        
        main_llm = ChatOllama(
            model="glm-4.7-flash:latest",
            temperature=0.7,
        ).bind_tools(tools)
        
        vision_llm = ChatOllama(
            model="qwen3.5:9b",
            temperature=0.2,
        )
        
        agent = LocalAgent(main_llm, tools, system_msg, human_msg)

        # 3. Define the nodes for THIS use case
        def chat_node(state: MessagesState):
            response = agent.llm.invoke(state["messages"])
            return {"messages": [response]}

        def tool_node(state: dict):
            result = []
            for tool_call in state["messages"][-1].tool_calls:
                tool_instance = agent.tools_by_name[tool_call["name"]]
                observation = tool_instance.invoke(tool_call["args"])
                result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            return {"messages": result}

        def vision_node(state: MessagesState):
            """Node to describe the generated images using a multimodal model"""
            last_msg = state["messages"][-1]
            if not isinstance(last_msg, ToolMessage):
                return {"messages": []}

            # Extract image paths from ToolMessage content using regex
            content = last_msg.content
            image_paths = re.findall(r"['\"]([^'\"]+\.png)['\"]", content)
            
            descriptions = []
            for path in image_paths:
                if os.path.exists(path):
                    print(f"--- 👁️ Vision Node: Analyzing {path} ---")
                    img_base64 = self._encode_image(path)
                    
                    # Call multimodal Qwen 3.5
                    vision_msg = HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe this image in detail. Focus on the composition, style, and whether it looks 'cool' and high quality."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    )
                    response = vision_llm.invoke([vision_msg])
                    descriptions.append(f"Image {path} description: {response.content}")
                else:
                    descriptions.append(f"Could not find image at {path}")

            verification_msg = HumanMessage(
                content=f"Image Verifier results:\n" + "\n".join(descriptions) + 
                        "\n\nDoes this match your original intent and quality standards? If it doesn't, try again by calling 'generate_image' with improved prompts. If it is satisfactory, you can finish."
            )
            return {"messages": [verification_msg]}

        def should_continue(state: MessagesState) -> Literal["tool_node", END]:
            """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
            messages = state["messages"]
            last_message = messages[-1]

            if last_message.tool_calls:
                return "tool_node"

            return END

        # 4. Build the graph
        def _build_graph():
            graph = StateGraph(MessagesState)
            graph.add_node("chat_node", chat_node)
            graph.add_node("tool_node", tool_node)
            graph.add_node("vision_node", vision_node)
            
            graph.add_edge(START, "chat_node")
            graph.add_conditional_edges(
                "chat_node",
                should_continue,
                ["tool_node", END]
            )
            graph.add_edge("tool_node", "vision_node")
            graph.add_edge("vision_node", "chat_node")
            
            return graph.compile()

        # 5. Execute
        agent.app = _build_graph()
        response = agent.main()
        print(response)
