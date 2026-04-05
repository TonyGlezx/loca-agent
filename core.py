from interfaces import Interfaces
import uuid6, re
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage

class GenerateImageSchema(BaseModel):
    prompt: str = Field(description="Description of the image to generate.")
    base_image_name: str = Field(description="A name for the image.")
    spins: int = Field(default=1, description="How many spins you need of the image.")

class Tools():
    def __init__(self):
        self.interface = Interfaces()
        self.sd_pipe = None
        # The tool definition is now strictly inside the class instance
        self.generate_image = StructuredTool.from_function(
            func=self._generate_image,
            name="generate_image",
            description="Generate an image based on a description",
            args_schema=GenerateImageSchema
        )
    
    def start_sd(self):
        print("Trying to start SD pipeline")
        self.sd_pipe = self.interface.get_sd_pipe()

    def _generate_image(self, prompt: str, base_image_name: str, spins: int = 1 ) -> str:
        images = []
        for spin in range(spins):
            image = self.sd_pipe(prompt).images[0]
            img_uuid = str(uuid6.uuid7())
            clean_name = re.sub(r'[\\/*?:"<>|]', "", base_image_name)
            clean_name = clean_name.replace(" ", "_")
            clean_name = clean_name.strip(". ")
            path = f"{img_uuid}_{clean_name}_{spin}.png"
            image.save(path)
            images.append(path)
        
        return f"The following images were saved succesfully. {images}"

class LocalAgent():
    def __init__(self, llm, tools, system_msg, human_msg):
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm = llm
        
        self.app = None
        
        self.human_msg = human_msg
        self.system_msg = system_msg
    
    def main(self):
        system_msg = SystemMessage(content=self.system_msg)
        human_msg = HumanMessage(content=self.human_msg)
        print("\n🚀 Starting Agent Execution...\n")

        # Use .stream() with stream_mode="updates"
        events = self.app.stream(
            {"messages": [system_msg, human_msg]}, 
            stream_mode="updates"
        )

        for event in events:
            # Event is a dict mapping the node_name to the state update it returned
            for node_name, state_update in event.items():
                print(f"--- 🟢 Update from node: {node_name} ---")
                
                # Grab the message that was just generated
                latest_msg = state_update["messages"][-1]
                
                # Check if it was a tool call
                if hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                    print(f"🛠️  Agent decided to use tool: {latest_msg.tool_calls[0]['name']}")
                    print(f"📋 Arguments: {latest_msg.tool_calls[0]['args']}")
                
                # Or if it was the tool responding
                elif node_name == "tool_node":
                    print(f"✅ Tool Result: {latest_msg.content}")
                
                # Or if it was the vision node
                elif node_name == "vision_node":
                    print(f"👁️  Vision Analysis Complete.")
                    print(f"📝 {latest_msg.content}\n")
                
                # Or if it was a standard text response
                else:
                    print(f"🤖 Agent says: {latest_msg.content}\n")

        return "Agent finished."
