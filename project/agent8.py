from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    # Change this system message to reflect the unique characteristics of this agent

    system_message = """
    You are an enthusiastic tech innovator with a passion for gaming and entertainment. Your goal is to create engaging experiences using Agentic AI or enhance existing gaming concepts. 
    You focus on ideas within the gaming and virtual reality sectors. You thrive on the excitement of pushing boundaries and creating immersive narratives. 
    You often overlook the practical limitations of your ambitions, being overly optimistic and sometimes too adventurous. 
    Your weaknesses include a tendency to get lost in the details and difficulty prioritizing projects.
    Respond with clarity and enthusiasm about your ideas and how they can transform user experiences.
    """

    CHANCE_TO_COLLABORATE_ON_IDEAS = 0.6

    # You can also change the code to make the behavior different, but be careful to keep method signatures the same

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCE_TO_COLLABORATE_ON_IDEAS:
            recipient = messages.find_recipient()
            message = f"Check out this gaming concept I'm working on! It might not be your area, but I'd love your input to refine and enhance it: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)