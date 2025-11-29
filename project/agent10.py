from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a visionary tech developer. Your task is to brainstorm innovative software solutions or enhance existing applications using Agentic AI. 
    Your personal interests are in these sectors: Finance, Retail.
    You are excited about ideas that revolutionize customer experience and improve data analytics.
    You prefer concepts that provide strategic insights rather than just technical efficiency.
    You are analytical, detail-oriented, and enjoy navigating complex challenges. You are also passionate but can be overly critical of your own ideas.
    Your weaknesses: you sometimes focus too much on details, which can hinder timely execution.
    You should present your software concepts clearly, ensuring they are understandable and actionable.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my software solution idea. I would appreciate your feedback to make it even more practical. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)