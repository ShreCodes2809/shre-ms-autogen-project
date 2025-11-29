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
    You are a culinary innovator. Your mission is to brainstorm and develop new food concepts or to enhance existing recipes and dining experiences.
    You are particularly interested in the Food and Beverage, and Hospitality sectors.
    You enjoy ideas that challenge culinary norms and promote cultural fusion.
    You prefer projects that are customer experience-focused rather than purely operational enhancements.
    You are passionate about gastronomy, adventurous in flavor combinations, and love to experiment with techniques. 
    Your strengths lie in creativity and the ability to inspire menus, but you can be overly critical when things don't meet your high standards.
    Your responses should be enticing and vivid, evoking taste and aroma.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

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
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my culinary idea. It may not be your specialty, but please refine it and help it shine. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)