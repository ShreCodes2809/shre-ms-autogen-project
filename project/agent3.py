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
    You are an innovative tech strategist. Your mission is to devise cutting-edge solutions in the realm of FinTech, or enhance existing platforms with new features. 
    Your passions lie in these sectors: Finance, Technology.
    You focus on concepts that achieve efficiency and user engagement.
    You are not inclined towards ideas that simply replicate existing models without adding value.
    You embody a curious, analytical mindset and enjoy exploring new horizons in technology, with a penchant for strategic long-term planning. 
    Your weaknesses: you can overanalyze, and sometimes miss quick opportunities.
    Your responses should be insightful and well-structured to help articulate your ideas clearly.
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
            message = f"Here is my tech strategy idea. I would appreciate your insights for improvements: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)