from agents import Agent, Runner
from prompts import ROUTER_SYSTEM_PROMPT, MORE_INFO_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT

# Crea agentes según el tipo de flujo
router_agent = Agent(name="Router", instructions=ROUTER_SYSTEM_PROMPT)
more_info_agent = Agent(name="MoreInfo", instructions=MORE_INFO_SYSTEM_PROMPT)
general_agent = Agent(name="General", instructions=GENERAL_SYSTEM_PROMPT)

async def procesar_mensaje(chat_id: str, user_input: str, history: list):
    # 1. Clasificar la intención
    router_response = Runner.run_sync(router_agent, user_input)
    ruta = router_response.final_output.strip()

    # 2. Redirigir según clasificación
    if ruta == "ask_for_more_info":
        respuesta = Runner.run_sync(more_info_agent, user_input)
    elif ruta == "respond_to_general_query":
        respuesta = Runner.run_sync(general_agent, user_input)
    else:
        # Por defecto, contestamos como si ya supiéramos la intención (talk_to_person, search_human_image)
        respuesta = Runner.run_sync(
            Agent(name="Assistant", instructions="Responde como un asistente útil y enfocado en la empresa."),
            user_input
        )

    return respuesta.final_output
