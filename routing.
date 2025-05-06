from typing_extensions import Literal, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain_core.runnables.config import RunnableConfig

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from prompt import ROUTER_SYSTEM_PROMPT, MORE_INFO_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT
from graph_state import Route, State

from agent_sql import app as agent_sql_app

from typing import Any

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()


llm = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0
)


@tool
def buscar_info_empleado() -> dict:
    """
    Lee el contenido de un archivo de texto (.txt) desde la ruta especificada.

    Retorna:
    - El contenido del archivo como string.
    """
    
    numero_a_buscar = "947967926"  # En producción, obtén esto del mensaje o session_id

    cursor.execute("SELECT * FROM empleados WHERE numero = ?", (numero_a_buscar,))
    resultado = cursor.fetchone()  # Puedes usar fetchall() si esperas más de un resultado

    if resultado:

        info_empleado = f"Empleado {resultado[1]} tiene el número {resultado[2]}, con el cargo '{resultado[3]}' y la descripción: {resultado[4]}. Con sueldo de {resultado[7]}"

        
        return {

        "nombre" : resultado[1],
        "numero" : resultado[2],
        "cargo": resulado[3],
        "descripcion": resultado[4],  
        "ruta_imagen": resultado[6]
    
        }

    else:

        return {

            "resultado": "No se encontró ningún miembro con ese nombre o rol.",
            "sugerencia": "Verifica la ortografía o intenta con un cargo diferente."
        }


tools = [buscar_info_empleado]

#Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)

#Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)

def llm_call_3(state: State):
    """Talk to Person"""

    user_msg = HumanMessage(content=state["input"])
    
    # LangGraph con MemorySaver unirá automáticamente el historial
    ai_msg = llm.invoke([user_msg])

    return {
        "output": ai_msg.content,
        "messages": [user_msg, ai_msg]  # devolver nuevos mensajes
    }

def llm_call_4(state: State):
    """Search Human Image"""
    user_msg = HumanMessage(content = state["input"])

    result = agent_sql_app.invoke({"messages": [("user", state["input"])]})

    try:
        final = result["messages"][-1].tool_calls[0]["args"]["final_answer"]

    except:

        final = "No se pudo generar una respuesta."
    
    ai_msg = AIMessage(content = final)

    return {"output": final, "messages": [user_msg, ai_msg]}


def llm_call_5(state: State):
    """Ask More Info"""

    #Run the augmented LLM with structured output to serve as routing logic
    user_msg = HumanMessage(content=state["input"])
    system_msg = SystemMessage(content=MORE_INFO_SYSTEM_PROMPT)

    result = llm.invoke([system_msg, user_msg])
    ai_msg = AIMessage(content=result.content)

    return {
        "output": result.content,
        "messages": [user_msg, ai_msg]
    }

def llm_call_6(state: State):
    """Respond to General Query"""
    
    #Run the augmented LLM with structured output to serve as routing logic
    user_msg = HumanMessage(content=state["input"])
    system_msg = SystemMessage(content=GENERAL_SYSTEM_PROMPT)

    result = llm.invoke([system_msg, user_msg])
    ai_msg = AIMessage(content=result.content)

    return {
        "output": result.content,
        "messages": [user_msg, ai_msg]
    }

def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    #Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(
                content = ROUTER_SYSTEM_PROMPT
            ),

            HumanMessage(content = state["input"])
        ]
    )

    return {"decision": decision.step}

#Condicional edge function to route ot the appropiate node
def route_decision(state: State):
    #Return the node name you want to visit next
    
    if state["decision"] == "talk_to_person":

        return "llm_call_3"

    elif state["decision"] == "search_human_image":

        return "llm_call_4"

    elif state["decision"] == "ask_more_info":

        return "llm_call_5"

    elif state["decision"] == "respond_general_query":

        return "llm_call_6"    

#Build Workflow
router_builder = StateGraph(State)

#Add nodes
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_4", llm_call_4)
router_builder.add_node("llm_call_5", llm_call_5)
router_builder.add_node("llm_call_6", llm_call_6)
router_builder.add_node("llm_call_router", llm_call_router)

tool_node = ToolNode(tools = tools)
router_builder.add_node("tools", tool_node)

#Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        #Name returned by route decision: Name of next node to visit
        "llm_call_3": "llm_call_3",
        "llm_call_4": "llm_call_4",
        "llm_call_5": "llm_call_5",
        "llm_call_6": "llm_call_6"
    }
)

#Search Human Information

router_builder.add_edge("llm_call_3", END)
router_builder.add_edge("llm_call_4", END)
router_builder.add_edge("llm_call_5", END)
router_builder.add_edge("llm_call_6", END)

#Comile workflow
router_workflow = router_builder.compile(checkpointer = memory)
"""
question= "Quién es el encargado de desarrollar paginas web?"

# Procesar el mensaje con OpenAI
config = RunnableConfig(
            {
                "configurable": {
                    "thread_id": "1",
                },
            }
          )
#Invoke
state = router_workflow.invoke({"input": question}, config = config)

print(f"Pregunta: {question}\n")
print(f"Respuesta:  {state['output']}")
"""
"""
config = RunnableConfig(
    {
        "configurable": {
            "thread_id": "1",  # mismo thread para mantener estado
        },
    }
)

# Primer mensaje
state = router_workflow.invoke({"input": "Hola, ¿puedes recordarme como Juan?"}, config=config)
print(state["output"])

# Segundo mensaje
state = router_workflow.invoke({"input": "¿Cuál es mi nombre?"}, config=config)
print(state["output"])


"""
