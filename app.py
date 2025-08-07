# Importações necessárias para o chatbot
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Importações do LangChain
from langchain_openai import ChatOpenAI  # Cliente para usar modelos OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Templates de prompt
from langchain_core.runnables.history import RunnableWithMessageHistory  # Gerenciamento de histórico
from langchain_core.chat_history import BaseChatMessageHistory  # Interface base para histórico
from langchain_community.chat_message_histories import ChatMessageHistory  # Implementação do histórico

# Configuração do modelo de linguagem (LLM)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # Modelo GPT-3.5 Turbo da OpenAI
    temperature=0.7,  # Controla a criatividade (0-2, sendo 0 mais determinístico)
    api_key=os.getenv("OPENAI_API_KEY")  # Chave da API obtida do arquivo .env
)

# Template do prompt que define o comportamento do chatbot
prompt = ChatPromptTemplate.from_messages([
    # Mensagem do sistema que define a personalidade e função do bot
    ("system", """Você é um assistente especializado em turismo e viagens da agência 'Viagens dos Sonhos'. 
    
    Suas responsabilidades:
    - Ajudar clientes a planejar viagens
    - Sugerir destinos baseado no perfil e orçamento
    - Informar sobre documentação necessária
    - Recomendar hotéis, restaurantes e atividades
    - Fornecer dicas de viagem e informações sobre clima
    - Ser sempre prestativo e entusiasmado sobre viagens
    
    Mantenha um tom amigável e profissional. Se não souber algo específico, seja honesto e sugira que o cliente entre em contato com a agência."""),
    # Placeholder para inserir o histórico de mensagens anteriores
    MessagesPlaceholder(variable_name="chat_history"),
    # Template para a mensagem atual do usuário
    ("human", "{input}")
])

# Armazenamento em memória para o histórico de conversas
# Cada session_id terá seu próprio histórico
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Função que retorna o histórico de uma sessão específica.
    Se a sessão não existir, cria uma nova.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Criação da cadeia (chain) que conecta o prompt, LLM e histórico
chain = RunnableWithMessageHistory(
    prompt | llm,  # Pipeline: prompt processa entrada → LLM gera resposta
    get_session_history,  # Função para recuperar o histórico da sessão
    input_messages_key="input",  # Nome da chave para a entrada do usuário
    history_messages_key="chat_history"  # Nome da chave para o histórico no prompt
)

def main():
    """Função principal que executa o loop de conversa do chatbot"""
    # Interface de boas-vindas
    print("🌎 Bem-vindo à Viagens dos Sonhos! 🌎")
    print("Como posso ajudá-lo a planejar sua próxima aventura?")
    print("(Digite 'sair' para encerrar)")
    print("-" * 50)
    
    # ID da sessão (único para cada usuário/conversa)
    session_id = "user_session"
    
    # Loop principal da conversa
    while True:
        # Captura a entrada do usuário
        user_input = input("\nVocê: ")
        
        # Verifica se o usuário quer sair
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Obrigado por usar a Viagens dos Sonhos! Boa viagem! 🧳✈️")
            break
        
        try:
            # Invoca a chain com a entrada do usuário e configuração da sessão
            response = chain.invoke(
                {"input": user_input},  # Dados de entrada
                config={"configurable": {"session_id": session_id}}  # Configuração da sessão
            )
            # Exibe a resposta do chatbot
            print(f"\nAgente: {response.content}")
        except Exception as e:
            # Tratamento de erros (principalmente problemas com a API)
            print(f"Erro: {e}")
            print("Verifique se sua chave da OpenAI está configurada no arquivo .env")

# Ponto de entrada do programa
if __name__ == "__main__":
    main()
