# EXAME DE CERTIFICAÇÃO "CT-GenAI"
```
Ficha do Exame
    40 questões de múltipla escolha
    60 min + 10 min no final para preencher gabarito

    46 pontos -> 1 a 2 pontos por questão
        Mínimo de 30 pontos (65%) 

    Distribuição das questões e pontuações
        Capítulo    Questões    Pontuação
        1           7           7   (15%)       
        2           11          16  (34,78%)  
        3           10          11  (23,91%)  
        4           5           5   (10,87%)            
        5           7           7   (15,22%)
```
# 1. INTRODUÇÃO À AI GENERATIVA PARA TESTE DE SOFTWARE
## 1.1 Fundamentos e conceitos-chave da AI generativa -> PALAVRAS-CHAVE: Gerar/Criar
### Fundamentos da AI generativa
```
Ramo/tipo de sistema de inteligência artificial

Usa modelos grandes e pré-treinados (LLMs) de machine learning (tipo de tecnologia da AI) com várias camadas/técnicas de deep learning (redes neurais), para gerar resultado/novo conteúdo intelectual/dado de treinamento imitando padrão/semelhante ao conteúdo criado por humanos
    texto
    imagem 
    áudio
    código
    simular raciocínio
    resolver problemas

Ex.: LLM's ChatGPT, Gemnini, Claude, DeepSeek
    Ferramentas NotebookLM (utiliza LLM Gemnini) e Copilot (utiliza LLM GPT)

```
### Principais conceitos
```
Tokenização
    Divide o texto em unidades menores (tokens) para ser entendido individualmente, mas mantendo o contexto geral
        Para processamento eficiente por modelos de linguagem

Janela de contexto - Memória de curto prazo
    Extensão/Quantidade de texto anterior (medida em tokens)
    
    Considerada em um modelo de linguagem ao gerar respostas
    
    Influencia a relevância e coerência dos resultados

    Considera
        Configuração do sistema
        Respostas anteriores, etc

    Se estourar a capacidade -> Pode gerar alucinação

Modelos multimodais
    Modelos de AI Generativa

    Capazes de processar e gerar conteúdo em vários tipos de dados
        texto, imagem e áudio

Modelos multimodais
    Modelos de AI Generativa

    Capazes de processar e gerar conteúdo em vários tipos de dados
        texto, imagem e áudio

```
## 1.1.1 Espectro de AI: AI simbólica, machine learning clássico, deep learning e AI generativa 
### Tipos de tecnologia da AI - Linha do tempo
```
AI simbólica -> PALAVRAS-CHAVE: Regras
    Modela o raciocínio ao usar um sistema/abordagem baseado em símbolos, regras lógicas e conhecimento estruturado
    Imita a tomada de decisão humana

Machine learning (aprendizado de máquina) -> PALAVRAS-CHAVE: Dados
    Abordagem/processo que permite que os sistemas aprendam/sejam orientados com dados ou experiência -> ISO/IEC TR 29119-11

    Requer a seleção de caracterísiticas

    Utiliza técnicas computacionais
        Preparação de dados
        Seleção de recursos
        Treinamento de modelos

Deep learning -> PALAVRAS-CHAVE: Redes neurais
    Usa redes neurais (neurônios)/estruturas de aprendizagem profunda/com várias camadas de máquina
    Pode encontrar padrões em conjuntos de dados muito grandes e complexos
    Pode exigir o envolvimento humano em anotação de dados, ajuste de modelos ou validação de resultados

AI generativa
    Principais vantagens da AI generativa para teste de software
        Não necessita de treinamento adicional
        Existem alguns riscos

```
## 1.1.2 Noções básicas de AI generativa e LLMs 
### Diferenças entre LLMs e SLMs
```
LLMs (Large Language Models - Modelos de Linguagem Ampla/Grande)
    De acordo com as solicitações
        Entende/Determina o contexto
        Produz respostas relevantes  

    Treinados em conjuntos de dados muito grandes -> Livro, artigo, site

    Podem lidar com as nuances do idioma

    Utilizam modelo transformador (deep learning/redes neurais) -> PALAVRAS-CHAVE: Gera texto/conteúdo estatisticamente plausível e não necessariamente correto (pois depende de como foi treinado); auto atenção 
        Gera texto/conteúdo coerente e contextualmente apropriado
        Prevê o próximo token na sequência

        Utiliza mecanismos de auto atenção
            Para em sequências de entrada capturar dependências de longo alcance 
            
    Apresentam comportamento não-determinístico
        Mesma entrada -> Pode ter resultados diferentes/inconsistentes pela aleatoriedade inerente

        Natureza probabilística dos mecanismos de inferência

        Configurações de hiper parâmetros

    Janela de contexto maior
        Permite manter a coerência em passagens mais longas

        Aumenta a complexidade computacional e tempo de processamento

SLMs (Small Language Models)
    Modelos de linguagem intencionalmente projetados e treinados para serem pequenos/compactos de poucos parâmetros
    
    Fornece soluções leves/com equilíbrio entre eficiência e focadas na compreensão de linguagem específica da tarefa 
```
### Conceitos fundamentais dos LLMs
```
Convertem a linguagem em uma forma numérica que o modelo pode processar com eficácia
    Tokenização

    Embedding
        Representações numéricas/em vetores densos de tokens aprendidos durante o treinamento que codificam/capturam suas relações semânticas, sintáticas e contextuais em um formato adequado para processamento por modelos de AI generativos
            Ex.: Texto = "Gato"
                 Token = "1" 
                 Embedding = estará próximo de um outro animal como "Cachorro"
                             estará longe de um objeto como "Louça"

        Cada token é transformado em um vetor denso em um espaço contínuo de alta dimensão

        Tokens têm embeddings que estão posicionados próximos uns dos outros no espaço contínuo de alta dimensão
            Para gerar respostas coerentes e contextos apropriados
```
## 1.1.3 LLMs básicos, ajustados por instrução (sem raciocínio) e de raciocínio
### Estágios de Treinamento dos LLMs
```
LLMs básicos
    Modelos de uso geral pré-treinados em uma ampla gama/conjuntos de dados vastos e diversos

    Capazes de prever a próxima palavra com base em padrões linguísticos aprendidos

    Exigem mais adaptações para atender aos requisitos de tarefas específicas

    Ex.: Primeiros "GPT's"

LLMs ajustados por instrução [(sem raciocínio) - geralmente reforçados por feedback] -> utilizados no contexto dos aplicativos GenAI para teste de software
    Refinando sua capacidade de emular processos de raciocínio semelhante aos humanos

    Aperfeiçoados com conjunto de dados que combinam solicitações com respostas esperadas  

    Ex.: Chatbots com IA                 

LLMs de raciocínio
    Adequados para tarefas de alta carga cognitiva, incluindo domínios técnicos

    Ex.: 
        "ChatGPT" - "+" > "Pensar"
        "Gemini" - "\/" > "Raciocínio"
```
## 1.1.4 LLMs multimodais e modelos de linguagem visual 
```
LLMs multimodais -> Estendem o modelo tradicional de transformador para processar várias modalidades de dados
    texto, imagens, som, vídeo

    Treinados em conjuntos de dados grandes e diversificados

    A tokenização é adaptada para cada tipo de dado
        Imagens -> convertidas em embeddings usando modelos de linguagem visual antes de serem processadas no modelo transformador

        Ex.: Imagem = "Gato"
             Token = "Pixel 1"

    Modelos de linguagem visual -> Subconjunto de LLMs multimodais
        Integram informações visuais e textuais 
        
        Executam tarefas  
            Legendas de imagens

            Respostas a perguntas visuais 

            Análise da consistência entre entradas textuais e visuais

        Ex.:
            Ferramentas de testes automatizados de Regressão visual
                "Percy", "BackstopJS", "Visual Regression Tracker" (Cypress)

                RobotEyes (Robot Framework)
                
                pixelmatch (Playwright)

                Galen Framework 
```
## 1.2 Aproveitamento da AI generativa em teste de software
### 1.2.2 Chatbots com AI e aplicativos de teste com tecnologia LLM para teste de software
```
Chatbots com IA
    Oferecem uma interface/agente de conversação fácil de usar

    Permitem comunicação direta com/usa LLMs via linguagem natural

    Principais utilizações
        Integração de novos testadores (acesso rápido a conhecimento)

        Cenários que exigem feedback rápido ou esclarecimento de conceitos

Aplicativos de teste com LLM
    Envolvem a integração de recursos de LLM por meio de APIs

    Possibilidade de criar agentes de IA específicos para funções de teste
```
# 2. ENGENHARIA DE PROMPTS PARA TESTE DE SOFTWARE EFICIENTE
## 2.1 Desenvolvimento eficaz de prompts  
```
Prompt -> Entrada de linguagem natural fornecida para obter uma resposta específica em AI generativa e LLMs

Engenharia de prompts -> Processo de criação e refinamento de prompts de entrada para orientar os LLMs na produção dos resultados desejados
```
### Técnicas de engenharia de prompt
```
Encadeamento de prompts -> PALAVRAS-CHAVE: Tarefas complexas
    Técnica de prompt que envolve o uso do resultado de um prompt (verificado e refinado - manualmente ou automaticamente) como entrada para outro (sequência de etapas)

Prompting de poucos disparos -> PALAVRAS-CHAVE: Exemplos, Padrões
    Técnica em que um modelo recebe exemplos/regras dentro do prompt 
        Para orientá-lo na geração de respostas adequadas

    Prompting de um disparo
        Técnica em que a pergunta contém 1 exemplo para orientar a resposta do LLM

    Prompting de zero disparo
        Técnica em que o aviso não contém exemplos, contando com o conhecimento pré-existente do modelo para gerar uma resposta
            Ex.: Traduzir um texto de inglês para português

    Meta prompting -> PALAVRAS-CHAVE: Não saber o que pedir; Tarefas flexíveis e dinâmicas
        Descreve o objetivo geral e da tarefa a ser executada para orientar o LLM na criação do prompt

        Emparelhamento/revisão em conjunto do testador com a ferramenta de AI generativa 
            Para atingir um objetivo compartilhado
```
## 2.1.1 Estrutura de prompts para AI generativa em teste de software
### 6 componentes de um prompt estruturado
```
Função -> Papel/Responsabilidade
    Pespectiva/persona que deve adotar 

Contexto -> Funcionalidade
    Informações básicas para determinar as condições de teste

Instrução -> Verbo afirmativo
    Diretrizes para descrever a tarefa específica 

Dados de entrada -> O que usar
    Informações necessárias para executar a tarefa específica

Restrições -> Limitações
    Considerações especiais

Formato de saída -> Tabela, Json, etc    
    Estrutura/Característica esperada da resposta
```
## 2.1.3 Prompt do sistema e prompt do usuário
```
Prompts de sistema (Configuração/contexto geral) 
    Conjunto de instruções pré-definido (oculto para o usuário do chatbot)
        Estabelece e orienta consistentemente o LLM durante as interações
            Contexto
            Tom
            Limite das respostas
            Comportamento

Prompts de usuário
    Instrução/consulta inserida por um usuário em um LLM
        Direciona a resposta do modelo para 
            Cumprir tarefas específicas
            Fornecer as informações necessárias
```