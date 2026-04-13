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
## 2.2 Aplicação de Técnicas da Engenharia de Prompts a Tarefas de Teste de Software 
### Engenharia de prompts para tarefas de teste
```
Análise de teste
    Identificação de defeitos
    Geração de condições de teste
    Priorização das condições de teste
    Análise de cobertura
    Sugestão de técnicas de teste

Projeto de teste
    Geração de casos de teste
    Síntese de dados de teste
    Geração de scripts
    Programação e priorização da execução dos testes

Automação de teste de regressão
    Scripts orientados por palavras-chave
    Análise de impacto
    Testes autocorretivos
    Relatórios automatizados

Monitoramento e controle de teste
    Monitoramento de testes e análise de métricas
    Controle de testes adaptativos
    Insights de conclusão e aprendizado contínuo
    Visualização e relatórios aprimorados de métricas de teste
```
##  2.3 Avaliar os Resultados da AI Generativa e Refinar as Instruções para as Tarefas de Teste de Software
### 2.3.1 Métricas para avaliar os resultados da AI generativa em tarefas de teste de software
```
Acurácia -> PALAVRAS-CHAVE: Geral/Todo
    Descrição 
        Mede a correção geral do resultado em relação a referência

    Exemplo - 100% de acurácia -> Casos de testes que cobrem 100% dos requisitos
        Grau em que os casos de testes gerados abrangem todos os requisitos especificados

Precisão -> PALAVRAS-CHAVE: Execução/Resultado obtido
    Descrição 
        Avalia a correção do resultado em relação ao objetivo específico

    Exemplo - 80% de precisão -> 100 casos de testes, mas 80 cobrem corretamente as anomalias
        Grau em que os casos de testes gerados identificam corretamente as anomalias

Recuperação -> PALAVRAS-CHAVE: Mínimo necessário
    Descrição
        Mede a capacidade de identificar as instâncias relevantes dos dados 

    Exemplo 
        Grau em que os casos de teste gerados abrangem a partição de equivalência válida e inválida em uma classe de dados

Relevância e ajuste contextual  
    Descrição 
        Determina se o resultado é aplicável e apropriado

    Exemplo
        Grau em que os casos de teste gerados são consistentes com a base de teste e integram os requisitos específicos de domínio

Diversidade  
    Descrição 
        Garante a cobertura de uma variedade de entradas e cenários sem repetição

    Exemplo      
        Grau em que os casos de teste gerados abrangem os comportamentos do usuário e exploram casos extremos    

Taxa de sucesso da execução 
    Descrição  
        Mede a proporção de casos/scripts de teste gerados que podem ser executados com êxito

    Exemplo  
        Quantidade de casos/scripts de teste gerados que podem ser executados sem erros

Eficiência de tempo 
    Descrição  
        Avalia a economia de tempo em relação aos testes manuais 

    Exemplo  
        Comparação de tempo exigido pela AI para gerar casos de teste em relação à criação manual por um ser humano 

```
##  2.3.2 Técnicas de avaliação e refinamento iterativo de prompts
```
Modificação iterativa do prompt
    Teste A/B de prompts
        Criar várias versões de prompts e avaliar qual versão produz melhor resultado com base em métricas predefinidas

    Análise de resultados -> PALAVRAS-CHAVE: Retrospectiva

    Integração do feedback do usuário -> PALAVRAS-CHAVE: Revisão entre equipe

    Ajuste do comprimento e especificidade do prompt

```
# 3. GERENCIANDO RISCOS DE AI GENERATIVA EM TESTE DE SOFTWARE
## 3.1 Alucinações, Erros de Raciocínio e Vieses 
Defeitos que reduzem a qualidade do testware (artefatos de teste) gerado e que podem reaparecer por causa do comportamento não determinístico
  - Resultam da natureza dos dados de treinamento e das limitações do modelo transformador
      - O reconhecimento e a abordagem aplicada para estes desafios podem aumentar a qualidade dos resultados nos processos de teste
```
Alucinações
    LLM gera/cria resultados/informações que parecem factualmente incorretos/irrelevantes para uma determinada tarefa

    Ex.: Criação de casos de testes fictícios/irrelevantes

Erros de raciocínio 
    LLM interpreta erroneamente estruturas lógicas, pois não tem raciocínio lógico verdadeiro e depende de correspondência de padrões 
    
    Ex.: Planejamento e priorização de casos de testes

Vieses 
    LLM utiliza dados no qual foi treinado que leva a resultados que favorecem determinados tipos 
        Informações
        Abordagens
        Suposições

    Ex.: LLM treinado a dar exemplos de CEP's de uma determinada região vai acabar gerando CEP's da mesma região que foi treinado
```
### Detecções
```
Detecção de alucinação 
    Verificação cruzada
        Comparar a saída com
            Documentação

            Requisitos

            Comportamento conhecido do sistema

    Consulta a especialistas no domínio

Detecção de erros de raciocínio 
    Validação lógica
        Avaliar a coerência e correção do fluxo lógico

    Teste de saída
        Executar casos/scripts de teste gerados para verificar resultados

Detecção de vieses
    Revisão do material de teste
        Avaliar a representação justa e precisa de dados de teste sintéticos

    Avaliação de vieses por tipo de teste
        Identificar sub-representação de testes
            Ex.: testes não funcionais

```  
### Mitigações
```
Mitigação para atenuar os riscos da GenAI 
    Fornecer contexto completo 

    Dividir os prompts

    Usar formato de dados claros e interpretáveis

    Comparar resultados entre modelos
        Teste A/B de prompts

Mitigação do comportamento não determinístico (probabilístico) dos LLMs
    Temperatura -> PALAVRAS-CHAVE: Resultado consistente
        Parâmetro que controla a aleatoriedade/criatividade dos resultados de um LLM

        Próximo a 0 -> menor aleatoriedade/criatividade - maior determinismo
            Ex.: Criação de casos de teste

        Longe de 0 -> maior aleatoriedade/criatividade - menor determinismo
            Ex.: Geração de imagens

    Sementes aleatórias -> PALAVRAS-CHAVE: Mesmo resultado 
        Definição de um valor fixo de semente para o gerador de números aleatórios garantir que a mesma sequência pseudoaletória seja usada para melhorar a reprodutibilidade
```
## 3.2 Privacidade de dados e riscos de segurança da AI generativa em teste de software 
### Vetores de Ataques Comuns
```
Exfiltração de dados -> PALAVRAS-CHAVE: Exceder janela de contexto
    Descrição
        Envio de solicitações projetadas para extrair dados de treinamento confidenciais

    Exemplo
        Exceder a janela de contexto do LLM com solicitações longas
            Para sobrecarregar a memória da AI
                Pode levar a AI a revelar trechos/informações confidenciais

Manipulação de solicitações -> PALAVRAS-CHAVE: Sites piratas que NÃO devem ser acessados; Esteganografia com Prompt Injection em imagens utilizando OCR (Reconhecimento Óptico de Caracteres)
    Descrição
        Introdução de dados que atrapalhem o resultado da AI

    Exemplo
        Introduzir imagens que atraem a AI para um contexto diferente
            Para provocar alucinações

Envenenamento de dados -> PALAVRAS-CHAVE: Dados de treinamento
    Descrição
        Manipulação de dados de treinamento

    Exemplo
        Fornecer avaliações falsas ao classificar os resultados de um relatório de teste gerado por AI

Geração de código malicioso -> PALAVRAS-CHAVE: Backdoors (chamadas de comandos externos); Uso
    Descrição
        Manipulação de um LLM para gerar backdoors (chamadas de comandos externos) durante o uso

    Exemplo
        Gerar um código para abrir um canal de comunicação com um IP específico e malicioso

```
### Mitigações 
```
Minimização, anonimização e pseudonimização de dados
Treinamento de recursos
Manter-se atualizado com as práticas recomendadas de segurança
```
## 3.3 Consumo de energia e impacto ambiental da AI generativa em teste de software
### 
```
Impacto no consumo de energia e emissões de CO2 
    Alto consumo de energia devido
        Complexidade da tarefa e os recursos computacionais necessários

        Efeito cumulativo com milhões de usuários

    Práticas de mitigação
        Limitação de interações desnecessárias entre modelos
```
##  3.4 Regulamentações, padrões e estruturas de AI generativa em teste de software
```
Norma "ISO/IEC 42001:2023 TI - IA - Sistema de gerenciamento" -> PALAVRAS-CHAVE: Aderência às práticas recomendadadas, Consistência, Confiabilidade
    Descrição
        Especifica os requisitos para o gerenciamento de sistemas de AI em uma organização
        
    Aplicação em teste
        Garante que a AI generativa em teste de software adere às práticas recomendadadas
            Promovendo consistência e confiabilidade

Padrão "ISO/IEC 23053:2022 Estrutura para sistemas de IA usando Machine Learning" -> PALAVRAS-CHAVE: Tolerância a falhas, Transparência
    Descrição   
        Fornece uma estrutura para processos de ciclo de vida de AI
            Enfatizando a tolerância a falhas e a transparência

    Aplicação em teste
        Fornece uma estrutura para qualidade de dados, transparência e tolerância a falhas ao usar a AI generativa para teste de software

Regulamento "Lei de AI da UE" -> PALAVRAS-CHAVE: Classificação/Níveis de risco, Obrigação legal
    Descrição
        Estabelece uma estrutura legal que aborda os riscos de AI
            Classificando os aplicativos por nível de risco

    Aplicação em teste
        Obriga a conformidade com a transparência, a responsabilidade e mitigação de vieses para a AI generativa usada em teste de software

Estrutura "Estrutura de gerenciamento de riscos de AI do NIST AI RMF (EUA)" -> PALAVRAS-CHAVE: Imparcialidade, Justiça, Segurança, Evitar resultados tendenciosos 
    Descrição
        Oferece diretrizes para o gerenciamento de riscos de AI
            Com foco em justiça, transparência e segurança

    Aplicação em teste
        Garante a imparcialidade e reduz riscos de AI generativa
            Evitando resultados de testes tendenciosos
```
# 4. INFRAESTRUTURA DE TESTE COM TECNOLOGIA LLM PARA TESTE DE SOFTWARE
## 4.1 Abordagens Arquitetônicas para Infraestrutura de Teste com Tecnologia LLM 
```
Ampliam a funcionalidade e a utilidade do uso de LLM
    Geração aumentada por recuperação (RAG)
        Ex.: NotebookLM

    Arquiteturas de agentes com tecnologia LLM
``` 
### 4.1.1 Principais componentes e conceitos arquitetônicos da infraestrutura de teste com tecnologia LLM
```
Componentes da infraestrutura de teste com LLM 
    Front-end
    Back-end
    LLM integrado
    Fonte de dados externas
```  
## 4.2 Ajuste Fino e LLMOps: Operacionalização da AI Generativa para teste de software
```
Ajuste fino 
    Adaptar LLM/SLM pré-treinado para executar tarefas específicas

Gerenciamento do pipeline operacional (LLMOps)
    Conjunto de práticas, ferramentas e processos projetados para simplificar o desenvolvimento, a implementação e a manutenção de LLMs em ambientes de produção

```
# 5. IMPLEMENTAÇÃO E INTEGRAÇÃO DA AI GENERATIVA EM ORGANIZAÇÕES DE TESTE 
```
AI invisível (Shadow AI)
    Uso de ferramentas ou sistemas GenAI em uma organização sem aprovação ou supervisão formal
```
## 5.1.4 Fases da adoção da AI generativa em testes de software 
### Des - In & Def Us - Ut & It
```
"Des"coberta
"In"iciação e "Def"inição de "Us"o
"Ut"ilização e "It"eração
```
## 5.2.3 Evolução dos processos de teste em organizações de teste habilitadas para IA 
```
Com a adoção da AI generativa
    O papel de um testador evolui de especialista em projeto e execução de testes para especialista em testes assistidos por AI
    O gerente de teste se concentra na supervisão de equipes híbridas (humanos e agentes de testes GenAI)
```
